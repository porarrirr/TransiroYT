"""Whisper large-v3-turbo transcripts of reference and generated speech, on MLX.

One sequential, persistent worker; run with the TTS Python. The model directory
is in TRANSIROYT_WHISPER_MODEL. stdout is a JSON-lines protocol: one ready line
after loading, then one reply per request {"pcm": base64 s16le 16 kHz mono,
"max_tokens": int, "language": str}, or {"audio": path of a 16 kHz mono WAV,
"timestamps": true, "language": str | null} for subtitles with word times.
With "windows": [[start, end], ...] and "language", the windows of the WAV are
transcribed together ("texts", "complete"); with "detect": [[start, end], ...]
their spoken language is identified ("language", "probabilities"); with
"pauses": minimum seconds, the silent stretches of separated speech are listed.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import time
import traceback
import wave

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 30 * SAMPLE_RATE
# The decoder has 448 positions; the prompt takes 4 of them.
MAX_TOKENS = 444
# Windows decoded together: the decoder step costs about the same for one
# sequence or eight, the encoder grows linearly (measured on an M2).
WINDOW_BATCH = 8
PAUSE_HOP = 160


def import_whisper():
    # The package __init__ imports transcribe -> timing (numba, SciPy): about
    # 1 s of startup for word timestamps, which this worker never uses.
    # Register the package without executing it; submodules import normally.
    if "mlx_whisper" not in sys.modules:
        spec = importlib.util.find_spec("mlx_whisper")
        sys.modules["mlx_whisper"] = importlib.util.module_from_spec(spec)
    from mlx_whisper import audio, load_models, tokenizer
    return audio, load_models, tokenizer


class Runtime:
    def __init__(self, directory):
        self.directory = directory
        self.audio, load_models, tokenizer = import_whisper()
        self.model = load_models.load_model(directory, dtype=mx.float16)
        self.get_tokenizer = tokenizer.get_tokenizer
        initial_tokenizer = self.get_tokenizer(True, num_languages=self.model.num_languages,
                                               language="ja", task="transcribe")
        # Everything after <|endoftext|> is a special or timestamp token. The
        # first step also suppresses <|endoftext|>, as Whisper's SuppressBlank does.
        vocabulary = mx.arange(self.model.dims.n_vocab)
        eot = initial_tokenizer.eot
        self.suppress = mx.where(vocabulary > eot, -mx.inf, 0.0)
        self.suppress_first = mx.where(vocabulary >= eot, -mx.inf, 0.0)
        # Build the GPU kernels before the first request (about 1 s otherwise).
        self.transcribe(bytes(SAMPLE_RATE * 2), 2)

    def transcribe(self, pcm: bytes, max_tokens: int, language="ja"):
        tokenizer = self.get_tokenizer(True, num_languages=self.model.num_languages,
                                       language=language, task="transcribe")
        prompt = mx.array([list(tokenizer.sot_sequence_including_notimestamps)])
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        texts, complete = [], True
        for start in range(0, max(1, len(samples)), WINDOW_SAMPLES):
            text, finished = self._window(samples[start:start + WINDOW_SAMPLES], max_tokens, prompt, tokenizer)
            texts.append(text)
            complete = complete and finished
        return "".join(texts).strip(), complete

    def transcribe_timed(self, path, language=None):
        """Whole-programme subtitles; a null language is detected from the first 30 s."""
        # Imports numba and SciPy (word timestamps) only when a video has no subtitles.
        whisper = importlib.import_module("mlx_whisper.transcribe")
        # Share the loaded weights instead of a second 1.6 GB copy.
        whisper.ModelHolder.model, whisper.ModelHolder.model_path = self.model, self.directory
        samples = self._samples(path)
        # Without the previous window as a prompt, a hallucinated line cannot
        # repeat through the rest of the programme; silent stretches with
        # isolated words (music, breathing) are skipped.
        result = whisper.transcribe(samples, path_or_hf_repo=self.directory, language=language,
                                    word_timestamps=True, condition_on_previous_text=False,
                                    hallucination_silence_threshold=2.0, verbose=None)
        return {"language": result["language"],
                "segments": [{"start": s["start"], "end": s["end"], "text": s["text"],
                              "words": [{"start": w["start"], "end": w["end"], "word": w["word"]}
                                        for w in s.get("words", [])]}
                             for s in result["segments"]]}

    def transcribe_windows(self, path, windows, language):
        """Text of each [start, end] window (at most 30 s) of a WAV, without timestamps."""
        samples = self._samples(path)
        tokenizer = self.get_tokenizer(True, num_languages=self.model.num_languages,
                                       language=language, task="transcribe")
        texts, complete = [], []
        for first in range(0, len(windows), WINDOW_BATCH):
            chunks = [samples[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)]
                      for start, end in windows[first:first + WINDOW_BATCH]]
            for output, finished in self._decode(self._features(chunks), tokenizer, MAX_TOKENS):
                texts.append(tokenizer.decode(output).strip())
                complete.append(finished)
        return texts, complete

    def detect_language(self, path, windows):
        """Whisper's language probabilities summed over the windows."""
        samples = self._samples(path)
        features = self._features([samples[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)]
                                   for start, end in windows])
        tokenizer = self.get_tokenizer(True, num_languages=self.model.num_languages)
        logits = self._step(mx.array([[tokenizer.sot]] * len(windows)), 0, self._cross(features), None)
        languages = mx.array(list(tokenizer.all_language_tokens))
        totals = mx.softmax(logits[:, languages], axis=-1).sum(axis=0) / len(windows)
        ranked = sorted(zip(tokenizer.all_language_codes, totals.tolist(), strict=True), key=lambda item: -item[1])
        return ranked[0][0], {code: round(p, 4) for code, p in ranked[:5]}

    def pauses(self, path, minimum):
        """[start, end] of stretches at least ``minimum`` seconds long, 30 dB below loud speech."""
        samples = self._samples(path)
        frames = samples[:len(samples) // PAUSE_HOP * PAUSE_HOP].reshape(-1, PAUSE_HOP)
        level = 10 * np.log10(np.square(frames).mean(axis=1) + 1e-10)
        silent = np.concatenate([[False], level < np.percentile(level, 90) - 30, [False]])
        edges = np.flatnonzero(np.diff(silent.astype(np.int8)))
        seconds = PAUSE_HOP / SAMPLE_RATE
        return [[round(start * seconds, 2), round(end * seconds, 2)]
                for start, end in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True)
                if (end - start) * seconds >= minimum]

    def _samples(self, path):
        # A programme's windows arrive in several requests; read the WAV once.
        if getattr(self, "_audio", (None,))[0] != path:
            with wave.open(path, "rb") as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
                    raise ValueError("expected 16 kHz mono 16-bit PCM")
                self._audio = path, np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        return self._audio[1]

    def _features(self, chunks):
        audio = self.audio
        mels = [audio.pad_or_trim(audio.log_mel_spectrogram(chunk, n_mels=self.model.dims.n_mels),
                                  audio.N_FRAMES, axis=-2) for chunk in chunks]
        return self._encode(mx.stack(mels).astype(mx.float16))

    def _window(self, samples, max_tokens, prompt, tokenizer):
        [(output, finished)] = self._decode(self._features([samples]), tokenizer, max_tokens, prompt)
        return tokenizer.decode(output), finished

    def _decode(self, features, tokenizer, max_tokens, prompt=None):
        """Greedy decoding of every window in the batch; each stops at its own <|endoftext|>."""
        batch = features.shape[0]
        cross = self._cross(features)
        own = [None] * len(cross)
        tokens = prompt if prompt is not None else mx.array(
            [list(tokenizer.sot_sequence_including_notimestamps)] * batch)
        offset, outputs, finished = 0, [[] for _ in range(batch)], [False] * batch
        for step in range(max_tokens):
            logits = self._step(tokens, offset, cross, own)
            offset += tokens.shape[1]
            chosen = mx.argmax(logits + (self.suppress_first if step == 0 else self.suppress), axis=-1).tolist()
            for index, output in enumerate(outputs):
                if finished[index]:
                    continue
                if chosen[index] == tokenizer.eot:
                    finished[index] = True
                else:
                    output.append(chosen[index])
            if all(finished):
                break
            tokens = mx.array([[token] for token in chosen])
        # A take that loops on unintelligible audio never reaches <|endoftext|>.
        return list(zip(outputs, finished, strict=True))

    def _cross(self, features):
        return [(block.cross_attn.key(features), block.cross_attn.value(features))
                for block in self.model.decoder.blocks]

    def _step(self, tokens, offset, cross, own):
        """Logits after ``tokens``; ``own`` holds each layer's self-attention keys and values."""
        decoder = self.model.decoder
        x = (decoder.token_embedding(tokens)
             + decoder.positional_embedding[offset:offset + tokens.shape[1]])
        for layer, block in enumerate(decoder.blocks):
            h = block.attn_ln(x)
            k, v = block.attn.key(h), block.attn.value(h)
            if own is not None:
                if own[layer] is not None:
                    k = mx.concatenate([own[layer][0], k], axis=1)
                    v = mx.concatenate([own[layer][1], v], axis=1)
                own[layer] = (k, v)
            x = x + attend(block.attn, block.attn.query(h), k, v,
                           "causal" if tokens.shape[1] > 1 else None)
            h = block.cross_attn_ln(x)
            x = x + attend(block.cross_attn, block.cross_attn.query(h), *cross[layer])
            x = x + block.mlp2(nn.gelu(block.mlp1(block.mlp_ln(x))))
        return decoder.token_embedding.as_linear(decoder.ln(x[:, -1])).astype(mx.float32)

    def _encode(self, mel):
        encoder = self.model.encoder
        x = nn.gelu(encoder.conv1(mel))
        x = nn.gelu(encoder.conv2(x))
        x = x + encoder._positional_embedding
        for block in encoder.blocks:
            h = block.attn_ln(x)
            x = x + attend(block.attn, block.attn.query(h), block.attn.key(h), block.attn.value(h))
            x = x + block.mlp2(nn.gelu(block.mlp1(block.mlp_ln(x))))
        return encoder.ln_post(x)


def attend(attention, q, k, v, mask=None):
    """mlx_whisper's attention, through the fused kernel instead of an explicit
    1500 x 1500 score matrix per head (the encoder's largest cost)."""
    heads = attention.n_head
    batch, length, width = q.shape
    q, k, v = (t.reshape(batch, t.shape[1], heads, -1).transpose(0, 2, 1, 3) for t in (q, k, v))
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=(width // heads) ** -0.5, mask=mask)
    return attention.out(out.transpose(0, 2, 1, 3).reshape(batch, length, width))


def main():
    started = time.perf_counter()
    try:
        mx.set_cache_limit(256 * 1024 * 1024)
        runtime = Runtime(os.environ["TRANSIROYT_WHISPER_MODEL"])
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        return 1
    print(json.dumps({"ok": True, "ready": True,
                      "load_seconds": round(time.perf_counter() - started, 3)}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            begun = time.perf_counter()
            if "windows" in request:
                texts, complete = runtime.transcribe_windows(request["audio"], request["windows"], request["language"])
                reply = {"ok": True, "texts": texts, "complete": complete}
            elif "pauses" in request:
                reply = {"ok": True, "pauses": runtime.pauses(request["audio"], float(request["pauses"]))}
            elif "detect" in request:
                language, probabilities = runtime.detect_language(request["audio"], request["detect"])
                reply = {"ok": True, "language": language, "probabilities": probabilities}
            elif request.get("timestamps"):
                reply = {"ok": True, **runtime.transcribe_timed(request["audio"], request.get("language"))}
            else:
                text, complete = runtime.transcribe(base64.b64decode(request["pcm"]), int(request["max_tokens"]), request.get("language", "ja"))
                reply = {"ok": True, "text": text, "complete": complete}
            reply["seconds"] = round(time.perf_counter() - begun, 3)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = {"ok": False, "error": str(exc)}
        print(json.dumps(reply, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

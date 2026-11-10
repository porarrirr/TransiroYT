"""Identify subtitle speakers and collect per-speaker reference audio.

Nemotron identifies speech activity; ECAPA compares voice identity. Caption
speaker markers define whole turns. Whisper verifies reference transcripts."""
from __future__ import annotations

import array
import asyncio
import io
import json
import logging
import os
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

from .caption_annotations import reaction_emojis
from .captions import Cue
from .native_audio import frame_energies
from .speech_check import transcript_ratio
from .tts import tts_python

logger = logging.getLogger(__name__)

ANALYSIS_RATE = 16000
EMBEDDING_MIN_SECONDS = 0.4
ANCHOR_MIN_SECONDS = 1.2
HALVES_MIN_SECONDS = 2.5
# Average-linkage cosine threshold, tuned on the synthetic and BBC test media.
MERGE_THRESHOLD = 0.3
# Acoustic decisions were correct on every real-voice line down to margin 0.047
# (0.5-1.1s fragments). A line whose top two voices are practically tied keeps
# its label but is not used as reference audio ("acoustic_uncertain").
AMBIGUOUS_MARGIN = 0.03
# A speaker is active in a 10 ms frame above this probability (the model's own
# segment threshold). A line takes the speaker with the most activity over it,
# if that speaker is active somewhere in it; a second active speaker marks the
# line as overlapped speech ("acoustic_uncertain"). At 0.3 the same 176 lines
# had 31 in another person's voice and 4 across genders, against 15 and 0.
ACTIVE_PROBABILITY = 0.5
# Kept in step with scripts/speaker_activity.py, which runs in the TTS Python.
ACTIVITY_MODEL_REPO = "nvidia/Nemotron-3-Diarization"
ACTIVITY_MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
ACTIVITY_MODEL_FILES = ("config.json", "processor_config.json", "model.safetensors", "README.md")
# Irodori's speaker encoder takes about two minutes of reference. On a 57-minute
# programme a 60 s reference brought the dub closer to the speaker (ECAPA 0.68
# -> 0.78) with fewer garbled takes (4/12 -> 0/12) than 12.6 s, for 9% more
# generation time. Counted as voiced speech (AnalysisAudio.voiced):
# on two programmes, 60 s of voice beat the old 60 s of whole lines (ECAPA +0.009
# and +0.035 on 8 and 16 lines); 30 s was worse in both and 90 s not better in one.
REFERENCE_TARGET_SECONDS = 60.0
REFERENCE_MAX_SECONDS = 70.0
REFERENCE_MIN_SECONDS = 2.0
# A speaker with too little transcript-verified audio (people heard for a few
# seconds) is cloned from their own remaining clean clips, marked unverified.
OWN_VOICE_MIN_SECONDS = 1.0
OWN_VOICE_CLIP_SECONDS = 0.6
# Near-tie lines keep their voice label and are dubbed with that speaker's
# reference, so they are that speaker's own clips too: never
# verified reference, but usable (after confident clips) when nothing else is.
# A montage of one-off reactions (C782A726) labelled 28 of its 37 brief
# speakers only on such lines; excluding them left 89 lines in the original voice.
REFERENCE_SOURCES = {"acoustic", "single_speaker", "acoustic_uncertain"}
# Rejections that still leave the speaker's own voice, short clips included.
OWN_VOICE_REJECTIONS = {"atypical_voice", "uncertain_speaker"}
CANDIDATE_MIN_SECONDS = 1.5
CANDIDATE_MAX_SECONDS = 15.0
TRANSCRIPT_MIN_RATIO = 0.75
# A clip far from its speaker's voice is not used as verified reference. One
# 1.2 s clip at 0.49 among ten at 0.77-0.88 (A051D6A3; it led on its 74 dB SNR)
# halved the dubbed takes that passed the transcript check (58/76 -> 35/74).
REFERENCE_MIN_SCORE = 0.6
ADJACENT_SECONDS = 0.1
EDGE_SECONDS = 0.08
# Reference clips keep only the voice of each verified line (AnalysisAudio.voiced).
VOICE_FRAME_SECONDS = 0.02
VOICE_RANGE_DB = 30.0
VOICE_PAUSE_SECONDS = 0.3
VOICE_PAD_SECONDS = 0.05
# A lone click or breath is not speech.
VOICE_MIN_SECONDS = 0.1


@dataclass
class SpeakerLine:
    index: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    score: float | None = None
    margin: float | None = None
    source: str = "unassigned"
    acoustic_speaker: str | None = None
    # The subtitle cues of this turn, (start, end, text), for sentence timing.
    parts: list[tuple[float, float, str]] = field(default_factory=list)
    boundary_before: bool = False
    speaker_label: str | None = None

    @property
    def duration(self):
        return self.end - self.start

    def public(self):
        return {"index": self.index, "start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text, "speaker": self.speaker,
                "score": None if self.score is None else round(self.score, 4),
                "margin": None if self.margin is None else round(self.margin, 4),
                "source": self.source, "acoustic_speaker": self.acoustic_speaker,
                "boundary_before": self.boundary_before, "speaker_label": self.speaker_label,
                "parts": [[round(a, 3), round(b, 3), text] for a, b, text in self.cues()]}

    def cues(self) -> list[tuple[float, float, str]]:
        return self.parts or [(self.start, self.end, self.text)]


class Similarity:
    """Row-major float32 cosine matrix written by scripts/speaker_embedding.py."""

    def __init__(self, values: array.array, size: int):
        if len(values) != size * size:
            raise ValueError("speaker similarity matrix has an unexpected size")
        self.values, self.size = values, size

    def __call__(self, a: int, b: int) -> float:
        return self.values[a * self.size + b]


class EmbeddingWorker:
    def __init__(self, settings, *, allow_download=False, download_progress=None):
        self.settings, self.allow_download = settings, allow_download
        self.download_progress = download_progress

    async def similarity(self, audio: Path, items: list[tuple[float, float]], work: Path):
        if self.allow_download and self.download_progress:
            from .model_downloads import download_model
            await download_model(self.settings, "speaker_model", self.download_progress)
        output = work / "speaker-similarity.f32"
        request = {"audio": str(audio), "cache": str(self.settings.data / "models/huggingface"),
                   "download": self.allow_download and not self.download_progress, "items": items, "output": str(output)}
        response = await self._run("speaker_embedding.py", request, "話者埋め込みの計算に失敗しました")
        values = array.array("f")
        values.frombytes(output.read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        output.unlink(missing_ok=True)
        return Similarity(values, len(items)), response["stats"]

    async def _run(self, script: str, request: dict, failure: str) -> dict:
        env = {**os.environ, **({"PYTHONPATH": os.environ["TRANSIROYT_TTS_SITE_PACKAGES"]}
                                if "TRANSIROYT_TTS_SITE_PACKAGES" in os.environ else {})}
        process = await asyncio.create_subprocess_exec(
            str(tts_python(self.settings)), "-B", str(self.settings.root / "scripts" / script),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env)
        stdout, stderr = await asyncio.wait_for(process.communicate(json.dumps(request).encode()), 1800)
        try:
            response = json.loads(stdout.decode().strip().splitlines()[-1])
        except (ValueError, IndexError):
            response = {"ok": False, "error": stderr.decode(errors="replace")[-600:]}
        if process.returncode or not response.get("ok"):
            raise RuntimeError(f"{failure}: {response.get('error')}")
        return response

    async def activity(self, audio: Path, work: Path) -> SpeakerActivity:
        if self.allow_download and self.download_progress:
            from .model_downloads import download_model
            await download_model(self.settings, "diarization_model", self.download_progress)
        output = work / "speaker-activity.f32"
        request = {"audio": str(audio), "cache": str(self.settings.data / "models/huggingface"),
                   "download": self.allow_download and not self.download_progress, "output": str(output)}
        response = await self._run("speaker_activity.py", request, "話者の区間の判定に失敗しました")
        values = array.array("f")
        values.frombytes(output.read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        output.unlink(missing_ok=True)
        return SpeakerActivity(values, response["speakers"])


class SpeakerActivity:
    """Per-speaker activity probabilities, one row per 10 ms frame, written by scripts/speaker_activity.py."""

    FRAME_SECONDS = 0.01

    def __init__(self, values: array.array, speakers: int):
        if speakers < 1 or len(values) % speakers:
            raise ValueError("speaker activity has an unexpected size")
        self.values, self.speakers = values, speakers
        self.frames = len(values) // speakers

    def over(self, start: float, end: float) -> list[tuple[float, float]]:
        """(summed probability, peak probability) of each speaker over the interval."""
        first = min(self.frames - 1, max(0, int(start / self.FRAME_SECONDS)))
        last = min(self.frames, max(first + 1, int(end / self.FRAME_SECONDS)))
        rows = [self.values[frame * self.speakers:(frame + 1) * self.speakers] for frame in range(first, last)]
        return [(sum(row[k] for row in rows), max(row[k] for row in rows)) for k in range(self.speakers)]


class AnalysisAudio:
    """16 kHz mono PCM of the programme, sliced for context and reference checks."""

    def __init__(self, path: Path):
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, ANALYSIS_RATE):
                raise ValueError("analysis audio must be 16 kHz mono 16-bit")
            self.frames = wav.readframes(wav.getnframes())
        self.path = path
        self.duration = len(self.frames) / 2 / ANALYSIS_RATE

    def voiced(self, start: float, end: float) -> list[tuple[float, float]]:
        """The voice inside [start, end]: 20 ms frames within 30 dB of the clip's loud speech.

        Subtitle times include lead-in silence, trailing breaths and pauses.
        Gaps up to 0.3 s stay inside a span; longer pauses are left out.
        """
        size = round(VOICE_FRAME_SECONDS * ANALYSIS_RATE)
        first = max(0, round(start * ANALYSIS_RATE))
        energy = frame_energies(self.frames, size, start=first,
                                stop=round(end * ANALYSIS_RATE), partial=False)
        loud = sorted(energy)[int(0.95 * (len(energy) - 1))] if energy else 0
        if not loud:
            return []
        floor = loud * 10 ** (-VOICE_RANGE_DB / 10)
        spans = []
        for number, value in enumerate(energy):
            if value < floor:
                continue
            a = (first + number * size) / ANALYSIS_RATE
            if spans and a - spans[-1][1] <= VOICE_PAUSE_SECONDS:
                spans[-1][1] = a + VOICE_FRAME_SECONDS
            else:
                spans.append([a, a + VOICE_FRAME_SECONDS])
        return [(round(max(start, a - VOICE_PAD_SECONDS), 3), round(min(end, b + VOICE_PAD_SECONDS), 3))
                for a, b in spans if b - a >= VOICE_MIN_SECONDS]

    def wav(self, start: float, end: float) -> bytes:
        a = max(0, round(start * ANALYSIS_RATE)) * 2
        b = min(len(self.frames), round(end * ANALYSIS_RATE) * 2)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(ANALYSIS_RATE)
            wav.writeframes(self.frames[a:b])
        return buffer.getvalue()


def cluster(anchors: list[int], similarity, threshold: float, speakers: int | None = None) -> list[list[int]]:
    """Average-linkage agglomerative clustering, cut like a dendrogram.

    The nearest-neighbour chain builds the complete hierarchy in O(n^2). Average
    linkage is reducible, so replaying merges from the most similar downwards
    gives the same flat clusters as greedy merging. With a known speaker count,
    cut at that count; otherwise at the similarity threshold.
    """
    if not anchors:
        return []
    link = {a: {b: similarity(anchors[a], anchors[b]) for b in range(len(anchors)) if b != a}
            for a in range(len(anchors))}
    size = {a: 1 for a in link}
    merges = []
    chain: list[int] = []
    while len(link) > 1:
        if not chain:
            chain.append(next(iter(link)))
        current = chain[-1]
        previous = chain[-2] if len(chain) > 1 else None
        best = max(link[current], key=lambda other: (link[current][other], other == previous))
        if best != previous:
            chain.append(best)
            continue
        chain.pop()
        chain.pop()
        height = link[current][best]
        keep, gone = min(current, best), max(current, best)
        merges.append((height, keep, gone))
        for other in list(link[keep]):
            if other == gone:
                continue
            value = (link[keep][other] * size[keep] + link[gone][other] * size[gone]) / (size[keep] + size[gone])
            link[keep][other] = link[other][keep] = value
        for other in link.pop(gone):
            link[other].pop(gone, None)
        link[keep].pop(gone, None)
        size[keep] += size.pop(gone)
    parent = list(range(len(anchors)))

    def root(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    merges.sort(key=lambda merge: merge[0], reverse=True)
    allowed = len(anchors) - max(1, speakers) if speakers else None
    for count, (height, a, b) in enumerate(merges):
        if (allowed is not None and count >= allowed) or (allowed is None and height < threshold):
            break
        parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for position, item in enumerate(anchors):
        groups.setdefault(root(position), []).append(item)
    return sorted(groups.values(), key=lambda members: min(members))


def speaker_scores(item: int, groups: dict[str, list[int]], similarity, *, exclude_self=True) -> dict[str, float]:
    """Mean of the three closest members: robust to one atypical anchor clip."""
    scores = {}
    for label, members in groups.items():
        values = sorted((similarity(item, member) for member in members
                         if not (exclude_self and member == item)), reverse=True)
        if values:
            scores[label] = sum(values[:3]) / len(values[:3])
    return scores


def speaker_turns(cues: list[Cue]) -> list[list[Cue]]:
    """Group captions into turns at speaker markers; unmarked captions stay single."""
    # Named labels can be sparse (MrBeast labels Karl and Jimmy only once).
    # Only the >> convention promises that unmarked captions continue a turn.
    if not any(cue.speaker_change and cue.speaker_label is None for cue in cues):
        return [[cue] for cue in cues]
    turns: list[list[Cue]] = []
    for cue in cues:
        if cue.speaker_change or not turns or turns[-1][0].speaker_label is not None:
            turns.append([cue])
        else:
            turns[-1].append(cue)
    return turns


class SpeakerDiarizer:
    def __init__(self, embeddings, *, transcriber=None, language="en",
                 speakers: int | None = None, progress=None):
        self.embeddings = embeddings
        self.transcriber, self.language = transcriber, language
        self.speakers = speakers
        self.progress = progress or (lambda *_: None)
        self.reference_transcriptions = 0

    async def run(self, cues: list[Cue], audio: AnalysisAudio, work: Path) -> list[SpeakerLine]:
        lines, ambiguous = await self.assign_voices(cues, audio, work)
        for line in ambiguous:
            if line.speaker is not None:
                line.source = "acoustic_uncertain"
        rejected = set()
        for line in lines:
            # A caption explicitly starts a named turn; an unrelated dominant
            # voice must not supply its reference when ECAPA cannot match it.
            if line.speaker_label and (line.score is None or line.score < MERGE_THRESHOLD):
                line.speaker, line.source = None, "caption_voice_unverified"
                if line.index in self.item_of:
                    rejected.add(self.item_of[line.index])
        if rejected:
            self.groups = {name: [item for item in members if item not in rejected]
                           for name, members in self.groups.items()}
        return lines

    async def assign_voices(self, cues: list[Cue], audio: AnalysisAudio, work: Path):
        """Speaker of each turn from the voice alone; returns (lines, lines the voice cannot decide)."""
        lines = [SpeakerLine(i, turn[0].start, turn[-1].end, " ".join(cue.text for cue in turn),
                             parts=[(cue.start, cue.end, cue.text) for cue in turn],
                             boundary_before=turn[0].speaker_change, speaker_label=turn[0].speaker_label)
                 for i, turn in enumerate(speaker_turns(cues))]
        embeddable = [line for line in lines if line.duration >= EMBEDDING_MIN_SECONDS
                      and line.end <= audio.duration + 0.05]
        if not embeddable:
            raise RuntimeError("話者を判定できる長さの字幕区間がありません。")
        items = [(line.start, min(line.end, audio.duration)) for line in embeddable]
        self.item_of = {line.index: n for n, line in enumerate(embeddable)}
        # Halves detect a clip containing a second voice; only reference candidates
        # need them (_reference_for): not over-long or captioned-reaction lines,
        # and never when one speaker was requested.
        self.halves = {}
        for line in embeddable:
            if (HALVES_MIN_SECONDS <= line.duration <= CANDIDATE_MAX_SECONDS
                    and self.speakers != 1 and not reaction_emojis(line.text)):
                middle = (line.start + line.end) / 2
                self.halves[line.index] = (len(items), len(items) + 1)
                items += [(line.start, middle), (middle, min(line.end, audio.duration))]
        self.progress("speaker_embedding", 0.0)
        if self.speakers:
            self.similarity, self.stats = await self.embeddings.similarity(audio.path, items, work)
            return self._cluster_voices(lines, embeddable)
        (self.similarity, self.stats), activity = await asyncio.gather(
            self.embeddings.similarity(audio.path, items, work), self.embeddings.activity(audio.path, work))
        return self._follow_activity(lines, embeddable, activity)

    def _follow_activity(self, lines, embeddable, activity):
        """Each line's speaker from Nemotron's activity; ECAPA scores how typical each clip is of its speaker."""
        found, overlapped = {}, []
        for line in lines:
            scores = activity.over(line.start, line.end)
            best = max(range(len(scores)), key=lambda k: scores[k][0])
            if scores[best][1] < ACTIVE_PROBABILITY:
                continue  # No one heard: dubbed from the line's own voice (FileDubber._voices).
            found[line.index] = best
            if any(peak >= ACTIVE_PROBABILITY for k, (_, peak) in enumerate(scores) if k != best):
                overlapped.append(line)
        # Label in order of first appearance, as the model numbers its speakers.
        names = {}
        for line in lines:
            if line.index in found:
                names.setdefault(found[line.index], f"S{len(names) + 1}")
        members = {name: [] for name in names.values()}
        for line in embeddable:
            if line.index in found and line.duration >= ANCHOR_MIN_SECONDS:
                members[names[found[line.index]]].append(self.item_of[line.index])
        self.groups = members
        for line in lines:
            if line.index not in found:
                continue
            line.speaker = line.acoustic_speaker = names[found[line.index]]
            line.source = "acoustic"
            item = self.item_of.get(line.index)
            if item is not None:
                scores = speaker_scores(item, self.groups, self.similarity)
                line.score = scores.get(line.speaker)
                others = [score for other, score in scores.items() if other != line.speaker]
                if line.score is not None:
                    line.margin = line.score - max(others, default=-1.0)
        return lines, [line for line in lines if line.speaker is None] + overlapped

    def _cluster_voices(self, lines, embeddable):
        """ECAPA clusters cut at the requested speaker count."""
        anchors = [self.item_of[line.index] for line in embeddable if line.duration >= ANCHOR_MIN_SECONDS]
        if not anchors:
            anchors = [self.item_of[line.index] for line in embeddable]
        groups = cluster(anchors, self.similarity, MERGE_THRESHOLD, self.speakers)
        # Label in order of first appearance; the largest speaking time is not assumed to be a host.
        self.groups = {f"S{n + 1}": members for n, members in enumerate(groups)}
        cluster_of = {item: label for label, members in self.groups.items() for item in members}
        for line in embeddable:
            item = self.item_of[line.index]
            scores = speaker_scores(item, self.groups, self.similarity)
            label = cluster_of.get(item)
            if label is None:  # Short line: nearest speaker by its closest clips.
                ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
                label, own = ranked[0]
            else:
                # Clustered clips keep the clustering decision. A one-clip speaker
                # has no other member to score against; its distance to the other
                # voices relative to the merge threshold stands in for the margin.
                own = scores.get(label)
            others = [score for other, score in scores.items() if other != label]
            best_other = max(others, default=-1.0)
            line.speaker = line.acoustic_speaker = label
            line.score = own
            line.margin = (own if own is not None else MERGE_THRESHOLD) - best_other
            line.source = "acoustic"
        ambiguous = [line for line in lines if line.speaker is None or (
            len(self.groups) > 1 and line.margin < AMBIGUOUS_MARGIN)]
        if len(self.groups) == 1:
            for line in lines:
                if line.speaker is None:
                    line.speaker, line.source = "S1", "single_speaker"
            ambiguous = []
        return lines, ambiguous

    async def references(self, lines: list[SpeakerLine], audio: AnalysisAudio) -> dict[str, dict]:
        """Choose clean single-speaker clips per speaker and verify them with Whisper.

        Returns accepted clip intervals; the caller cuts them from full-quality audio.
        """
        speakers = sorted({line.speaker for line in lines if line.speaker})
        # One Whisper worker checks references sequentially, stopping as soon
        # as each speaker has enough verified audio.
        limit = asyncio.Semaphore(1)
        completed = 0
        self.progress("reference_selection", 0.0)

        async def select(speaker):
            nonlocal completed
            async with limit:
                reference = await self._reference_for(speaker, lines, audio)
                completed += 1
                self.progress("reference_selection", completed / max(1, len(speakers)))
                return reference

        tasks = [asyncio.create_task(select(speaker)) for speaker in speakers]
        try:
            selected = await asyncio.gather(*tasks)
            return dict(zip(speakers, selected, strict=True))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if completed == len(speakers):
                self.progress("reference_selection", 1.0)

    async def _reference_for(self, speaker, lines, audio):
        candidates = self._reference_candidates(speaker, lines)
        accepted, rejected, total = [], [], 0.0
        unchecked = []
        for line, start, end, rejection, snr in candidates:
            if line.duration < CANDIDATE_MIN_SECONDS:
                if rejection is None or rejection in OWN_VOICE_REJECTIONS:
                    unchecked.append((line, start, end, snr))
                else:
                    rejected.append({"line": line.index, "start": round(line.start, 3),
                                     "end": round(line.end, 3), "reason": rejection})
                continue
            if total >= REFERENCE_TARGET_SECONDS:
                if rejection is None:
                    unchecked.append((line, start, end, snr))
                continue
            if rejection is None:
                if total + end - start > REFERENCE_MAX_SECONDS:
                    continue
                self.reference_transcriptions += 1
                heard = await self.transcriber.transcribe(audio.wav(start, end), line.text, language=self.language)
                ratio = transcript_ratio(line.text, heard)
                voiced = audio.voiced(start, end)
                if ratio < TRANSCRIPT_MIN_RATIO:
                    rejection = "transcript_mismatch"
                elif not voiced:
                    rejection = "too_quiet"
                entry = {"line": line.index, "start": round(start, 3), "end": round(end, 3),
                         "text": line.text, "heard": heard[:400],
                         "transcript_ratio": round(ratio, 3), "score": round(line.score or 0, 4),
                         "snr_db": round(snr, 1)}
                if rejection is None:
                    accepted.append({**entry, "verified": True, "voiced": voiced})
                    total += sum(b - a for a, b in voiced)
                    continue
                rejected.append({**entry, "reason": rejection})
                if rejection == "transcript_mismatch":
                    unchecked.append((line, start, end, snr))
            else:
                rejected.append({"line": line.index, "start": round(line.start, 3),
                                 "end": round(line.end, 3), "reason": rejection})
                if rejection in OWN_VOICE_REJECTIONS:
                    unchecked.append((line, start, end, snr))
        verified = total >= REFERENCE_MIN_SECONDS
        if not verified:
            # Someone heard for a few seconds rarely has 2 s that pass the
            # transcript check. Their own clean clips still carry their voice;
            # another speaker's voice is never used. Marked unverified.
            unchecked.sort(key=lambda item: (item[0].source == "acoustic_uncertain", -(item[3] // 5),
                                             -(item[0].score or 0), -item[0].duration))
            for line, start, end, snr in unchecked:
                voiced = audio.voiced(start, end)
                if not voiced or total + end - start > REFERENCE_MAX_SECONDS:
                    continue
                accepted.append({"line": line.index, "start": round(start, 3), "end": round(end, 3),
                                 "text": line.text, "score": round(line.score or 0, 4),
                                 "snr_db": round(snr, 1), "verified": False, "voiced": voiced})
                total += sum(b - a for a, b in voiced)
                if total >= REFERENCE_TARGET_SECONDS:
                    break
        usable = verified or total >= OWN_VOICE_MIN_SECONDS
        return {"clips": sorted(accepted, key=lambda clip: clip["start"]),
                "rejected": rejected, "duration": total, "verified": verified, "usable": usable}

    def _reference_candidates(self, speaker, lines):
        """Rank own-voice clips after checking speaker overlap and recording quality."""
        members = self.groups.get(speaker, [])
        others = [line for line in lines if line.speaker and line.speaker != speaker]
        candidates = []
        for line in lines:
            if (line.speaker != speaker or line.source not in REFERENCE_SOURCES
                    or not OWN_VOICE_CLIP_SECONDS <= line.duration <= CANDIDATE_MAX_SECONDS
                    or line.index not in self.item_of):
                continue
            stats = self.stats[self.item_of[line.index]]
            rejection = None
            # Tightly timed subtitles touch the next speaker's line. Step inside
            # those edges; a real overlap of two voices is never a reference.
            start, end = line.start, line.end
            for other in others:
                if other.start < line.end - EDGE_SECONDS and other.end > line.start + EDGE_SECONDS:
                    rejection = "overlaps_other_speaker"
                if abs(other.end - line.start) <= ADJACENT_SECONDS:
                    start = max(start, line.start + EDGE_SECONDS)
                if abs(other.start - line.end) <= ADJACENT_SECONDS:
                    end = min(end, line.end - EDGE_SECONDS)
            if rejection:
                pass
            elif reaction_emojis(line.text):
                # Captioned laughter or a sigh is not plain speech of this voice;
                # the reference conditions every dubbed line of the speaker.
                rejection = "nonverbal_sound"
            elif stats["rms"] < 0.01:
                rejection = "too_quiet"
            elif stats["clipped"] > 0.001:
                rejection = "clipped"
            elif line.score is not None and line.score < REFERENCE_MIN_SCORE:
                rejection = "atypical_voice"
            elif line.index in self.halves and len(self.groups) > 1:
                halves = self.halves[line.index]
                own = [speaker_scores(h, {speaker: members}, self.similarity, exclude_self=False)
                       .get(speaker, -1) for h in halves]
                if min(own) < MERGE_THRESHOLD:
                    rejection = "inconsistent_halves"
            if rejection is None and line.source == "acoustic_uncertain":
                rejection = "uncertain_speaker"
            snr = stats["speech_db"] - stats["floor_db"]
            candidates.append((line, start, end, rejection, snr))
        # Cleanest recording first (5 dB bands): the TTS copies background
        # music and noise from the reference. Within a band, most typical voice.
        candidates.sort(key=lambda item: (item[3] is not None, -(item[4] // 5),
                                          -(item[0].score or 0), -item[0].duration))
        return candidates

---
license: apache-2.0
language:
- zh
- en
pipeline_tag: audio-classification
tags:
- speaker-verification
- speaker-diarization
- FunASR
- CAM++
library_name: funasr
---

<div align="center">

### ⭐ Powered by [FunASR](https://github.com/modelscope/FunASR) — please give us a GitHub Star!

This model is part of the **FunASR** ecosystem — one industrial-grade open-source toolkit for **ASR · VAD · punctuation · speaker diarization · emotion / event · LLM-ASR**. A Star really helps the project (and keeps you updated):

[**🌟 FunASR**](https://github.com/modelscope/FunASR)  ·  [**🌟 SenseVoice**](https://github.com/FunAudioLLM/SenseVoice)  ·  [**🌟 Fun-ASR**](https://github.com/FunAudioLLM/Fun-ASR)  ·  [**🌟 FunClip**](https://github.com/modelscope/FunClip)

</div>


# CAM++

**Speaker Verification & Diarization** — identify and distinguish speakers in audio.

CAM++ is a speaker embedding model used for speaker verification (is this the same person?) and speaker diarization (who spoke when?).

## Quick Start

```python
from funasr import AutoModel

# Speaker diarization as part of ASR pipeline
model = AutoModel(
    model="funasr/paraformer-zh",
    hub="hf",
    vad_model="funasr/fsmn-vad",
    punc_model="funasr/ct-punc",
    spk_model="funasr/campplus",
    device="cuda",
)
result = model.generate(input="meeting.wav")
# Each sentence has a speaker label
for sentence in result[0]["sentence_info"]:
    print(f"[Speaker {sentence['spk']}] {sentence['text']}")
```

## Features

- Speaker embedding extraction
- Speaker verification (same/different speaker)
- Speaker diarization (multi-speaker segmentation)
- Works with FunASR pipeline via `spk_model` parameter

## Model Details

| Property | Value |
|----------|-------|
| Architecture | CAM++ (Class-Aware Multi-scale) |
| Embedding Dim | 192 |
| Sample Rate | 16kHz |

## Links

- **GitHub**: [FunASR](https://github.com/modelscope/FunASR)
- **Docs**: [modelscope.github.io/FunASR](https://modelscope.github.io/FunASR/)

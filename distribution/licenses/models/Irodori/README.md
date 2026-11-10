---
license: mit
language:
- ja
pipeline_tag: text-to-speech
tags:
- speech
- voice
- tts
- meanflow
base_model:
- Aratako/Irodori-TTS-v4.1-Small
---

# Irodori-TTS-v4.1-Small-MF

[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/Aratako/Irodori-TTS)

**Irodori-TTS-v4.1-Small-MF** is a Japanese Text-to-Speech model distilled from [Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) using MeanFlow for generation with fewer sampling steps. It supports zero-shot voice cloning, text-based Voice Design, style-controlled voice cloning, long-reference conditioning, and emoji-based style control.

At four sampling steps, MeanFlow substantially improves Japanese reading accuracy and speaker similarity over the RF model at the same step count, bringing these metrics closer to the 40-step RF results.

## ✨ MeanFlow Distillation

The original Rectified Flow (RF) model generates audio by repeatedly predicting the instantaneous velocity along a path from noise to audio latents. MeanFlow learns the average velocity over a sampling interval, allowing larger updates with fewer model evaluations.

This model was distilled from a frozen v4.1-Small RF teacher. An interval-length embedding was added to the student DiT, and the entire student DiT was trained to predict the teacher's average velocity. The condition encoders and duration predictor were kept frozen.

Classifier-free guidance (CFG) is incorporated into the teacher targets during distillation. MeanFlow inference therefore uses one conditional DiT evaluation per sampling step, without separate CFG evaluations.

## 🚀 Usage

For inference code, installation instructions, and training scripts, please refer to the GitHub repository:

👉 **[GitHub: Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)**

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small-MF \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample_meanflow.wav
```

The default is four sampling steps. Use `--num-steps` to change the step count. The CLI and Gradio UIs detect MeanFlow checkpoints automatically. In Gradio, enter `Aratako/Irodori-TTS-v4.1-Small-MF` in the checkpoint field and leave **Num Steps** blank to use the default.

Text, captions, reference audio, and duration controls use the same interface as v4.1-Small. Inference-time CFG settings and Sway Sampling do not apply to MeanFlow. See the [MeanFlow guide](https://github.com/Aratako/Irodori-TTS/blob/main/docs/meanflow.md) for inference and distillation details.

## 📊 Benchmarks

The following evaluations compare MeanFlow at four steps with RF at four and 40 steps. All runs use FP32 inference and five sampling seeds, 0 through 4. Values with ± report the mean and population standard deviation across seeds; top-1 accuracy is reported as the five-seed mean.

### Japanese Reading

Reading was evaluated without reference audio or captions. The RF runs use linear Euler sampling and text CFG 3.0; MeanFlow uses guidance learned during distillation.

#### Joyo Kanji Yomi Benchmark: Parakeet Edition

This evaluation uses [JKYB-Parakeet](https://github.com/Parakeet-Inc/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition), rather than the original Joyo Kanji Yomi Benchmark used in the earlier model cards.

| Model | Steps | Accuracy ↑ | Relaxed accuracy ↑ | Target Kana-CER ↓ | Target Kana-CER@1 ↓ | Sentence Kana-CER ↓ | Text CER ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Irodori-TTS-v4.1-Small | RF 40 | **93.42 ± 0.04%** | **93.51 ± 0.04%** | **6.88 ± 0.08%** | **5.19 ± 0.05%** | **1.25 ± 0.01%** | **4.68 ± 0.06%** |
| Irodori-TTS-v4.1-Small | RF 4 | 89.65 ± 0.15% | 89.73 ± 0.14% | 11.10 ± 0.38% | 7.93 ± 0.09% | 2.65 ± 0.04% | 6.35 ± 0.13% |
| **Irodori-TTS-v4.1-Small-MF** | **MF 4** | 92.76 ± 0.09% | 92.84 ± 0.09% | 7.67 ± 0.13% | 5.68 ± 0.05% | 1.44 ± 0.01% | 4.96 ± 0.04% |

#### JSUT BASIC5000

| Model | Steps | Sentence Kana-CER ↓ | Standard CER ↓ |
| :--- | ---: | ---: | ---: |
| Irodori-TTS-v4.1-Small | RF 40 | **3.43 ± 0.01%** | **7.22 ± 0.12%** |
| Irodori-TTS-v4.1-Small | RF 4 | 6.50 ± 0.34% | 11.13 ± 0.63% |
| **Irodori-TTS-v4.1-Small-MF** | **MF 4** | 3.78 ± 0.03% | 7.74 ± 0.16% |

MeanFlow improves every reported reading metric over four-step RF. For example, JKYB-Parakeet target Kana-CER decreases from 11.10% to 7.67%, and JSUT sentence Kana-CER decreases from 6.50% to 3.78%. The 40-step RF model remains better on these reading metrics.

### Voice Cloning and Reference Length

Voice cloning was evaluated on JVS using all 100 speakers, five target texts, and five synthesis seeds. Independently encoded reference utterances were concatenated into one-clip, approximately 30-second, approximately 60-second, and 120-second conditions. CAM++ similarity was measured against a fixed centroid of ten held-out natural utterances per speaker. Top-1 accuracy measures whether the generated voice is matched to the correct speaker among the 100 JVS speakers.

The RF runs use text CFG 3.0 and speaker CFG 5.0. The RF 40-step baseline is the published **v4-Small** result; the RF four-step and MeanFlow results use **v4.1-Small** and **v4.1-Small-MF**, respectively.

#### JVS CAM++ Cosine Similarity ↑

| Reference | v4-Small, RF 40 | v4.1-Small, RF 4 | v4.1-Small-MF, MF 4 |
| :--- | ---: | ---: | ---: |
| One clip | **0.6610 ± 0.0013** | 0.3593 ± 0.0009 | 0.6604 ± 0.0013 |
| ~30 seconds | **0.7521 ± 0.0008** | 0.3835 ± 0.0043 | 0.7429 ± 0.0006 |
| ~60 seconds | **0.7646 ± 0.0003** | 0.3694 ± 0.0013 | 0.7532 ± 0.0011 |
| 120 seconds | **0.7753 ± 0.0009** | 0.3398 ± 0.0030 | 0.7647 ± 0.0012 |

#### JVS CAM++ Top-1 Accuracy ↑

| Reference | v4-Small, RF 40 | v4.1-Small, RF 4 | v4.1-Small-MF, MF 4 |
| :--- | ---: | ---: | ---: |
| One clip | 84.60% | 24.56% | **84.64%** |
| ~30 seconds | **98.56%** | 37.92% | 97.52% |
| ~60 seconds | **99.56%** | 35.04% | 98.36% |
| 120 seconds | **99.76%** | 30.28% | 98.96% |

Reducing RF sampling to four steps sharply reduces speaker similarity in this evaluation. MeanFlow retains much more of the 40-step baseline's cloning performance: with approximately 30 seconds of reference audio, CAM++ cosine similarity is 0.7429 versus 0.3835 for four-step RF and 0.7521 for 40-step RF.

## ⚠️ Limitations

  - **Japanese Only:** This model currently supports Japanese text input only.
  - **Few-step Quality:** Four-step MeanFlow improves over four-step RF in the reported benchmarks, but does not fully match 40-step RF reading accuracy or long-reference speaker similarity.
  - **Fixed Guidance:** CFG is learned during distillation and cannot be adjusted through inference-time CFG scales.
  - **Long-reference Composition:** Training and long-reference evaluation used multiple short utterances from the same speaker concatenated together. A single uninterrupted long recording is supported as input, but its effect has not been evaluated.
  - **Conditioning Conflicts:** When using both reference audio and a text caption, contradictory instructions may result in unstable audio quality, unnatural artifacts, or one condition overriding the other. For optimal results, use the caption to guide emotion, style, or environment while keeping the base voice characteristics aligned with the reference audio.
  - **Prompt Adherence:** Highly complex or contradictory captions may produce inconsistent results.
  - **Emoji Control:** The effect of emoji-based control may vary depending on context and is not always perfectly consistent.
  - **Kanji Reading:** Uncommon names, specialized terminology, and context-dependent readings may still be pronounced incorrectly.
  - **Evaluation Scope:** The reported benchmarks measure reading and speaker similarity, not Voice Design or emoji-control quality. Automatic benchmark scores do not fully represent human perception.

## 📜 License & Ethical Restrictions

### License

This model is released under **[MIT](https://choosealicense.com/licenses/mit/)**.

### Ethical Restrictions

In addition to the license terms, the following ethical restrictions apply:

1.  **No Impersonation:** Do not use this model to clone or impersonate the voice of any individual (e.g., voice actors, celebrities, public figures) without their explicit consent.
2.  **No Misinformation:** Do not use this model to generate deepfakes or synthetic speech intended to mislead others or spread misinformation.
3.  **Voice Generation Disclaimer:** When generating speech purely from text or captions without using reference audio, it is possible that the generated voice may coincidentally resemble that of a real person. This is strictly a probabilistic artifact within the latent space. The model was not trained with the intent of reproducing specific individuals.
4.  **Liability Disclaimer:** The developers assume no liability for any misuse of this model. Users are solely responsible for ensuring their use of the generated content complies with applicable laws and regulations in their jurisdiction.

## 🙏 Acknowledgments

This project builds upon the following works:

  - [dots.tts](https://github.com/studio-dots-ai/dots.tts) — MeanFlow implementation reference
  - [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — Architecture and training design reference
  - [DACVAE](https://github.com/facebookresearch/dacvae) — Audio VAE
  - [sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m) — Pretrained Japanese text and caption encoder
  - [SilentCipher](https://github.com/sony/silentcipher) — Audio watermarking integration

We would also like to extend our special thanks to **[Respair](https://huggingface.co/Respair)** for the inspiration behind the emoji annotation feature, and to [gabrielclark3330](https://huggingface.co/gabrielclark3330) and [kikouousya](https://huggingface.co/kikouousya) for supporting this project.

## 🖊️ Citation

If you use Irodori-TTS in your research or project, please cite it as follows:

```bibtex
@misc{irodori-tts-v4.1-small-mf,
  author = {Chihiro Arata},
  title = {Irodori-TTS: A Flow Matching-based Text-to-Speech Model with Emoji-driven Style Control},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face repository},
  howpublished = {\url{https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF}}
}
```

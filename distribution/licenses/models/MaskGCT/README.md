---
license: cc-by-nc-4.0
datasets:
- amphion/Emilia-Dataset
language:
- en
- zh
- ko
- ja
- fr
- de
base_model:
- amphion/MaskGCT
pipeline_tag: text-to-speech
---
## MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer

[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2409.00750) [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/amphion/maskgct) [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-demo-pink)](https://huggingface.co/spaces/amphion/maskgct) [![readme](https://img.shields.io/badge/README-Key%20Features-blue)](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)

## Quickstart

**Clone and install**

```bash
git clone https://github.com/open-mmlab/Amphion.git
# create env
bash ./models/tts/maskgct/env.sh
```

**Model download**

We provide the following pretrained checkpoints:


| Model Name          | Description   |    
|-------------------|-------------|
| [Semantic Codec](https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec)      | Converting speech to semantic tokens. |
| [Acoustic Codec](https://huggingface.co/amphion/MaskGCT/tree/main/acoustic_codec)      | Converting speech to acoustic tokens and reconstructing waveform from acoustic tokens. |
| [MaskGCT-T2S](https://huggingface.co/amphion/MaskGCT/tree/main/t2s_model)         | Predicting semantic tokens with text and prompt semantic tokens.             |
| [MaskGCT-S2A](https://huggingface.co/amphion/MaskGCT/tree/main/s2a_model)         | Predicts acoustic tokens conditioned on semantic tokens.              |

You can download all pretrained checkpoints from [HuggingFace](https://huggingface.co/amphion/MaskGCT/tree/main) or use huggingface api.

```python
from huggingface_hub import hf_hub_download

# download semantic codec ckpt
semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")

# download acoustic codec ckpt
codec_encoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model.safetensors")
codec_decoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors")

# download t2s model ckpt
t2s_model_ckpt = hf_hub_download("amphion/MaskGCT", filename="t2s_model/model.safetensors")

# download s2a model ckpt
s2a_1layer_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors")
s2a_full_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors")
```

**Basic Usage**

You can use the following code to generate speech from text and a prompt speech.
```python
from models.tts.maskgct.maskgct_utils import *
from huggingface_hub import hf_hub_download
import safetensors
import soundfile as sf

if __name__ == "__main__":

    # build model
    device = torch.device("cuda:0")
    cfg_path = "./models/tts/maskgct/config/maskgct.json"
    cfg = load_config(cfg_path)
    # 1. build semantic model (w2v-bert-2.0)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    # 2. build semantic codec
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    # 3. build acoustic codec
    codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
    # 4. build t2s model
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    # 5. build s2a model
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full =  build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    # download checkpoint
    ...

    # load semantic codec
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    # load acoustic codec
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
    # load t2s model
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
    # load s2a model
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

    # inference
    prompt_wav_path = "./models/tts/maskgct/wav/prompt.wav"
    save_path = "[YOUR SAVE PATH]"
    prompt_text = " We do not break. We never give in. We never back down."
    target_text = "In this paper, we introduce MaskGCT, a fully non-autoregressive TTS model that eliminates the need for explicit alignment information between text and speech supervision."
    # Specify the target duration (in seconds). If target_len = None, we use a simple rule to predict the target duration.
    target_len = 18

    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )

    recovered_audio = maskgct_inference_pipeline.maskgct_inference(
        prompt_wav_path, prompt_text, target_text, "en", "en", target_len=target_len
    )
    sf.write(save_path, recovered_audio, 24000)        
```

**Training Dataset**

We use the [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) dataset to train our models. Emilia is a multilingual and diverse in-the-wild speech dataset designed for large-scale speech generation. In this work, we use English and Chinese data from Emilia, each with 50K hours of speech (totaling 100K hours).

**Citation**

If you use MaskGCT in your research, please cite the following paper:
```bibtex
@article{wang2024maskgct,
  title={MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer},
  author={Wang, Yuancheng and Zhan, Haoyue and Liu, Liwei and Zeng, Ruihong and Guo, Haotian and Zheng, Jiachen and Zhang, Qiang and Zhang, Xueyao and Zhang, Shunsi and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2409.00750},
  year={2024}
}
@inproceedings{amphion,
    author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and Li, Jiaqi and He, Haorui and Wang, Chaoren and Song, Ting and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```
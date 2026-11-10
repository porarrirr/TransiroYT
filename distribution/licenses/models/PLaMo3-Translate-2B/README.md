---
license: "other"
license_name: "plamo-community-license"
license_link: "https://plamo.preferredai.jp/info/plamo-community-license-en"
base_model: pfnet/plamo-3-translate-2b
base_model_relation: quantized
model_name: "PLaMo 3 Translate 2B GGUF"
library_name: gguf
language:
- ar
- az
- be
- bg
- bn
- ca
- ceb
- de
- en
- es
- et
- eu
- fa
- fr
- gu
- he
- hi
- hr
- hu
- hy
- id
- it
- ja
- jv
- ka
- kk
- km
- ko
- lt
- lv
- ml
- mr
- ms
- my
- ne
- nl
- pa
- pl
- ru
- si
- sk
- sl
- sr
- sw
- ta
- te
- th
- tl
- ur
- uz
- vi
- zh
pipeline_tag: translation
tags:
- translation
- multilingual
- plamo
- gguf
- llama.cpp

---

# PLaMo 3 Translate 2B GGUF

[日本語版のREADME / Japanese README](README_ja.md)

PLaMo 3 Translate 2B GGUF is a version of [PLaMo 3 Translate 2B](https://huggingface.co/pfnet/plamo-3-translate-2b) converted to the GGUF format and quantized for [llama.cpp](https://github.com/ggml-org/llama.cpp). See [Choosing a Distribution Format](#choosing-a-distribution-format) to compare it with the other distribution formats.

PLaMo 3 Translate 2B is a translation-specialized LLM developed by Preferred Networks (PFN) that delivers world-class translation quality.
It is built on PLaMo 3 NICT, a pretrained model developed through joint research by PFN and the National Institute of Information and Communications Technology (NICT).
We further trained this base model on tens of billions of tokens of translation-focused data.
For development details, see the [tech blog](https://www.preferred.jp/en/blog/tech/plamo-3-translate-weight-release).

PLaMo 3 Translate 2B supports translation among 53 languages (see [Supported Languages](#supported-languages)).
It can also take surrounding context and a glossary into account, and lets you choose between plain and polite style for Japanese output.
The maximum length during training was 32,768 tokens for the input and output combined. The model can translate long passages in one go, but split anything longer so that the input and output together stay within this limit.

PLaMo 3 Translate 2B is available under the [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en). Personal use, such as learning or hobby projects, and other non-commercial use, including academic use, are free of charge with no prior registration. See [License](#license) for the terms of commercial use.

## Benchmarks

PLaMo 3 Translate delivers world-class quality in both Japanese–English and multilingual translation. Even the 2B model outperforms the translation-specialized LLMs we compared against on every metric in the figure below.

![Translation quality of PLaMo 3 Translate, TranslateGemma and HY-MT2](figures/benchmark_en.png)

The top row shows Japanese–English translation quality as the average of PFTB, SaaS and WMT24++ COMET-22. The middle row shows translation between Japanese and 14 other languages, and the bottom row between English and 14 other languages, each as the average of FLORES+ chrF++ in both directions. Higher is better for all scores.

For the FLORES+ multilingual evaluation, we selected the following 15 of the 53 supported languages. Translation was evaluated in both directions between Japanese or English and each of the remaining 14 languages.

- Japanese, English, Arabic, Indonesian, Italian, Dutch, Korean, Spanish, Thai, Chinese (Simplified and Traditional), German, French, Vietnamese, Russian

For details, see the [tech blog](https://www.preferred.jp/en/blog/tech/plamo-3-translate-weight-release).

## Choosing a Model

### Model List
PLaMo 3 Translate comes in four sizes: 0.8B, 2B, 8B and 31B.
For each size, quantized and converted variants are provided in separate repositories.

| Size | BF16 | FP8 | W4A16 | W4AFP8 | NVFP4 | GGUF | MLX |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8B | [BF16](https://huggingface.co/pfnet/plamo-3-translate-0.8b) | [FP8](https://huggingface.co/pfnet/plamo-3-translate-0.8b-FP8) | [W4A16](https://huggingface.co/pfnet/plamo-3-translate-0.8b-W4A16) | [W4AFP8](https://huggingface.co/pfnet/plamo-3-translate-0.8b-W4AFP8) | [NVFP4](https://huggingface.co/pfnet/plamo-3-translate-0.8b-NVFP4) | [GGUF](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF) | [MLX-4bit](https://huggingface.co/pfnet/plamo-3-translate-0.8b-MLX-4bit) |
| 2B | [BF16](https://huggingface.co/pfnet/plamo-3-translate-2b) | [FP8](https://huggingface.co/pfnet/plamo-3-translate-2b-FP8) | [W4A16](https://huggingface.co/pfnet/plamo-3-translate-2b-W4A16) | [W4AFP8](https://huggingface.co/pfnet/plamo-3-translate-2b-W4AFP8) | [NVFP4](https://huggingface.co/pfnet/plamo-3-translate-2b-NVFP4) | [GGUF](https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF) | [MLX-4bit](https://huggingface.co/pfnet/plamo-3-translate-2b-MLX-4bit) |
| 8B | [BF16](https://huggingface.co/pfnet/plamo-3-translate-8b) | [FP8](https://huggingface.co/pfnet/plamo-3-translate-8b-FP8) | [W4A16](https://huggingface.co/pfnet/plamo-3-translate-8b-W4A16) | [W4AFP8](https://huggingface.co/pfnet/plamo-3-translate-8b-W4AFP8) | [NVFP4](https://huggingface.co/pfnet/plamo-3-translate-8b-NVFP4) | [GGUF](https://huggingface.co/pfnet/plamo-3-translate-8b-GGUF) | [MLX-4bit](https://huggingface.co/pfnet/plamo-3-translate-8b-MLX-4bit) |
| 31B | [BF16](https://huggingface.co/pfnet/plamo-3-translate-31b) | [FP8](https://huggingface.co/pfnet/plamo-3-translate-31b-FP8) | [W4A16](https://huggingface.co/pfnet/plamo-3-translate-31b-W4A16) | [W4AFP8](https://huggingface.co/pfnet/plamo-3-translate-31b-W4AFP8) | [NVFP4](https://huggingface.co/pfnet/plamo-3-translate-31b-NVFP4) | [GGUF](https://huggingface.co/pfnet/plamo-3-translate-31b-GGUF) | [MLX-4bit](https://huggingface.co/pfnet/plamo-3-translate-31b-MLX-4bit) |

<details>
<summary>Accuracy changes from quantization</summary>

For each model size, the tables compare the translation accuracy of BF16 with that of each distribution format. The value in parentheses under each score is the difference from BF16 of the same size (a negative value means lower accuracy). Higher is better for all scores.

For details on the evaluation metrics, see the [tech blog](https://www.preferred.jp/en/blog/tech/plamo-3-translate-weight-release).

Overall, `GGUF Q2_K` shows lower translation accuracy on some metrics, while the other formats stay close to BF16.

Size is the total size of the weight files.

#### 31B

| Format | Size (GB) | PFTB (En↔Ja) | SaaS (En→Ja) | WMT24++ COMET-22 (En→Ja) | FLORES+ chrF++ (Ja↔Multilingual) | FLORES+ chrF++ (En↔Multilingual) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BF16** | 63.51 | 0.6244 | 0.5786 | 0.8799 | 37.81 | 56.61 |
| FP8 | 32.31 | 0.6240<br>(-0.0004) | 0.5803<br>(+0.0017) | 0.8799<br>(0.0000) | 37.79<br>(-0.02) | 56.60<br>(-0.01) |
| W4A16 | 17.19 | 0.6258<br>(+0.0014) | 0.5777<br>(-0.0009) | 0.8805<br>(+0.0006) | 37.74<br>(-0.07) | 56.49<br>(-0.12) |
| W4AFP8 | 17.19 | 0.6223<br>(-0.0021) | 0.5716<br>(-0.0070) | 0.8813<br>(+0.0014) | 37.71<br>(-0.10) | 56.53<br>(-0.08) |
| NVFP4 | 18.66 | 0.6258<br>(+0.0014) | 0.5748<br>(-0.0038) | 0.8809<br>(+0.0010) | 37.70<br>(-0.11) | 56.49<br>(-0.12) |
| GGUF BF16 | 63.52 | 0.6241<br>(-0.0003) | 0.5793<br>(+0.0007) | 0.8799<br>(0.0000) | 37.82<br>(+0.01) | 56.59<br>(-0.02) |
| GGUF Q4_K_M | 19.53 | 0.6237<br>(-0.0007) | 0.5770<br>(-0.0016) | 0.8810<br>(+0.0011) | 37.84<br>(+0.03) | 56.55<br>(-0.06) |
| GGUF Q3_K_M | 15.86 | 0.6223<br>(-0.0021) | 0.5780<br>(-0.0006) | 0.8813<br>(+0.0014) | 37.78<br>(-0.03) | 56.54<br>(-0.07) |
| GGUF Q2_K | 12.61 | 0.6126<br>(-0.0118) | 0.5745<br>(-0.0041) | 0.8801<br>(+0.0002) | 37.36<br>(-0.45) | 56.16<br>(-0.45) |
| MLX 4bit | 18.66 | 0.6222<br>(-0.0022) | 0.5779<br>(-0.0007) | 0.8800<br>(+0.0001) | 37.82<br>(+0.01) | 56.58<br>(-0.03) |

#### 8B

| Format | Size (GB) | PFTB (En↔Ja) | SaaS (En→Ja) | WMT24++ COMET-22 (En→Ja) | FLORES+ chrF++ (Ja↔Multilingual) | FLORES+ chrF++ (En↔Multilingual) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BF16** | 17.06 | 0.6119 | 0.5750 | 0.8805 | 37.45 | 56.22 |
| FP8 | 9.41 | 0.6123<br>(+0.0004) | 0.5764<br>(+0.0014) | 0.8805<br>(0.0000) | 37.40<br>(-0.05) | 56.17<br>(-0.05) |
| W4A16 | 5.71 | 0.6114<br>(-0.0005) | 0.5749<br>(-0.0001) | 0.8789<br>(-0.0016) | 37.36<br>(-0.09) | 56.19<br>(-0.03) |
| W4AFP8 | 5.71 | 0.6174<br>(+0.0055) | 0.5759<br>(+0.0009) | 0.8794<br>(-0.0011) | 37.34<br>(-0.11) | 56.12<br>(-0.10) |
| NVFP4 | 6.07 | 0.6040<br>(-0.0079) | 0.5783<br>(+0.0033) | 0.8792<br>(-0.0013) | 37.26<br>(-0.19) | 56.10<br>(-0.12) |
| GGUF BF16 | 17.07 | 0.6110<br>(-0.0009) | 0.5755<br>(+0.0005) | 0.8802<br>(-0.0003) | 37.46<br>(+0.01) | 56.22<br>(0.00) |
| GGUF Q4_K_M | 5.28 | 0.6100<br>(-0.0019) | 0.5765<br>(+0.0015) | 0.8802<br>(-0.0003) | 37.39<br>(-0.06) | 56.18<br>(-0.04) |
| GGUF Q3_K_M | 4.31 | 0.6019<br>(-0.0100) | 0.5707<br>(-0.0043) | 0.8826<br>(+0.0021) | 37.20<br>(-0.25) | 55.93<br>(-0.29) |
| GGUF Q2_K | 3.45 | 0.5989<br>(-0.0130) | 0.5676<br>(-0.0074) | 0.8791<br>(-0.0014) | 36.54<br>(-0.91) | 55.31<br>(-0.91) |
| MLX 4bit | 6.07 | 0.6102<br>(-0.0017) | 0.5751<br>(+0.0001) | 0.8796<br>(-0.0009) | 37.41<br>(-0.04) | 56.19<br>(-0.03) |

#### 2B

| Format | Size (GB) | PFTB (En↔Ja) | SaaS (En→Ja) | WMT24++ COMET-22 (En→Ja) | FLORES+ chrF++ (Ja↔Multilingual) | FLORES+ chrF++ (En↔Multilingual) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BF16** | 5.76 | 0.6061 | 0.5612 | 0.8780 | 37.02 | 55.74 |
| FP8 | 3.43 | 0.6067<br>(+0.0006) | 0.5613<br>(+0.0001) | 0.8776<br>(-0.0004) | 36.98<br>(-0.04) | 55.76<br>(+0.02) |
| W4A16 | 2.30 | 0.6032<br>(-0.0029) | 0.5594<br>(-0.0018) | 0.8783<br>(+0.0003) | 36.96<br>(-0.06) | 55.69<br>(-0.05) |
| W4AFP8 | 2.30 | 0.6022<br>(-0.0039) | 0.5612<br>(0.0000) | 0.8772<br>(-0.0008) | 36.95<br>(-0.07) | 55.71<br>(-0.03) |
| NVFP4 | 2.41 | 0.6032<br>(-0.0029) | 0.5626<br>(+0.0014) | 0.8783<br>(+0.0003) | 36.82<br>(-0.20) | 55.64<br>(-0.10) |
| GGUF BF16 | 5.76 | 0.6062<br>(+0.0001) | 0.5619<br>(+0.0007) | 0.8782<br>(+0.0002) | 37.02<br>(0.00) | 55.75<br>(+0.01) |
| GGUF Q4_K_M | 1.81 | 0.6062<br>(+0.0001) | 0.5602<br>(-0.0010) | 0.8774<br>(-0.0006) | 37.00<br>(-0.02) | 55.76<br>(+0.02) |
| GGUF Q4_0 | 1.70 | 0.6069<br>(+0.0008) | 0.5604<br>(-0.0008) | 0.8791<br>(+0.0011) | 36.12<br>(-0.90) | 55.62<br>(-0.12) |
| GGUF Q3_K_M | 1.50 | 0.6053<br>(-0.0008) | 0.5644<br>(+0.0032) | 0.8789<br>(+0.0009) | 36.81<br>(-0.21) | 55.59<br>(-0.15) |
| GGUF Q2_K | 1.23 | 0.5977<br>(-0.0084) | 0.5610<br>(-0.0002) | 0.8777<br>(-0.0003) | 36.45<br>(-0.57) | 55.02<br>(-0.72) |
| MLX 4bit | 2.41 | 0.6040<br>(-0.0021) | 0.5624<br>(+0.0012) | 0.8777<br>(-0.0003) | 36.99<br>(-0.03) | 55.64<br>(-0.10) |

#### 0.8B

| Format | Size (GB) | PFTB (En↔Ja) | SaaS (En→Ja) | WMT24++ COMET-22 (En→Ja) | FLORES+ chrF++ (Ja↔Multilingual) | FLORES+ chrF++ (En↔Multilingual) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BF16** | 1.78 | 0.5862 | 0.5530 | 0.8724 | 36.08 | 54.44 |
| FP8 | 1.22 | 0.5846<br>(-0.0016) | 0.5491<br>(-0.0039) | 0.8726<br>(+0.0002) | 36.00<br>(-0.08) | 54.60<br>(+0.16) |
| W4A16 | 0.95 | 0.5847<br>(-0.0015) | 0.5439<br>(-0.0091) | 0.8711<br>(-0.0013) | 35.78<br>(-0.30) | 53.95<br>(-0.49) |
| W4AFP8 | 0.95 | 0.5783<br>(-0.0079) | 0.5456<br>(-0.0074) | 0.8718<br>(-0.0006) | 35.75<br>(-0.33) | 54.38<br>(-0.06) |
| NVFP4 | 0.98 | 0.5787<br>(-0.0075) | 0.5436<br>(-0.0094) | 0.8702<br>(-0.0022) | 35.54<br>(-0.54) | 54.20<br>(-0.24) |
| GGUF BF16 | 1.78 | 0.5873<br>(+0.0011) | 0.5533<br>(+0.0003) | 0.8727<br>(+0.0003) | 36.10<br>(+0.02) | 54.45<br>(+0.01) |
| GGUF Q4_K_M | 0.57 | 0.5832<br>(-0.0030) | 0.5451<br>(-0.0079) | 0.8716<br>(-0.0008) | 35.90<br>(-0.18) | 54.47<br>(+0.03) |
| GGUF Q4_0 | 0.55 | 0.5853<br>(-0.0009) | 0.5500<br>(-0.0030) | 0.8723<br>(-0.0001) | 34.97<br>(-1.11) | 54.11<br>(-0.33) |
| GGUF Q3_K_M | 0.49 | 0.5798<br>(-0.0064) | 0.5520<br>(-0.0010) | 0.8710<br>(-0.0014) | 35.32<br>(-0.76) | 54.15<br>(-0.29) |
| GGUF Q2_K | 0.41 | 0.5620<br>(-0.0242) | 0.5397<br>(-0.0133) | 0.8659<br>(-0.0065) | 33.56<br>(-2.52) | 51.15<br>(-3.29) |
| MLX 4bit | 0.98 | 0.5773<br>(-0.0089) | 0.5476<br>(-0.0054) | 0.8724<br>(0.0000) | 35.82<br>(-0.26) | 54.41<br>(-0.03) |

</details>

### Choosing a Distribution Format

Each size of PLaMo 3 Translate is available in the following distribution formats, depending on your use case and hardware.

| Format | Target hardware | Inference engine | Typical use |
| --- | --- | --- | --- |
| BF16 | GPUs with BF16 support | Transformers / vLLM | Prioritizes accuracy with no quantization, when memory allows |
| FP8 | NVIDIA Ada Lovelace / Hopper / Blackwell | vLLM | Balances accuracy with memory savings and speed |
| W4A16 | NVIDIA Ampere or later | vLLM | 4-bit weights to run with less GPU memory |
| W4AFP8 | NVIDIA Hopper | vLLM | 4-bit weights with FP8 compute, aiming at both low memory and high throughput |
| NVFP4 | NVIDIA Blackwell | vLLM | FP4 compute, aiming at low memory and high speed |
| **GGUF** | CPU / various GPUs | llama.cpp | Runs on a local PC, optionally combining CPU and GPU |
| MLX-4bit | Apple silicon | mlx-lm | Memory-efficient inference on a Mac |

## Usage

### llama.cpp

Install [llama.cpp](https://github.com/ggml-org/llama.cpp).

| File | Size | Typical use |
| --- | ---: | --- |
| `plamo-3-translate-2b-Q4_K_M.gguf` | 1.81 GB | Default. A good balance of accuracy and size; start here |
| `plamo-3-translate-2b-Q4_0.gguf` | 1.70 GB | For smartphones. llama.cpp's mobile backends (Arm CPU, Qualcomm Adreno GPU, Hexagon NPU) have optimized kernels for this format, so it tends to run fast on Snapdragon-based devices and similar hardware. Accuracy is about the same as Q4_K_M or slightly lower |
| `plamo-3-translate-2b-Q3_K_M.gguf` | 1.50 GB | When you need a smaller file. Slightly lower accuracy |
| `plamo-3-translate-2b-Q2_K.gguf` | 1.23 GB | Smallest file. Accuracy is somewhat lower than the other formats |

If you need a smaller file, Q3_K_M and Q2_K trade some accuracy for size. See "Accuracy changes from quantization" above for the accuracy of each format (GGUF BF16 in that table is a reference value for comparison and is not distributed). The examples below use Q4_K_M; substitute the file name to use another format.

Download the GGUF file you want to use and the bundled `chat_template.jinja`, then move into that directory. Run the following examples in this directory.

```bash
pip install -U huggingface_hub
hf download pfnet/plamo-3-translate-2b-GGUF plamo-3-translate-2b-Q4_K_M.gguf chat_template.jinja --local-dir plamo-3-translate-2b-GGUF
cd plamo-3-translate-2b-GGUF
```

Start the server with the saved GGUF file and the template. The legacy template embedded in the GGUF does not handle history, so specify `--chat-template-file`.

```bash
llama-server -m plamo-3-translate-2b-Q4_K_M.gguf -c 16384 --port 8000 --alias plamo-3-translate --chat-template-file chat_template.jinja --no-prefill-assistant
```

This example sets the maximum combined length of the input and output to 16,384 tokens with `-c 16384`. Adjust this limit to fit the available memory.

To use a GPU, install a llama.cpp build with GPU support. Recent llama.cpp places the model on the GPU automatically by default; use `-ngl` to set the number of layers explicitly. `--alias` sets the model name used in API requests.

Next, request translations by following [Translating with the API](#translating-with-the-api) below.

### Translating with the API

This section explains how to send source text to the Chat Completions API. To send PFML3 prompts directly, see the [Completions API example](#translating-with-the-completions-api).

The Python examples use the OpenAI client.

```bash
pip install -U openai
```

#### Translating with the Chat Completions API

A chat template is provided so you can try translation easily. Given the source text and the target language, it builds a prompt in the PFML3 format.

For production use, such as integrating the model into an application, we recommend building the prompt directly in the [PFML3 format](#pfml3-prompt-format) and passing it to the [Completions API](#translating-with-the-completions-api) of the running server.

The chat template of the GGUF version ignores `system` messages. Pass reference context through `context` in [Translation options](#translation-options) below.

With the Chat Completions API, put the current source text in the last `user` message and pass the translation options in `chat_template_kwargs`.

When `source_lang` is omitted, the model detects the language from the source text. To specify it explicitly, add `"source_lang": "ja"`, for example, to `chat_template_kwargs`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "plamo-3-translate",
    "messages": [{"role": "user", "content": "会議は来週の月曜日に変更されました。"}],
    "chat_template_kwargs": {"target_lang": "en"},
    "temperature": 0,
    "max_tokens": 512,
    "stop": ["<|plamo:tag|>", "<|plamo:bos|>"]
  }'
```

`max_tokens` is the generation limit for the translation. This example sets it to 512 tokens. When translating long text, adjust the limit so that the whole input (source text, history, context, glossary and so on) plus the output fits within the limit of your runtime.

Example output:

> The meeting has been rescheduled for next Monday.

The translation is returned in `choices[0].message.content`. If `target_lang` is omitted, the text is translated into Japanese.

**Translating multiple texts at once**

To translate multiple texts, send one request per text and run the requests in parallel. The following example sends up to four requests in parallel. Translation options such as `target_lang` specified inside `translate()` apply to every text.

```python
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
texts = [
    "会議は来週の月曜日に変更されました。",
    "プロジェクトの管理画面を開いてください。",
]

def translate(text):
    response = client.chat.completions.create(
        model="plamo-3-translate",
        messages=[{"role": "user", "content": text}],
        temperature=0,
        max_tokens=512,
        stop=["<|plamo:tag|>", "<|plamo:bos|>"],
        extra_body={"chat_template_kwargs": {"target_lang": "en"}},
    )
    return response.choices[0].message.content

with ThreadPoolExecutor(max_workers=4) as executor:
    translations = list(executor.map(translate, texts))

for source, translation in zip(texts, translations):
    print(f"{source} → {translation}")
```

**Translating with history via the Chat Completions API**

Pass the already translated source–translation pairs as `user`/`assistant` pairs. Only the last `user` message is translated.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
messages = [
    {"role": "user", "content": "プロジェクトの管理画面を開きます。"},
    {"role": "assistant", "content": "Open the project dashboard."},
    {"role": "user", "content": "進行中のタスクがすべて表示されます。"},
    {"role": "assistant", "content": "It lists all active tasks."},
    {"role": "user", "content": "オーナーでフィルタリングすることもできます。"},
]

response = client.chat.completions.create(
    model="plamo-3-translate",
    messages=messages,
    temperature=0,
    max_tokens=512,
    stop=["<|plamo:tag|>", "<|plamo:bos|>"],
    extra_body={"chat_template_kwargs": {"source_lang": "ja", "target_lang": "en"}},
)
print(response.choices[0].message.content)
```

### Continuing a translation with previous translations

When translating a document paragraph by paragraph, you can pass the previous source–translation pairs as history. This lets the model follow the earlier word choices and style when translating the next passage.

With the chat template, alternate the source text as `user` and its translation as `assistant`, and put the current source text in the final `user` message. History is not stored automatically, so include the pairs you need in each request.

The translation direction is shared by the whole request: `source_lang` applies to every source text and `target_lang` to every translation.

History also counts toward the input tokens. Keep the whole input plus the generated translation within 32,768 tokens and within the limit of your runtime. As the history grows, we recommend dropping the oldest source–translation pairs first and keeping the most recent ones.

### PFML3 Prompt Format

PFML3 is PLaMo's prompt format, in which the text to translate, the target language and other elements are delimited by special tokens. In the Chat Completions API example above, the chat template converts the source text and target language into PFML3. This section explains how to read the converted prompt and how to write prompts by hand.

For production use, we recommend building the prompt directly in PFML3 and passing it to the [Completions API](#translating-with-the-completions-api) of the running server.

Converting a request that translates the Japanese meeting notice above into English gives the following PFML3 prompt. The beginning-of-sequence token is omitted here, and the Python string is split across lines for readability. The actual prompt contains no line breaks.

```python
prompt = (
    "<|plamo:tag|>dataset<|plamo:msg|>translation"
    "<|plamo:tag|>input"
    "<|plamo:msg|>会議は来週の月曜日に変更されました。"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
)
```

- `translation` in `dataset` denotes the translation task. Because this model is dedicated to translation, always set `dataset` to `translation`.
- The part following `<|plamo:msg|>` in `input` is the source text to translate.
- `<|plamo:key|>lang<|plamo:val|>en` in `output` sets the target language to English. The translation is generated right after the final `<|plamo:msg|>`.

`<|plamo:tag|>` marks the start of each element, and `<|plamo:msg|>` marks the start of its body. To specify attributes such as the language or style, write the attribute name after `<|plamo:key|>` and its value after `<|plamo:val|>`, following the element name. For example, adding `<|plamo:key|>form<|plamo:val|>plain` to `output` requests plain style.

Context and a glossary are optional. To use them, add `context` and then `glossary`, in that order, between `dataset` and `input`. When translation history is also used, put `context` right after `dataset`, list the previous `input`/`output` pairs, and put `glossary` immediately before the current `input`. In the glossary, write a source term after `<|plamo:index|>` and its translation after `<|plamo:item|>`, repeating this pair for each term. See [Translation Options](#translation-options) for concrete examples.

#### Translating with the Completions API

When writing PFML3 by hand, start by setting `dataset` to `translation` and end the `output` element with `<|plamo:msg|>`. After generating the translation, the model emits `<|plamo:tag|>` or `<|plamo:bos|>` and stops.

The handwritten examples above omit `<|plamo:bos|>`, which marks the beginning of the sequence. llama.cpp adds it automatically, so do not add it to the beginning of a handwritten prompt.

A handwritten prompt is passed to the Completions API of the running server as `prompt`.

Pass one of the PFML3 prompts shown in the translation examples as `prompt`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
prompt = (
    "<|plamo:tag|>dataset<|plamo:msg|>translation"
    "<|plamo:tag|>input"
    "<|plamo:msg|>会議は来週の月曜日に変更されました。"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
)

response = client.completions.create(
    model="plamo-3-translate",
    prompt=prompt,
    temperature=0,
    max_tokens=512,
    stop=["<|plamo:tag|>", "<|plamo:bos|>"],
)
print(response.choices[0].text)
```

The translation is returned in `choices[0].text`.

**Translating multiple texts at once**

When using PFML3 prompts directly, you can likewise send one request per text in parallel.

```python
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
texts = [
    "会議は来週の月曜日に変更されました。",
    "プロジェクトの管理画面を開いてください。",
]

def translate(text):
    prompt = (
        "<|plamo:tag|>dataset<|plamo:msg|>translation"
        "<|plamo:tag|>input<|plamo:msg|>"
        f"{text}"
        "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
    )
    response = client.completions.create(
        model="plamo-3-translate",
        prompt=prompt,
        temperature=0,
        max_tokens=512,
        stop=["<|plamo:tag|>", "<|plamo:bos|>"],
    )
    return response.choices[0].text

with ThreadPoolExecutor(max_workers=4) as executor:
    translations = list(executor.map(translate, texts))

for source, translation in zip(texts, translations):
    print(f"{source} → {translation}")
```

#### Specifying translation history in PFML3

Translation history can also be written in PFML3 by hand. List the already translated `input` and `output` elements in document order, then add the current source text followed by an `output` element with an empty body. In the following example, only the last sentence, "オーナーでフィルタリングすることもできます。", is translated, with reference to the two previous pairs.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
prompt = (
    "<|plamo:tag|>dataset<|plamo:msg|>translation"
    "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>プロジェクトの管理画面を開きます。"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Open the project dashboard."
    "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>進行中のタスクがすべて表示されます。"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>It lists all active tasks."
    "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>オーナーでフィルタリングすることもできます。"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
)

response = client.completions.create(
    model="plamo-3-translate",
    prompt=prompt,
    temperature=0,
    max_tokens=512,
    stop=["<|plamo:tag|>", "<|plamo:bos|>"],
)
print(response.choices[0].text)
```

To continue translating, append the current source text and the generated translation to the history as a pair. Then add an `input` element containing the next source text and an `output` element with an empty body.

Keep only the body of each translation in the history. The translation returned by the example above can be used as is.

If `<|plamo:tag|>` or `<|plamo:bos|>` remains at the end of a translation, remove it so that exactly one `<|plamo:tag|>` separates it from the next `input`.

### Translation Options

Specify the target language, style, context, glossary and other options in `chat_template_kwargs` of the Chat Completions API, not in a `system` message or in the source text.

Run the following examples while the server is running.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
```

| Argument | Description |
| --- | --- |
| `target_lang` | Language code of the target language. Defaults to Japanese (`ja`) when omitted. |
| `source_lang` | Language code of the source text. Optional. |
| `context` | Reference context for interpreting the source text, such as the surrounding text or the situation. |
| `glossary` | A list of `{"source": "source term", "target": "translation"}` dictionaries, one per term. |
| `form` | `plain` for plain style or `polite` for polite style (only with `target_lang="ja"`). |

When combined with translation history, `context` is passed as reference context for the whole request, while `glossary` and `form` apply to the current translation. Previous translations themselves are not changed.

#### Specifying terms with a glossary

Pass pairs of source terms and their translations in `glossary` to control how the terms are translated.

```python
response = client.chat.completions.create(
    model="plamo-3-translate",
    messages=[{"role": "user", "content": "プロジェクトの管理画面を開いてください。"}],
    temperature=0, max_tokens=512,
    stop=["<|plamo:tag|>", "<|plamo:bos|>"],
    extra_body={
        "chat_template_kwargs": {
            "target_lang": "en",
            "glossary": [
                {"source": "管理画面", "target": "dashboard"},
                {"source": "プロジェクト", "target": "project"}
            ]
        }
    },
)
print(response.choices[0].message.content)
```

Example output:

> Please open the project dashboard.

<details>
<summary>PFML3 prompt for this example</summary>

```python
prompt = (
    "<|plamo:tag|>dataset<|plamo:msg|>translation"
    "<|plamo:tag|>glossary"
    "<|plamo:index|>管理画面<|plamo:item|>dashboard"
    "<|plamo:index|>プロジェクト<|plamo:item|>project"
    "<|plamo:tag|>input<|plamo:msg|>プロジェクトの管理画面を開いてください。"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
)
```

</details>

#### Providing context

Use `context` to pass the surrounding text or to describe the situation. In this example, the source text かみを切りました。 is ambiguous because かみ can mean either hair or paper. The context says that the speaker is making origami, so it is translated as paper.

```python
response = client.chat.completions.create(
    model="plamo-3-translate",
    messages=[{"role": "user", "content": "かみを切りました。"}],
    temperature=0, max_tokens=512,
    stop=["<|plamo:tag|>", "<|plamo:bos|>"],
    extra_body={
        "chat_template_kwargs": {
            "target_lang": "en",
            "context": "工作で折り紙を作っています。"
        }
    },
)
print(response.choices[0].message.content)
```

Example output:

> I cut the paper.

<details>
<summary>PFML3 prompt for this example</summary>

```python
prompt = (
    "<|plamo:tag|>dataset<|plamo:msg|>translation"
    "<|plamo:tag|>context"
    "<|plamo:msg|>工作で折り紙を作っています。"
    "<|plamo:tag|>input<|plamo:msg|>かみを切りました。"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
)
```

</details>

#### Specifying plain or polite style

When translating into Japanese, `form="plain"` requests the plain style (da/dearu) and `form="polite"` requests the polite style (desu/masu). Because `form` applies only to Japanese translations, this example translates English into Japanese.

```python
response = client.chat.completions.create(
    model="plamo-3-translate",
    messages=[{"role": "user", "content": "This system is convenient."}],
    temperature=0, max_tokens=512,
    stop=["<|plamo:tag|>", "<|plamo:bos|>"],
    extra_body={
        "chat_template_kwargs": {
            "target_lang": "ja",
            "form": "plain"
        }
    },
)
print(response.choices[0].message.content)
```

Example outputs for the same source text:

| Style | Option | Translation |
| --- | --- | --- |
| Plain | `form="plain"` | このシステムは便利だ。 |
| Polite | `form="polite"` | このシステムは便利です。 |

<details>
<summary>PFML3 prompt for this example</summary>

Plain:

```python
prompt = (
    "<|plamo:tag|>dataset<|plamo:msg|>translation"
    "<|plamo:tag|>input<|plamo:msg|>This system is convenient."
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja"
    "<|plamo:key|>form<|plamo:val|>plain<|plamo:msg|>"
)
```

Polite:

```python
prompt = (
    "<|plamo:tag|>dataset<|plamo:msg|>translation"
    "<|plamo:tag|>input<|plamo:msg|>This system is convenient."
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja"
    "<|plamo:key|>form<|plamo:val|>polite<|plamo:msg|>"
)
```
</details>

#### Specifying all options at once

Translation history, context, a glossary and the style can be combined in one request. The following example translates the third sentence after two previous pairs, passing the page type as context and fixing the translation of オーナー with the glossary. `context` is passed as reference context for the whole request, and `glossary` applies to the current translation. When translating into Japanese, `form` can be added in the same way.

```python
response = client.chat.completions.create(
    model="plamo-3-translate",
    messages=[
        {"role": "user", "content": "プロジェクトの管理画面を開きます。"},
        {"role": "assistant", "content": "Open the project dashboard."},
        {"role": "user", "content": "進行中のタスクがすべて表示されます。"},
        {"role": "assistant", "content": "It lists all active tasks."},
        {"role": "user", "content": "オーナーでフィルタリングすることもできます。"},
    ],
    temperature=0, max_tokens=512,
    stop=["<|plamo:tag|>", "<|plamo:bos|>"],
    extra_body={
        "chat_template_kwargs": {
            "source_lang": "ja",
            "target_lang": "en",
            "context": "プロジェクト管理ツールのヘルプページです。",
            "glossary": [
                {"source": "オーナー", "target": "assignee"}
            ]
        }
    },
)
print(response.choices[0].message.content)
```

Example output:

> You can also filter by assignee.

<details>
<summary>PFML3 prompt for this example</summary>

```python
prompt = (
    "<|plamo:tag|>dataset<|plamo:msg|>translation"
    "<|plamo:tag|>context<|plamo:msg|>プロジェクト管理ツールのヘルプページです。"
    "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>プロジェクトの管理画面を開きます。"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Open the project dashboard."
    "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>進行中のタスクがすべて表示されます。"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>It lists all active tasks."
    "<|plamo:tag|>glossary<|plamo:index|>オーナー<|plamo:item|>assignee"
    "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>オーナーでフィルタリングすることもできます。"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
)
```

</details>

## Supported Languages

53 languages are supported, including Japanese and English. Simplified and Traditional Chinese are counted as separate languages. Because Hugging Face groups Chinese as a single language in its language display, the model page shows 52 languages.

<details>
<summary>Supported languages and language codes</summary>

| Language | Code | Language | Code |
|---|---|---|---|
| Japanese | `ja` | English | `en` |
| Chinese (Simplified) | `zh-hans` | Chinese (Traditional) | `zh-hant` |
| Korean | `ko` | Arabic | `ar` |
| Azerbaijani | `az` | Belarusian | `be` |
| Bulgarian | `bg` | Bengali | `bn` |
| Catalan | `ca` | Cebuano | `ceb` |
| German | `de` | Spanish | `es` |
| Estonian | `et` | Basque | `eu` |
| Persian | `fa` | French | `fr` |
| Gujarati | `gu` | Hebrew | `he` |
| Hindi | `hi` | Croatian | `hr` |
| Hungarian | `hu` | Armenian | `hy` |
| Indonesian | `id` | Italian | `it` |
| Javanese | `jv` | Georgian | `ka` |
| Kazakh | `kk` | Khmer | `km` |
| Lithuanian | `lt` | Latvian | `lv` |
| Malayalam | `ml` | Marathi | `mr` |
| Malay | `ms` | Burmese | `my` |
| Nepali | `ne` | Dutch | `nl` |
| Punjabi | `pa` | Polish | `pl` |
| Russian | `ru` | Sinhala | `si` |
| Slovak | `sk` | Slovenian | `sl` |
| Serbian | `sr` | Swahili | `sw` |
| Tamil | `ta` | Telugu | `te` |
| Thai | `th` | Tagalog | `tl` |
| Urdu | `ur` | Uzbek | `uz` |
| Vietnamese | `vi` |  |  |

</details>

## License

PLaMo 3 Translate 2B GGUF is released under the [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en).

- [English](https://plamo.preferredai.jp/info/plamo-community-license-en)
- [Japanese](https://plamo.preferredai.jp/info/plamo-community-license-ja)

**Personal and non-commercial use**

For personal use, such as learning or hobby projects, you can download and use the model free of charge. Non-commercial use, including academic use, is also free, with no restriction based on revenue or sales and no prior registration with PFN.

**Commercial use when revenue or sales is 1 billion yen or less**

If the revenue or sales of the user or its affiliates in the most recent fiscal year is 1 billion yen or less, register your intended use through the [PLaMo Commercial Use Registration Form](https://forms.gle/mTL8tBLrMYXKNZD56). Commercial use is free of charge from the moment you submit the form.

**Commercial use when revenue or sales exceeds 1 billion yen**

A separate license agreement is required. Instead of the registration form above, please contact PFN through the [PLaMo Translation Model inquiry page](https://translate.preferredai.jp/llm/contact/).

See the full text of the PLaMo Community License for details.

## Preferred Networks Group AI Policy

- [日本語](https://www.preferred.jp/ja/company/aipolicy/)
- [English](https://www.preferred.jp/en/company/aipolicy/)

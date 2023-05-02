[![Tests](https://github.com/ulysses-camara/ulysses-segmenter/actions/workflows/tests.yml/badge.svg)](https://github.com/ulysses-camara/ulysses-segmenter/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/ulysses-segmenter/badge/?version=latest)](https://ulysses-segmenter.readthedocs.io/en/latest/?badge=latest)

## Brazilian Legislative Text Segmenter
Brazilian legislative bill segmenter models.

---

### Table of Contents
1. [Installation](#installation)
2. [Inference details](#inference-details)
3. [Available models](#available-models)
4. [Usage examples](#usage-examples)
    1. [Standard models (Torch format, Huggingface Transformers compatible)](#standard-models)
    2. [Noise subsegment removal](#noise-subsegment-removal)
    3. [Quantization in ONNX format](#quantization-in-onnx-format)
5. [Train and evaluation data](#train-and-evaluation-data)
6. [Package tests](#package-tests)
7. [License](#license)
8. [Citation](#citation)

---

### Installation
To install this package:
```bash
python -m pip install "git+https://github.com/ulysses-camara/ulysses-segmenter"
```

If you plan to use optimized models in ONNX format, you need to install some optional dependencies:
```bash
python -m pip install "segmentador[optimize] @ git+https://github.com/ulysses-camara/ulysses-segmenter"
```

---

### Inference details
The models used in this framework are Transformer Encoders (BERT) and Bidirectional LSTM (Bi-LSTM), with varying numbers of hidden layers (transformer blocks) and support for up to 1024 subword tokens in BERT models. However, since legal texts may exceed this limit, the framework automatically pre-segments the text into possibly overlapping 1024 subword windows in a moving window fashion, and feeds them to the segmenter model independently. The output from the model is then combined ("pooled"), and the final prediction for each token is derived.

<p align="center">
  <img src="./diagrams/segmenter_inference_pipeline.drawio.png" alt="Full segmenter inference pipeline."></img>
</p>

The *pooling* operations can be one of the following:

|Pooling         | Description                                                                                          |
| :---           | :---                                                                                                 |
| Sum (default)  | Sum overlapping logits.                                                                              |
| Max            | Keep maximal overlapping logits.                                                                     |
| Gaussian       | Weight logits by a Gaussian distribution centered in the middle of each moving window.   |
| Asymmetric-Max | Maximal logits for all classes except "No-op", which gets the minimal overlapping logit. |

---

### Available models
Pretrained Ulysses segmenter models are downloaded with [Ulysses Fetcher](https://github.com/ulysses-camara/ulysses-fetcher) API.

We use a two-stage training procedure: (1) weak supervision (data labeling with regular expressions), and (2) active learning.

In a curated dataset, comprised of 1447 ground-truth fragments of federal bills, Ulysses Segmenter achieves higher precision and recall for class 1 ("Start of sentence") when compared to other available popular segmentation tools: [NLTK](https://github.com/nltk/nltk), [SpaCy](https://github.com/explosion/spaCy), and [LexNLP](https://github.com/LexPredict/lexpredict-lexnlp), with the latter being suitable for segmenting legal texts. In the table below we compare these algorithms against Ulysses Segmenter, showing results for both estimated precision and recall.

- *v1* models were trained using weakly supervised data;
- *v2* models were trained using active learning from the corresponding *v1* model as base model;
- *v3* models were built on top of *v2* models to support state bills and additional legislative documents.

| Segmentation Method             | Est. Precision | Est. Recall | Size (MiB) |
|:---                             |:---            |:---         | :--------- |
| NLTK (v3.7)                     | 13.278%        | 19.738%     | --         |
| SpaCy (v3.5.0)                  | 13.422%        | 25.300%     | --         |
| LexNLP (v2.2.1.0)               | 13.462%        | 19.806%     | --         |
| Ulysses Segmenter v1 (LSTM-512) | 96.345%        | 93.004%     | 37         |
| Ulysses Segmenter v1 (BERT-2)   | 97.440%        | 93.530%     | 74         |
| Ulysses Segmenter v2 (LSTM-256) | 97.014%        | 94.909%     | **25**     |
| Ulysses Segmenter v2 (BERT-2)   | 97.981%        | 96.403%     | 74         |
| Ulysses Segmenter v2 (BERT-4)   | **98.555%**    | 96.854%     | 128        |
| Ulysses Segmenter v3 (LSTM-256) | 96.539%        | 95.953%     | **25**     |
| Ulysses Segmenter v3 (BERT-2)   | 97.850%        | 96.601%     | 74         |
| Ulysses Segmenter v3 (BERT-4)   | 98.334%        | **97.099%** | 128        |

The default models loaded for each algorithm are:

- *BERT*: `4_layer_6000_vocab_size_bert_v3`;
- *Bi-LSTM Model*: `256_hidden_dim_6000_vocab_size_1_layer_lstm_v3`;
- *Tokenizer*: `6000_subword_tokenizer`.

Note that `4_layer_6000_vocab_size_bert_v3` has its own built-in tokenizer, which happens to be identical to `6000_subword_tokenizer`.

---

### Usage examples
#### Standard models
Loading a model triggers download of the selected pretrained Ulysses segmenter models using [Ulysses Fetcher](https://github.com/ulysses-camara/ulysses-fetcher). Downloaded models are cached locally for future uses.

##### BERTSegmenter
```python
import segmentador

segmenter_bert = segmentador.BERTSegmenter(device="cpu")

sample_text = """
PROJETO DE LEI N. 0123 (Da Sra. Alguém)
Dispõe de algo. O Congresso Nacional decreta:
Artigo 1. Este projeto de lei não tem efeito.
    a) Item de exemplo; b) Segundo item; ou c) Terceiro item.
Artigo 2. Esta lei passa a vigorar na data de sua publicação.
"""

seg_result = segmenter_bert(sample_text, return_logits=True)

print(seg_result.segments)
# [
#     'PROJETO DE LEI N. 0123 ( Da Sra. Alguém )',
#     'Dispõe de algo.',
#     'O Congresso Nacional decreta :',
#     'Artigo 1. Este projeto de lei não tem efeito.',
#     'a ) Item de exemplo ;',
#     'b ) Segundo item ; ou',
#     'c ) Terceiro item.',
#     'Artigo 2. Esta lei passa a vigorar na data de sua publicação.',
# ]

print(seg_result.logits)
# [[ 7.75678301  0.15893856 -2.88991857 -5.1139946 ]
#  [10.15956116 -2.35737801 -3.08267331 -4.61426926]
#  [10.86083889 -2.60591483 -4.09350395 -4.16544533]
#  ...
#  [ 9.71361065 -1.58287859 -3.04793835 -5.78309536]
#  [ 2.31029105  7.32992315 -2.93384242 -7.3394866 ]]
```

##### LSTMSegmenter
```python
import segmentador

segmenter_lstm = segmentador.LSTMSegmenter(device="cpu")

sample_text = """
PROJETO DE LEI N. 0123 (Da Sra. Alguém)
Dispõe de algo. O Congresso Nacional decreta:
Artigo 1. Este projeto de lei não tem efeito.
    a) Item de exemplo; b) Segundo item; ou c) Terceiro item.
Artigo 2. Esta lei passa a vigorar na data de sua publicação.
"""

seg_result = segmenter_lstm(sample_text, return_logits=True)

print(seg_result.segments)
# [
#    'PROJETO DE LEI N. 0123 ( Da Sra. Alguém )',
#    'Dispõe de algo.',
#    'O Congresso Nacional decreta :',
#    'Artigo 1. Este projeto de lei não tem efeito.',
#    'a ) Item de exemplo ;',
#    'b ) Segundo item ; ou',
#    'c ) Terceiro item.',
#    'Artigo 2. Esta lei passa a vigorar na data de sua publicação.',
# ]

print(seg_result.logits)
# [[  6.2647295   -8.58741379   5.64134645  -7.10431194]
#  [  7.73504782  -2.77080107  -5.28328753 -10.26550961]
#  [ 10.03150749  -7.33715487  -5.94148588  -7.88663769]
#  ...
#  [  6.64764452  -2.28969622  -3.06246185  -8.4958601 ]
#  [ -0.75093395   5.79272366   2.84845114  -8.5399065 ]]
```

##### Local files or Huggingface HUB models
You can also provide local models (or compatible Huggingface HUB models) to initialize the segmenter model weights, by providing the `uri_model` and `uri_tokenizer` arguments, as depicted in the exemple below. Remember that BERT models often have their own tokenizer built-in, wheres LSTM models do not. Therefore, providing a tokenizer model for LSTM models is a requirement, whereas for BERT models is optional.
```python
segmenter_bert = segmentador.BERTSegmenter(
    uri_model="<path_to_local_model_or_hf_hub_model_name>",
    uri_tokenizer=None,
)

segmenter_lstm = segmentador.LSTMSegmenter(
    uri_model="<path_to_local_model>",
    uri_tokenizer="<path_to_model_tokenizer>",
)
```

---

#### Noise subsegment removal
Tokens are classified as one out of 4 available classes: No-op (0), Segment (1), Noise Start (2), and Noise End (3). Tokens in-between any pair of `Noise Start` (inclusive) and the closest `Noise End` or `Segment` (either exclusive) can be removed during the segmentation by using the argument `remove_noise_subsegments=True` to the segmenter model, as shown below:

```python
seg_result = segmenter(sample_text, ..., remove_noise_subsegments=True)
```

---

#### Quantization in ONNX format
We provide support for models in ONNX format (and also functions to convert from pytorch to such format), which are highly optimized and also support weight quantization. We apply 8-bit dynamic quantization. Effects of quantization in segmenter models are analyzed in [Optimization and Compression notebook](./notebooks/7_optimization_and_compression.ipynb).

First, in order to use models in ONNX format you need to install some optional dependencies, as shown in [Installation](#installation) section. Then, you need to create the ONNX quantized model using the `segmentador.optimize` subpackage API:

```python
import segmentador.optimize

# Load BERT Torch model
segmenter_bert = segmentador.BERTSegmenter()

# Create ONNX BERT model
quantized_model_paths = segmentador.optimize.quantize_model(
    segmenter_bert,
    model_output_format="onnx",
    verbose=True,
)
```

Lastly, load the optimized models with appropriate classes from `segmentador.optimize` module. While the ONNX segmenter model configuration may differ from their standard (Torch format) version, its inference usage remains the same:

```python
# Load ONNX model
segmenter_bert_quantized = segmentador.optimize.ONNXBERTSegmenter(
    uri_model=quantized_model_paths.output_uri,
    uri_tokenizer=segmenter_bert.tokenizer.name_or_path,
)

seg_result = segmenter_bert_quantized(sample_text, return_logits=True)
```

The procedure shown above is analogous for ONNX Bi-LSTM models:

```python
import segmentador.optimize

# Load Bi-LSTM standard model
segmenter_lstm = segmentador.LSTMSegmenter()

# Create ONNX Bi-LSTM model
quantized_lstm_paths = segmentador.optimize.quantize_model(
    segmenter_lstm,
    model_output_format="onnx",
    verbose=True,
)

# Load ONNX model
segmenter_lstm_quantized = segmentador.optimize.ONNXLSTMSegmenter(
    uri_model=quantized_lstm_paths.output_uri,
    uri_tokenizer=segmenter_lstm.tokenizer.name_or_path,
)

seg_result = segmenter_lstm_quantized(curated_df_subsample, return_logits=True)
```

---

### Train and evaluation Data

| Ver. | Dataset                                       | Size (MB) | [HF datasets](https://github.com/huggingface/datasets) format | .tsv format      |
| :--- | :---                                          | :---      | :---              | :---             |
| v1   | Weakly supervised                             | 99.7      | [Link 1](https://cloud.andrelab.icmc.usp.br/s/NoKR3AFGysXFySH) / [Link 2](https://drive.google.com/file/d/1eD832P7r31ey8lhHqDc_uLnyg6TdKlpd/view?usp=share_link) | [Link 1](https://cloud.andrelab.icmc.usp.br/s/KfbggCyfHWknREM) / [Link 2](https://drive.google.com/file/d/1K9Ckn0Q25vLIr_8LoTcTRA2PTho36Vnn/view?usp=share_link) |
| v2   | Active learning                               | 108.7     | [Link 1](https://cloud.andrelab.icmc.usp.br/s/WiejawW5ksJaZMS) / [Link 2](https://drive.google.com/file/d/1Wxfd4qyFCtjpsPPVY9Nx-DQYpxjQvRrJ/view?usp=share_link) | [Link 1](https://cloud.andrelab.icmc.usp.br/s/wLBHrsLfrJpm5Zk) / [Link 2](https://drive.google.com/file/d/1pjNJyd53komxvw68UanW8ESkY9ZwteDd/view?usp=share_link) |
| v2   | Active learning (curated only)                | 5.4       | [Link 1](https://cloud.andrelab.icmc.usp.br/s/eHzSR5Ai8bAEQi7) / [Link 2](https://drive.google.com/file/d/1pq6fKUJhzVQ5hcbqwrlgPeXBoFUsqFKQ/view?usp=share_link) | [Link 1](https://cloud.andrelab.icmc.usp.br/s/7Aniq5Ka9nn7Y43) / [Link 2](https://drive.google.com/file/d/1zHVFJc-QphKzrXkyuEeOeiEW2hh46Jtd/view?usp=share_link) |
| v2   | Extra: Emendas legislativas                   | 0.4       | [Link 1](https://cloud.andrelab.icmc.usp.br/s/KXwcmERqMwaPskd) / [Link 2](https://drive.google.com/file/d/1ywzIVarPy6JUOWDQ0-ShKDk2PaIAEs_0/view?usp=share_link) | [Link 1](https://cloud.andrelab.icmc.usp.br/s/HzpiyToAmswFSby) / [Link 2](https://drive.google.com/file/d/11jzh8kAW7hyVeiZkWkw_CJyX9xkOc5jy/view?usp=share_link) |
| v3   | State bills, Gov. Auctions, Codes, Acts, CF88 | 4.7      | [Link 1](https://cloud.andrelab.icmc.usp.br/s/5zq5pHDfar5MC3p) / [Link 2](https://drive.google.com/file/d/1d7iEsojyq62S2gUm36DwEioOzN9_NBrh/view?usp=share_link)  | [Link 1](https://cloud.andrelab.icmc.usp.br/s/JryBfFfcGz9YiTZ) / [Link 2](https://drive.google.com/file/d/1_RXd9jOZESftvdNKwmj2IxuLFBm-iAzy/view?usp=share_link) |

Note: you can easily convert the HF datasets into segments using the Ulysses segmenter `.generate_segments_from_ids(input_ids=..., label_ids=...)` method as follows:

```python
import datasets
import segmentador

segmenter = segmentador.Segmenter(device="cpu")
dt = datasets.load_from_disk("./dataset_ulysses_segmenter_train_v3")

all_segs: list[str] = []

for item in dt:
    segs: list[str] = segmenter.generate_segments_from_ids(
        input_ids=item["input_ids"],
        label_ids=item["labels"],
    )
    all_segs.extend(segs)

print(len(all_segs))
```


---

### Package tests
Tests for this package are run using tox, pytest, pylint (codestyle), and mypy (static type checking).
```bash
https://github.com/ulysses-camara/ulysses-segmenter
python -m pip install ".[test]"
python -m tox
```

---

### License
[MIT.](./LICENSE)

---

### Citation
```bibtex
@inproceedings{
    paper="",
    author="",
    date="",
}
```

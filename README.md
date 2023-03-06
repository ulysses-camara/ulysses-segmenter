[![Tests](https://github.com/ulysses-camara/ulysses-segmenter/actions/workflows/tests.yml/badge.svg)](https://github.com/ulysses-camara/ulysses-segmenter/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/ulysses-segmenter/badge/?version=latest)](https://ulysses-segmenter.readthedocs.io/en/latest/?badge=latest)

## Brazilian Legal Text Segmenter
This project presents a Legal Text Segmenter for Portuguese-Brazilian language.

The segmentation problem is formalized here by a 4-multiclass token-wise classification problem. Each token can be classified as follows:

|Class |Description             |
| :--- | :---                   |
|0     |No-op                   |
|1     |Start of sentence       |
|2     |Start of noise sequence |
|3     |End of noise sequence   |


In a curated dataset, comprised of ground-truth legal text segments, Ulysses Segmenter achieves higher Precision and Recall for the Class 1 (Segment) than other available popular segmentation tools, such as [NLTK](https://github.com/nltk/nltk), [SpaCy](https://github.com/explosion/spaCy), and [LexNLP](https://github.com/LexPredict/lexpredict-lexnlp), with the latter being suitable for segmenting legal texts. In the table below we compare these algorithms against Ulysses Segmenter, showing results for both estimated Precision and Recall by using over 2000 unseen curated examples:


| Segmentation Method             | Precision    | Recall       | Size (MiB) |
|:---                             |:---          |:---          | :--------- |
| NLTK (v3.7)                     | 13.1197%     | 19.6861%     | --         |
| SpaCy (v3.5.0)                  | 13.3962%     | 25.4032%     | --         |
| LexNLP (v2.2.1.0)               | 24.6631%     | 27.9249%     | --         |
| Ulysses Segmenter v1 (LSTM-512) | 96.2952%     | 92.8916%     | 37         |
| Ulysses Segmenter v1 (BERT-2)   | 96.5545%     | 93.0829%     | 74         |
| Ulysses Segmenter v2 (LSTM-256) | 96.6677%     | 94.9698%     | **25**     |
| Ulysses Segmenter v2 (BERT-2)   | 97.7987%     | 96.3871%     | 74         |
| Ulysses Segmenter v2 (BERT-4)   | **98.2213%** | **96.7523%** | 128        |

---

### Table of Contents
1. [Model details](#model-details)
    1. [Inference](#inference)
    2. [Training](#training)
2. [Trained models](#trained-models)
3. [Installation](#installation)
4. [Usage examples](#usage-examples)
    1. [Standard models (Torch format, Huggingface Transformers compatible)](#standard-models)
    2. [Quantization in ONNX format](#quantization-in-onnx-format)
    3. [Quantization in Torch JIT format](#quantization-in-torch-jit-format)
    4. [Noise subsegment removal](#noise-subsegment-removal)
5. [Experimental results](#experimental-results)
6. [Train data](#train-data)
7. [Package tests](#package-tests)
8. [License](#license)
9. [Citation](#citation)

---

### Model details

#### Inference
The trained models are Transformer Encoders (BERT) and Bidirectional LSTM (Bi-LSTM), with varyinng number of hidden layers (transformer blocks), and with support to up to 1024 subword tokens for BERT models. Since legal texts may exceed this limit, the present framework pre-segment the text into possibly overlapping 1024 subword windows automatically in a moving window fashion, feeding them to the Transformer Encoder independently. The encoder output is then combined ("pooled"), and the final prediction for each token is finally derived.

<p align="center">
    <img src="./diagrams/segmenter_inference_pipeline.png" alt="Full segmenter inference pipeline."></img>
</p>

The *pooling* operations can be one of the following:

|Pooling                            | Description                                                                                                        |
| :---                              | :---                                                                                                               |
| Max                               | Keep maximal overlapping logits.                                                                                   |
| Sum                               | Sum overlapping logits.                                                                                            |
| Gaussian (default for Bi-LSTM)    | Weight overlapping logits by a Gaussian distribution, centered at the middle of each moving window.                |
| Assymetric-Max (default for BERT) | Keep maximal overlapping logits for all classes except "No-op" (which gets the minimal overlapping logit instead). |

#### Training
The data labeling process is semi-automatic, employing several *ad-hoc* regular expressions (available in [Generate Labels from Regular Expressions notebook](./notebooks/2_generate_labels_from_regular_expressions.ipynb)).

---

### Trained models
Pretrained Ulysses segmenter models are downloaded by using the [Ulysses Fetcher](https://github.com/ulysses-camara/ulysses-fetcher) API.

The default models loaded for each algorithm is:
- *BERT*: `2_layer_6000_vocab_size_bert`.
- *Bi-LSTM Model*: `512_hidden_dim_6000_vocab_size_1_layer_lstm`.
- *Tokenizer*: `6000_subword_tokenizer`.

Note that the `2_layer_6000_vocab_size_bert` already has its own built-in tokenizer, which happens to be identical to `6000_subword_tokenizer`. Hence, providing `6000_subword_tokenizer` for BERT segmenter is unnecessary, and will give the same results if done.

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

### Usage examples
#### Standard models
When loading a model, pretrained Ulysses segmenter models are downloaded automatically and cached locally by using [Ulysses Fetcher](https://github.com/ulysses-camara/ulysses-fetcher).

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

#### Quantization in Torch JIT format

Models can also be quantized as Torch JIT format, by setting `segmentador.optimize.quantize_model(model, model_output_format="torch_jit", ...)`:
```Python
# LSTM models quantized as Torch JIT format
quantized_lstm_torch_paths = segmentador.optimize.quantize_model(
    segmenter_lstm,
    model_output_format="torch_jit",
    verbose=True,
)

segmenter_lstm_torch_quantized = segmentador.optimize.TorchJITLSTMSegmenter(
   uri_model=quantized_lstm_torch_paths.output_uri,
)

seg_result = segmenter_lstm_torch_quantized(sample_text, return_logits=True)
...
```

```Python
# BERT models quantized as Torch JIT format
quantized_bert_torch_paths = segmentador.optimize.quantize_model(
    segmenter_bert,
    model_output_format="torch_jit",
    verbose=True,
)

segmenter_bert_torch_quantized = segmentador.optimize.TorchJITBERTSegmenter(
   uri_model=quantized_bert_torch_paths.output_uri,
)

seg_result = segmenter_bert_torch_quantized(sample_text, return_logits=True)
...
```

#### Noise subsegment removal
Tokens are classified as one out of 4 available classes: No-op (0), Segment (1), Noise Start (2), and Noise End (3). Tokens between a pair of `Noise Start` (inclusive) and the closest `Noise End` or `Segment` (either exclusive) can be removed during the segmentation, by passing the argument `remove_noise_subsegments=True` to the segmenter model:

```python
seg_result = segmenter(sample_text, remove_noise_subsegments=True)
```

---

### Experimental results
Experimental results are available in [Result Analsys notebook](./notebooks/6_result_analysis.ipynb), with models tipically achieving per-class precision and recall higher than 95%, despite the problem being severely imbalanced. This same notebook also showcase some tests varying moving window size, moving window shift size, and Bidirectional LSTM models for comparison.

---

### Train data
TODO.

---

### Package tests
Tests for this package are run using Tox and Pytest.

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

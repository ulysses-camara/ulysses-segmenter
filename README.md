## Brazilian Legal Text Segmenter
This project presents a Legal Text Segmenter for Portuguese-Brazilian language.

The data labeling process is semi-automatic using several *ad-hoc* regular expressions (available in [a notebook in this repository](https://github.com/FelSiq/ulysses-segmenter/blob/master/notebooks/2_generate_labels_from_regular_expressions.ipynb)).

The trained models are Transformer Encoders, with varyinng number of hidden layers (transformer blocks), and with support to up to 1024 subword tokens. Since legal texts may exceed this limit, the present framework pre-segment the text into 1024 subword blocks automatically, feeding them to the Transformer Encoder independently.

The segmentation problem is formalized by a 4-multiclass token-wise classification problem. Each token can be classified as follows:


|Class |Description             |
| :--- | :---                   |
|0     |No-op                   |
|1     |Start of sentence       |
|2     |Start of noise sequence |
|3     |End of noise sequence   |

---

### Table of Contents
1. [Trained models](#trained-models)
2. [Experimental results](#experimental-results)
3. [Train data](#train-data)
4. [License](#license)

---

### Trained models
TODO.

### Experimental results
Experimental results are available in [a notebook in this repository](https://github.com/FelSiq/ulysses-segmenter/blob/master/notebooks/4_result_analysis.ipynb), with models tipically achieving per-class precision and recall higher than 95%, despite the problem being severely imbalanced.

### Train data
TODO.

### License
[MIT.](https://github.com/FelSiq/ulysses-segmenter/blob/master/LICENSE)

## Brazilian Legal Text Segmenter
This project presents a Legal Text Segmenter for Portuguese-Brazilian language.

The segmentation problem is formalized here by a 4-multiclass token-wise classification problem. Each token can be classified as follows:

|Class |Description             |
| :--- | :---                   |
|0     |No-op                   |
|1     |Start of sentence       |
|2     |Start of noise sequence |
|3     |End of noise sequence   |

---

### Table of Contents
1. [About the Model](#about-the-model)
    1. [Inference](#inference)
    2. [Training](#training)
2. [Trained models](#trained-models)
3. [Experimental results](#experimental-results)
4. [Train data](#train-data)
5. [License](#license)

---

### Model details

#### Inference
The trained models are Transformer Encoders, with varyinng number of hidden layers (transformer blocks), and with support to up to 1024 subword tokens. Since legal texts may exceed this limit, the present framework pre-segment the text into possibly overlapping 1024 subword blocks automatically in a moving window fashion, feeding them to the Transformer Encoder independently. The encoder output is then combined ("pooled"), and the final prediction for each token is finally derived.

<p align="center">
	<img src="./diagrams/segmenter_inference_pipeline.png" alt="Full segmenter inference pipeline."></img>
</p>

The *pooling* operations can be one of the following:

|Pooling                   | Description                                                                                                        |
| :---                     | :---                                                                                                               |
| Max                      | Keep maximal overlapping logits.                                                                                   |
| Sum                      | Sum overlapping logits.		                                                                                    |
| Gaussian                 | Weight overlapping logits by a Gaussian distribution, centered at the middle of each block.                        |
| Assymetric-Max (default) | Keep maximal overlapping logits for all classes except "No-op" (which gets the minimal overlapping logit instead). |


#### Training
The data labeling process is semi-automatic, employing several *ad-hoc* regular expressions (available in [a notebook in this repository](https://github.com/FelSiq/ulysses-segmenter/blob/master/notebooks/2_generate_labels_from_regular_expressions.ipynb)).


### Trained models
TODO.

### Experimental results
Experimental results are available in [a notebook in this repository](https://github.com/FelSiq/ulysses-segmenter/blob/master/notebooks/4_result_analysis.ipynb), with models tipically achieving per-class precision and recall higher than 95%, despite the problem being severely imbalanced.

### Train data
TODO.

### License
[MIT.](https://github.com/FelSiq/ulysses-segmenter/blob/master/LICENSE)

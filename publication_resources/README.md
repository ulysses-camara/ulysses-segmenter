## Resources regarding Ulysses Segmenter publication

Resources related to the Ulysses Segmenter publication.

Below we link modules from this directory to related Sections from the publication.

### Modules
- Section 5 (weak supervision):
    - `weak_supervision_data_preparation.py`
    - `train_tokenizer_from_leg_bills.py`
    - `train_weakly_supervised_bert_models.py`
    - `train_weakly_supervised_lstm_models.py`
    - `evaluate_weakly_supervised_models.py`
- Section 6 (active learning):
    - `train_active_learning_models.py`
    - `active_learning_label_curation.py`
    - `train_with_active_learning_vs_random_data.py`
- Section 7 (model assessment):
    - `approx_recall_and_precision.py`
    - `baseline_test_crf.py`
    - `baseline_test_lexnlp.py`
    - `baseline_test_nltk.py`
    - `baseline_test_spacy.py`
    - `baseline_test_topictiling.py`
    - `baseline_test_gazetteer.py`
- Section 8 (inference):
    - `evaluate_shrunken_windows.py`
    - `evaluate_inference_window_params.py`
- Section 9 (extra legislative data):
    - `fine_tune_state_leg_and_other_docs.py`
- Section 10 (international legislation):
    - `few_shot_fine_tuning_international_leg.py`
    - `evaluate_regex_international_leg.py`

### Pretrained models and Datasets

In order to execute some scripts in this directory, you need to download some pretrained models and datasets.
You can get more information in the following links:

- [Models](https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#available-models)
- [Datasets](https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data)

It should be very straightforward to determine which models and datasets are needed for each script due to all names being standardized.

### Results

Most results used to fill the publication tables and plots can be found in the [results](./results) directory.
Re-runing related scrips should provide very close results up to pseudo-random factors.

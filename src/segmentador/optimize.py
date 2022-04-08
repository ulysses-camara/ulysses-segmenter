import pickle

import onnxruntime
import onnx
import transformers
import datasets
import numpy as np

try:
    import optimum.onnxruntime

except ImportError as e_import:
    raise ImportError(
        "Optinal dependency 'optimum.onnxruntime' not found, which is necessary to "
        "use quantized segmenter models. Please install it with the following "
        "command:\n\n"
        "python -m pip install optimum[onnxruntime]\n\n"
        "See https://huggingface.co/docs/optimum/index for more information."
    ) from e_import

from . import _base


class QONNXBERTSegmenter(_base.BaseSegmenter):
    """Quantized ONNX BERT segmenter for PT-br legal text data.

    Uses a pretrained Transformer Encoder to segment Brazilian Portuguese legal texts.
    The pretrained models support texts up to 1024 subwords. Texts larger than this
    value are pre-segmented into 1024 subword blocks, and each block is feed to the
    segmenter individually.

    Parameters
    ----------
    uri_model : str
        URI to load pretrained model from. If `local_files_only=True`, then it must
        be a local file.

    uri_tokenizer : str
        URI to pretrained text Tokenizer.

    uri_onnx_config : str
        URI to pickled ONNX configuration.

    inference_pooling_operation : {"max", "sum", "gaussian", "assymetric-max"},\
            default="assymetric-max"
        Specify the strategy used to combine logits during model inference for documents
        larger than 1024 subword tokens. Larger documents are sharded into possibly overlapping
        windows of 1024 subwords each. Thus, a single token may have multiple logits (and,
        therefore, predictions) associated with it. This argument defines how exactly the
        logits should be combined in order to derive the final verdict for that said token.
        The possible choices for this argument are:
        - `max`: take the maximum logit of each token;
        - `sum`: sum the logits associated with the same token;
        - `gaussian`: build a gaussian filter that weights higher logits based on how close
            to the window center they are, diminishing its weights closer to the window
            limits; and
        - `assymetric-max`: take the maximum logit of each token for all classes other than
            the `No-operation` class, which in turn receives the minimum among all corresponding
            logits instead.

    local_files_only : bool, default=True
        If True, will search only for local pretrained model and tokenizers.
        If False, may download models from Huggingface HUB, if necessary.

    cache_dir_model : str, default="../cache/models"
        Cache directory for transformer encoder model.

    cache_dir_tokenizer : str, default="../cache/tokenizers"
        Cache directory for text tokenizer.
    """
    def __init__(
        self,
        uri_model: str,
        uri_onnx_config: str,
        uri_tokenizer: str,
        inference_pooling_operation: t.Literal[
            "max", "sum", "gaussian", "assymetric-max"
        ] = "assymetric-max",
        local_files_only: bool = True,
        cache_dir_tokenizer: str = "../cache/tokenizers",
    ):
        super().__init__(
            uri_tokenizer=uri_tokenizer if uri_tokenizer is not None else uri_model,
            local_files_only=local_files_only,
            inference_pooling_operation=inference_pooling_operation,
            device="cpu",
            cache_dir_tokenizer=cache_dir_tokenizer,
        )

        with open(uri_onnx_config, "rb") as f_in:
            onnx_config = pickle.load(f_in)

        self._model: optimum.onnxruntime.ORTModel = optimum.onnxruntime.ORTModel(
            uri_model,
            onnx_config,
        )

    def eval(self) -> "QONNXBERTSegmenter":
        """No-op method, created only to keep API consistent."""
        return self

    def train(self) -> "QONNXBERTSegmenter":
        """No-op method, created only to keep API consistent."""
        return self

    def _predict_minibatch(
        self,
        minibatch: t.Union[datasets.Dataset, transformers.BatchEncoding],
    ) -> npt.NDArray[np.float64]:
        """Predict a tokenized minibatch."""
        if not isinstance(minibatch, datasets.Dataset):
            minibatch = datasets.Dataset.from_dict(minibatch)

        model_out = self._model.evaluation_loop(minibatch)
        model_out = model_out.predictions

        logits = np.asfarray(model_out).astype(np.float64, copy=False)

        return logits

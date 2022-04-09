"""Apply quantization and hardware-specific optimizations in segmenter models."""
import typing as t
import pickle
import os
import pathlib
import warnings

import onnxruntime
import transformers
import datasets
import torch
import torch.nn
import numpy as np
import numpy.typing as npt

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

try:
    import colorama

except ImportError as e_import:
    warnings.warn(
        message= (
            "Optional dependency 'colorama' not found. The quantization output will be colorless. "
            "In order to (optionally) fix this issue, use the following command:\n\n"
            "python -m pip install colorama\n\n"
            "See https://pypi.org/project/colorama/ for more information."
        ),
        category=ImportWarning,
    )
    colorama = None

from . import _base
from . import segmenter


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
            minibatch = datasets.Dataset.from_dict(minibatch)  # type: ignore

        model_out = self._model.evaluation_loop(minibatch)
        model_out = model_out.predictions

        logits = np.asfarray(model_out).astype(np.float64, copy=False)

        return logits


class ONNXLSTMSegmenter(_base.BaseSegmenter):
    def __init__(
        self,
        uri_model: str,
        uri_tokenizer: str,
        inference_pooling_operation: t.Literal[
            "max", "sum", "gaussian", "assymetric-max"
        ] = "gaussian",
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

        self._model = onnxruntime.InferenceSession(uri_model)

    def eval(self) -> "ONNXBERTSegmenter":
        """No-op method, created only to keep API consistent."""
        return self

    def train(self) -> "ONNXBERTSegmenter":
        """No-op method, created only to keep API consistent."""
        return self

    def _predict_minibatch(
        self,
        minibatch: t.Union[datasets.Dataset, transformers.BatchEncoding],
    ) -> npt.NDArray[np.float64]:
        """Predict a tokenized minibatch."""
        input_ids = minibatch["input_ids"].detach().cpu().numpy()
        input_ids = np.atleast_2d(input_ids)

        logits = self._model.run(["logits"], dict(input_ids=input_ids))
        logits = np.asfarray(logits).astype(np.float64, copy=False)

        if logits.ndim > 3:
            logits = logits.squeeze(0)

        return logits


class QuantizeOutputBERT(t.NamedTuple):
    """BERT quantization output paths."""

    onnx_config_uri: str
    onnx_base_uri: str
    onnx_optimized_uri: str
    onnx_quantized_uri: str
    output_uri: str


class QuantizeOutputLSTM(t.NamedTuple):
    """LSTM quantization output paths."""

    output_uri: str


def quantize_bert_model(
    model: segmenter.BERTSegmenter,
    quantized_model_name: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    intermediary_onnx_optimized_model_name: t.Optional[str] = None,
    onnx_config_name: t.Optional[str] = None,
    quantized_models_dir: str = "./quantized_models",
    optimization_level: int = 99,
    check_cached: bool = True,
    verbose: bool = False,
) -> QuantizeOutputBERT:
    model_config: transformers.BertConfig = model.model.config  # type: ignore
    c_nhl = model_config.num_hidden_layers
    c_vs = model.tokenizer.vocab_size

    quantized_model_name = (
        quantized_model_name or f"q_{c_vs}_vocab_size_{c_nhl}_layers_bert_model.onnx"
    )
    intermediary_onnx_model_name = (
        intermediary_onnx_model_name or f"{quantized_model_name}-base.temp"
    )
    intermediary_onnx_optimized_model_name = (
        intermediary_onnx_optimized_model_name or f"{quantized_model_name}-opt.temp"
    )
    onnx_config_name = onnx_config_name or f"{quantized_model_name}.config"

    pathlib.Path(quantized_models_dir).mkdir(exist_ok=True, parents=True)

    onnx_config_uri = os.path.join(quantized_models_dir, onnx_config_name)
    onnx_base_uri = os.path.join(quantized_models_dir, intermediary_onnx_model_name)
    onnx_optimized_uri = os.path.join(quantized_models_dir, intermediary_onnx_optimized_model_name)
    onnx_quantized_uri = os.path.join(quantized_models_dir, quantized_model_name)

    ret = QuantizeOutputBERT(
        onnx_config_uri=onnx_config_uri,
        onnx_base_uri=onnx_base_uri,
        onnx_optimized_uri=onnx_optimized_uri,
        onnx_quantized_uri=onnx_quantized_uri,
        output_uri=onnx_quantized_uri,
    )

    if check_cached and os.path.isfile(onnx_quantized_uri):
        if verbose:
            print(f"Found cached model in '{onnx_quantized_uri}'. Skipping model quantization.")

        return ret

    optimization_config = optimum.onnxruntime.configuration.OptimizationConfig(
        optimization_level=optimization_level,
    )
    onnx_config = transformers.models.bert.BertOnnxConfig(model_config)

    transformers.onnx.export(
        model=model.model,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        config=onnx_config,
        opset=max(15, onnx_config.default_onnx_opset),
        output=pathlib.Path(onnx_base_uri),
    )

    optimizer = optimum.onnxruntime.ORTOptimizer(
        model=model.model,
        tokenizer=model.tokenizer,
        feature="token-classification",
    )

    optimizer.export(
        onnx_model_path=onnx_base_uri,
        onnx_optimized_model_output_path=onnx_optimized_uri,
        optimization_config=optimization_config,
    )

    onnxruntime.quantization.quantize_dynamic(
        model_input=onnx_optimized_uri,
        model_output=onnx_quantized_uri,
        weight_type=onnxruntime.quantization.QuantType.QUInt8,
        optimize_model=False,
        per_channel=True,
        extra_options=dict(
            EnableSubgraph=True,
            MatMulConstBOnly=True,
        ),
    )

    with open(onnx_config_uri, "wb") as f_out:
        pickle.dump(onnx_config, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        print(
            f"Saved quantized BERT (ONNX format) in {c_blu}'{onnx_quantized_uri}'{c_rst}, and "
            f"its configuration file in {c_blu}'{onnx_config_uri}'{c_rst}. "
            "To use it, load a BERT segmenter model as:\n\n"
            f"{__name__}.{QONNXBERTSegmenter.__name__}(\n"
            f"   {c_ylw}uri_model={c_blu}'{onnx_quantized_uri}'{c_rst},\n"
            f"   uri_tokenizer='{model.tokenizer.name_or_path}',\n"
            f"   {c_ylw}uri_onnx_config={c_blu}'{onnx_config_uri}'{c_rst},\n"
            "   ...,\n"
            ")"
        )

    return ret


def quantize_lstm_model(
    model: segmenter.LSTMSegmenter,
    quantized_model_name: t.Optional[str] = None,
    quantized_models_dir: str = "./quantized_models",
    modules_to_quantize: t.Union[
        set[t.Type[torch.nn.Module]], tuple[t.Type[torch.nn.Module], ...]
    ] = (
        torch.nn.Embedding,
        torch.nn.LSTM,
        torch.nn.Linear,
    ),
    check_cached: bool = True,
    verbose: bool = False,
) -> QuantizeOutputLSTM:
    quantized_model_name = quantized_model_name or (
        f"q_{model.lstm_hidden_layer_size}_hidden_dim_"
        f"{model.vocab_size}_vocab_size_"
        f"{model.lstm_num_layers}_num_layers"
        f"lstm.pt"
    )

    pathlib.Path(quantized_models_dir).mkdir(exist_ok=True, parents=True)
    output_uri = os.path.join(quantized_models_dir, quantized_model_name)

    ret = QuantizeOutputLSTM(output_uri=output_uri)

    if check_cached and os.path.isfile(output_uri):
        if verbose:
            print(f"Found cached model in '{output_uri}'. Skipping model quantization.")

        return ret

    pytorch_module = model.model

    if torch.nn.Embedding in modules_to_quantize:
        pytorch_module.embeddings.qconfig = (  # type: ignore
            torch.quantization.float_qparams_weight_only_qconfig
        )

    quantized_pytorch_module = torch.quantization.quantize_dynamic(
        pytorch_module,
        set(modules_to_quantize),
        dtype=torch.qint8,
    )

    torch.save(
        obj=quantized_pytorch_module.state_dict(),
        f=output_uri,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )

    if verbose:
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        print(
            f"Saved quantized Pytorch module in {c_blu}'{output_uri}'{c_rst}. "
            "To use it, load a LSTM segmenter model as:\n\n"
            f"LSTMSegmenter(\n"
            f"   {c_ylw}uri_model={c_blu}'{output_uri}'{c_rst},\n"
            f"   uri_tokenizer='{model.tokenizer.name_or_path}',\n"
            f"   {c_ylw}quantize_weights={c_blu}True{c_rst},\n"
            "   ...,\n"
            ")"
        )

    return ret


def quantize_model(
    model: t.Union[segmenter.BERTSegmenter, segmenter.LSTMSegmenter],
    quantized_model_name: t.Optional[str] = None,
    quantized_models_dir: str = "./quantized_models",
    optimization_level: int = 99,
    check_cached: bool = True,
    verbose: bool = False,
) -> t.Union[QuantizeOutputBERT, QuantizeOutputLSTM]:
    if not isinstance(model, (segmenter.BERTSegmenter, segmenter.LSTMSegmenter)):
        raise TypeError(
            f"Unknown segmenter type for quantization: '{type(model)}'. Please "
            "provide either BERTSegmenter or LSTMSegmenter."
        )

    common_kwargs: dict[str, t.Any] = dict(
        model=model,
        quantized_model_name=quantized_model_name,
        quantized_models_dir=quantized_models_dir,
        check_cached=check_cached,
        verbose=verbose,
    )

    if isinstance(model, segmenter.LSTMSegmenter):
        return quantize_lstm_model(**common_kwargs)

    return quantize_bert_model(**common_kwargs, optimization_level=optimization_level)

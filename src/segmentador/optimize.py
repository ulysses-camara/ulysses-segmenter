"""Apply quantization and hardware-specific optimizations in segmenter models."""
import typing as t
import pickle
import os
import pathlib
import warnings
import collections

import onnxruntime
import transformers
import datasets
import torch
import torch.nn
import torch.onnx
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
        message=(
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


class ONNXBERTSegmenter(_base.BaseSegmenter):
    """BERT segmenter in ONNX format.

    The ONNX format support faster inference, quantized and optimized models with
    hardware-specific instructions.

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


class QuantizationOutputONNX(t.NamedTuple):
    """Output paths for quantization as ONNX format."""

    onnx_base_uri: str
    onnx_optimized_uri: str
    onnx_quantized_uri: str
    output_uri: str
    onnx_config_uri: t.Optional[str] = None


class QuantizationOutputTorch(t.NamedTuple):
    """Quantization output paths as Torch format."""

    output_uri: str


def _build_onnx_default_uris(
    model_name: str,
    model_attributes: dict[str, t.Any],
    quantized_models_dir: str,
    quantized_model_name: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    intermediary_onnx_optimized_model_name: t.Optional[str] = None,
    onnx_config_name: t.Optional[str] = None,
) -> QuantizationOutputONNX:
    if not quantized_model_name:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        quantized_model_name = f"q_{attrs_to_name}_{model_name}_model.onnx"

    if not quantized_model_name.endswith(".onnx"):
        quantized_model_name += ".onnx"

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

    paths = QuantizationOutputONNX(
        onnx_base_uri=onnx_base_uri,
        onnx_optimized_uri=onnx_optimized_uri,
        onnx_quantized_uri=onnx_quantized_uri,
        output_uri=onnx_quantized_uri,
        onnx_config_uri=onnx_config_uri,
    )

    return paths


def _build_torch_default_uris(
    model_name: str,
    model_attributes: dict[str, t.Any],
    quantized_models_dir: str,
    quantized_model_name: t.Optional[str] = None,
) -> QuantizationOutputTorch:
    if not quantized_model_name:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        quantized_model_name = f"q_{attrs_to_name}_{model_name}_model.pt"

    if not quantized_model_name.endswith(".pt"):
        quantized_model_name += ".pt"

    pathlib.Path(quantized_models_dir).mkdir(exist_ok=True, parents=True)
    output_uri = os.path.join(quantized_models_dir, quantized_model_name)

    paths = QuantizationOutputTorch(output_uri=output_uri)

    return paths


def quantize_bert_model_as_onnx(
    model: segmenter.BERTSegmenter,
    quantized_model_name: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    intermediary_onnx_optimized_model_name: t.Optional[str] = None,
    onnx_config_name: t.Optional[str] = None,
    quantized_models_dir: str = "./quantized_models",
    optimization_level: int = 99,
    onnx_opset_version: int = 15,
    check_cached: bool = True,
    verbose: bool = False,
) -> QuantizationOutputONNX:
    model_config: transformers.BertConfig = model.model.config  # type: ignore

    model_attributes: dict[str, t.Any] = collections.OrderedDict((
        ("num_layers", model_config.num_hidden_layers),
        ("vocab_size", model.tokenizer.vocab_size),
        ("opt_level", optimization_level),
    ))

    paths = _build_onnx_default_uris(
        model_name="bert",
        model_attributes=model_attributes,
        quantized_models_dir=quantized_models_dir,
        quantized_model_name=quantized_model_name,
        intermediary_onnx_model_name=intermediary_onnx_model_name,
        intermediary_onnx_optimized_model_name=intermediary_onnx_optimized_model_name,
        onnx_config_name=onnx_config_name,
    )

    if check_cached and os.path.isfile(onnx_quantized_uri):
        if verbose:
            print(f"Found cached model in '{onnx_quantized_uri}'. Skipping model quantization.")

        return paths

    optimization_config = optimum.onnxruntime.configuration.OptimizationConfig(
        optimization_level=optimization_level,
    )
    onnx_config = transformers.models.bert.BertOnnxConfig(model_config)

    transformers.onnx.export(
        model=model.model,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        config=onnx_config,
        opset=onnx_opset_version,
        output=pathlib.Path(paths.onnx_base_uri),
    )

    optimizer = optimum.onnxruntime.ORTOptimizer(
        model=model.model,
        tokenizer=model.tokenizer,
        feature="token-classification",
    )

    optimizer.export(
        onnx_model_path=paths.onnx_base_uri,
        onnx_optimized_model_output_path=paths.onnx_optimized_uri,
        optimization_config=optimization_config,
    )

    onnxruntime.quantization.quantize_dynamic(
        model_input=paths.onnx_optimized_uri,
        model_output=paths.onnx_quantized_uri,
        weight_type=onnxruntime.quantization.QuantType.QUInt8,
        optimize_model=False,
        per_channel=True,
        extra_options=dict(
            EnableSubgraph=True,
            MatMulConstBOnly=True,
        ),
    )

    with open(paths.onnx_config_uri, "wb") as f_out:
        pickle.dump(onnx_config, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        print(
            f"Saved quantized BERT (ONNX format) in {c_blu}'{paths.onnx_quantized_uri}'{c_rst}, "
            f"and its configuration file in {c_blu}'{paths.onnx_config_uri}'{c_rst}. "
            "To use it, load a BERT segmenter model as:\n\n"
            f"{__name__}.{ONNXBERTSegmenter.__name__}(\n"
            f"   {c_ylw}uri_model={c_blu}'{paths.onnx_quantized_uri}'{c_rst},\n"
            f"   uri_tokenizer='{model.tokenizer.name_or_path}',\n"
            f"   {c_ylw}uri_onnx_config={c_blu}'{paths.onnx_config_uri}'{c_rst},\n"
            "   ...,\n"
            ")"
        )

    return paths


def quantize_lstm_model_as_onnx(
    model: segmenter.LSTMSegmenter,
    quantized_model_name: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    intermediary_onnx_optimized_model_name: t.Optional[str] = None,
    quantized_models_dir: str = "./quantized_models",
    optimization_level: int = 99,
    onnx_opset_version: int = 15,
    check_cached: bool = True,
    verbose: bool = False,
) -> QuantizationOutputONNX:
    model_attributes: dict[str, t.Any] = collections.OrderedDict((
        ("hidden_layer_dim", model.lstm_hidden_layer_size),
        ("vocab_size", model.tokenizer.vocab_size),
        ("num_layers", model.lstm_num_layers),
        ("opt_level", optimization_level),
    ))

    pytorch_module = model.model

    torch_sample_input = torch.ones(1, 256, dtype=torch.long)
    torch_sample_input = torch_sample_input.to(segmenter_lstm.model.device)

    torch.onnx.export(
        model=segmenter_lstm.model,
        args=(torch_sample_input,),
        f=intermediary_onnx_model_name,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=onnx_opset_version,
        dynamic_axes={
            "input_ids": {0: "batch_axis", 1: "sentence_length"},
            "logits": {0: "batch_axis", 1: "sentence_length"},
        },
    )

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = optimization_level
    sess_options.optimized_model_filepath = intermediary_onnx_optimized_model_name

    session = onnxruntime.InferenceSession(
        os.path.join(QUANTIZED_MODELS_DIR, "test_lstm_test.onnx"),
        sess_options,
    )


def quantize_lstm_model_as_torch(
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
) -> QuantizationOutputTorch:

    model_attributes: dict[str, t.Any] = collections.OrderedDict((
        ("num_layers", model.lstm_hidden_layer_size),
        ("vocab_size", model.tokenizer.vocab_size),
        ("num_layers", model.lstm_num_layers),
    ))

    paths = _build_torch_default_uris(
        model_name="lstm",
        model_attributes=model_attributes,
        quantized_models_dir=quantized_models_dir,
        quantized_model_name=quantized_model_name,
    )

    if check_cached and os.path.isfile(output_uri):
        if verbose:
            print(f"Found cached model in '{output_uri}'. Skipping model quantization.")

        return paths

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
        f=paths.output_uri,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )

    if verbose:
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        print(
            f"Saved quantized Pytorch module in {c_blu}'{paths.output_uri}'{c_rst}. "
            "To use it, load a LSTM segmenter model as:\n\n"
            f"LSTMSegmenter(\n"
            f"   {c_ylw}uri_model={c_blu}'{paths.output_uri}'{c_rst},\n"
            f"   uri_tokenizer='{model.tokenizer.name_or_path}',\n"
            f"   {c_ylw}quantize_weights={c_blu}True{c_rst},\n"
            "   ...,\n"
            ")"
        )

    return paths


def quantize_model(
    model: t.Union[segmenter.BERTSegmenter, segmenter.LSTMSegmenter],
    quantized_model_name: t.Optional[str] = None,
    quantized_models_dir: str = "./quantized_models",
    optimization_level: int = 99,
    model_output_format: t.Literal["onnx", "torch"] = "onnx",
    onnx_opset_version: int = 15,
    check_cached: bool = True,
    verbose: bool = False,
) -> t.Union[QuantizationOutputONNX, QuantizationOutputTorch]:
    """Generate a quantized segmenter model from an existing segmenter model.

    Parameters
    ----------
    model : segmenter.BERTSegmenter or segmenter.LSTMSegmenter
        Segmenter model to quantize.

    quantized_model_name : str or None, default=None
        Output file name.

    quantized_models_dir : str, default='./quantized_models'
        Output file directory, which the resulting quantized model will be
        stored, alongside any possible coproducts also generated during the
        quantization procedure.

    optimization_level : {0, 1, 2, 99}, default=99
        Optimization level for ONNX models. From the ONNX Runtime specification:
        0: Disable model optimization;
        1: Basic optimizations;
        2: Extended optimizations; and
        99: All optimizations (incluing layer and hardware-specific optimizations).
        See [1]_ for more informations.

    model_output_format : {"onnx", "torch"}, default="onnx"
        TODO.

    onnx_opset_version: int, default=15
        TODO.

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    verbose : bool, default=False
        If True, print information regarding the results.

    Returns
    -------
    paths : tuple of str
        Named tuple with all paths of files generated during the full quantization
        procedure.

    References
    ----------
    .. [1] Graph Optimizations in ONNX Runtime. Available at:
       https://onnxruntime.ai/docs/performance/graph-optimizations.html
    """
    if not isinstance(model, (segmenter.BERTSegmenter, segmenter.LSTMSegmenter)):
        raise TypeError(
            f"Unknown segmenter type for quantization: '{type(model)}'. Please "
            "provide either BERTSegmenter or LSTMSegmenter."
        )

    kwargs: dict[str, t.Any] = dict(
        model=model,
        quantized_model_name=quantized_model_name,
        quantized_models_dir=quantized_models_dir,
        check_cached=check_cached,
        verbose=verbose,
    )

    fn_quantization_factory: dict[tuple[t.Type[_base.BaseSegmenter], str], t.Callable] = {
        (segmenter.BERTSegmenter, "onnx"): quantize_bert_model_as_onnx,
        (segmenter.LSTMSegmenter, "torch"): quantize_lstm_model_as_torch,
        (segmenter.LSTMSegmenter, "onnx"): quantize_lstm_model_as_onnx,
    }

    try:
        fn_quantization = fn_quantization_factory[(type(model), model_output_format)]

    except KeyError as e_key:
        raise ValueError(
            f"Unsupported '{model_output_format=}' for segmenter type={type(model)}."
        ) from e_key

    if model_output_format == "onnx":
        kwargs["optimization_level"] = optimization_level
        kwargs["onnx_opset_version"] = onnx_opset_version

    return fn_quantization(**kwargs)

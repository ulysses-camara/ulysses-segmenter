"""Apply quantization and hardware-specific optimizations in segmenter models."""
import typing as t
import pickle
import os
import pathlib
import collections

import transformers
import torch
import torch.nn
import torch.onnx

from .. import _base
from .. import segmenter
from . import _optional_import_utils
from . import models


colorama = _optional_import_utils.load_optional_module("colorama")


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


QuantizationOutput = t.Union[QuantizationOutputONNX, QuantizationOutputTorch]


def _build_onnx_default_uris(
    model_name: str,
    model_attributes: dict[str, t.Any],
    quantized_model_dirpath: str,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    intermediary_onnx_optimized_model_name: t.Optional[str] = None,
    onnx_config_name: t.Optional[str] = None,
    optimization_level: t.Union[str, int] = 99,
    include_config_uri: bool = True,
) -> QuantizationOutputONNX:
    """Build default URIs for quantized output in ONNX format."""
    if not intermediary_onnx_model_name:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        intermediary_onnx_model_name = f"{attrs_to_name}_{model_name}_model"

    if not intermediary_onnx_optimized_model_name:
        intermediary_onnx_optimized_model_name = (
            f"{intermediary_onnx_model_name}_{optimization_level}_opt_level"
        )

    if not quantized_model_filename:
        quantized_model_filename = f"q_{intermediary_onnx_optimized_model_name}"

    onnx_config_name = onnx_config_name or f"{quantized_model_filename}.config"

    if not intermediary_onnx_model_name.endswith(".onnx"):
        intermediary_onnx_model_name += ".onnx"

    if not intermediary_onnx_optimized_model_name.endswith(".onnx"):
        intermediary_onnx_optimized_model_name += ".onnx"

    if not quantized_model_filename.endswith(".onnx"):
        quantized_model_filename += ".onnx"

    pathlib.Path(quantized_model_dirpath).mkdir(exist_ok=True, parents=True)

    onnx_base_uri = os.path.join(quantized_model_dirpath, intermediary_onnx_model_name)
    onnx_optimized_uri = os.path.join(
        quantized_model_dirpath, intermediary_onnx_optimized_model_name
    )
    onnx_quantized_uri = os.path.join(quantized_model_dirpath, quantized_model_filename)

    paths_dict: dict[str, str] = dict(
        onnx_base_uri=onnx_base_uri,
        onnx_optimized_uri=onnx_optimized_uri,
        onnx_quantized_uri=onnx_quantized_uri,
        output_uri=onnx_quantized_uri,
    )

    if include_config_uri:
        onnx_config_uri = os.path.join(quantized_model_dirpath, onnx_config_name)
        paths_dict["onnx_config_uri"] = onnx_config_uri

    paths = QuantizationOutputONNX(**paths_dict)

    return paths


def _build_torch_default_uris(
    model_name: str,
    model_attributes: dict[str, t.Any],
    quantized_model_dirpath: str,
    quantized_model_filename: t.Optional[str] = None,
) -> QuantizationOutputTorch:
    """Build default URIs for quantized output in Torch format."""
    if not quantized_model_filename:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        quantized_model_filename = f"q_{attrs_to_name}_{model_name}_model.pt"

    pathlib.Path(quantized_model_dirpath).mkdir(exist_ok=True, parents=True)
    output_uri = os.path.join(quantized_model_dirpath, quantized_model_filename)

    paths = QuantizationOutputTorch(output_uri=output_uri)

    return paths


def quantize_bert_model_as_onnx(
    model: segmenter.BERTSegmenter,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    intermediary_onnx_optimized_model_name: t.Optional[str] = None,
    onnx_config_name: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    optimization_level: int = 99,
    onnx_opset_version: int = 15,
    check_cached: bool = True,
    verbose: bool = False,
) -> QuantizationOutputONNX:
    """Create a quantized BERTSegmenter model as ONNX format.

    Models created from this format can be loaded for inference as:

    optimize.ONNXBERTSegmenter(
        uri_model="[quantized_model_uri]",
        uri_tokenizer=...,
        uri_onnx_config="[quantized_model_config_uri]",
        ...,
    )

    Parameters
    ----------
    model : segmenter.BERTSegmenter
        BERTSegmenter model to be quantized.

    quantized_model_filename : str or None, default=None
        Output filename. If None, a long and descriptive name will be derived from model's
        parameters.

    quantized_model_dirpath : str, default='./quantized_models'
        Path to output file directory, which the resulting quantized model will be stored,
        alongside any possible coproducts also generated during the quantization procedure.

    intermediary_onnx_model_name : str or None, default=None
        Name to save intermediary model in ONNX format in `quantized_model_dirpath`. This
        transformation is necessary to perform all necessary optimization and quantization.
        If None, a name will be derived from `quantized_model_filename`.

    intermediary_onnx_optimized_model_name : str or None, default=None
        Name to save intermediary optimized model in ONNX format in `quantized_model_dirpath`.
        This transformation is necessary to perform quantization. If None, a name will be
        derived from `quantized_model_filename`.

    onnx_config_name : str or None, default=None
        Name of pickled BERT configuration file, necessary to provide enough information
        during model loading. If None, a name will be derived from `quantized_model_filename`.

    optimization_level : {0, 1, 2, 99}, default=99
        Optimization level for ONNX models. From the ONNX Runtime specification:
        - 0: disable all optimizations;
        - 1: enable only basic optimizations;
        - 2: enable basic and extended optimizations; or
        - 99: enable all optimizations (incluing layer and hardware-specific optimizations).
        See [1]_ for more information.

    onnx_opset_version: int, default=15
        ONNX operator set version. Used only if `model_output_format='onnx'`. Check [2]_ for
        more information.

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    verbose : bool, default=False
        If True, print information regarding the results.

    Returns
    -------
    paths : tuple[str, ...]
        File URIs related from generated files during the quantization procedure. The
        final model URI can be accessed from the `output_uri` attribute.

    References
    ----------
    .. [1] Graph Optimizations in ONNX Runtime. Available at:
       https://onnxruntime.ai/docs/performance/graph-optimizations.html

    .. [2] ONNX Operator Schemas. Available at:
       https://github.com/onnx/onnx/blob/main/docs/Operators.md
    """
    optimum_onnxruntime = _optional_import_utils.load_required_module("optimum.onnxruntime")
    _optional_import_utils.load_required_module("onnxruntime")

    import onnxruntime.quantization

    model_config: transformers.BertConfig = model.model.config  # type: ignore

    model_attributes: dict[str, t.Any] = collections.OrderedDict(
        (
            ("num_layers", model_config.num_hidden_layers),
            ("vocab_size", model.tokenizer.vocab_size),
            ("pruned", bool(model_config.pruned_heads)),
        )
    )

    paths = _build_onnx_default_uris(
        model_name="bert",
        model_attributes=model_attributes,
        quantized_model_dirpath=quantized_model_dirpath,
        quantized_model_filename=quantized_model_filename,
        intermediary_onnx_model_name=intermediary_onnx_model_name,
        intermediary_onnx_optimized_model_name=intermediary_onnx_optimized_model_name,
        onnx_config_name=onnx_config_name,
        optimization_level=optimization_level,
    )

    if check_cached and os.path.isfile(paths.onnx_quantized_uri):
        if verbose:  # pragma: no cover
            print(
                f"Found cached model in '{paths.onnx_quantized_uri}'. "
                "Skipping model quantization."
            )

        return paths

    onnx_config = transformers.models.bert.BertOnnxConfig(model_config)

    if not check_cached or not os.path.isfile(paths.onnx_base_uri):
        transformers.onnx.export(
            model=model.model,  # type: ignore
            tokenizer=model.tokenizer,  # type: ignore
            config=onnx_config,
            opset=onnx_opset_version,
            output=pathlib.Path(paths.onnx_base_uri),
        )

    elif verbose:  # pragma: no cover
        print(f"Found cached ONNX model in '{paths.onnx_base_uri}'.")

    if not check_cached or not os.path.isfile(paths.onnx_optimized_uri):
        optimizer = optimum_onnxruntime.ORTOptimizer(
            model=model.model,
            tokenizer=model.tokenizer,
            feature="token-classification",
        )

        optimization_config = optimum_onnxruntime.configuration.OptimizationConfig(
            optimization_level=optimization_level,
            optimize_for_gpu=torch.device(model.device).type == "cuda",
            enable_gelu_approximation=True,
        )

        optimizer.export(
            onnx_model_path=paths.onnx_base_uri,
            onnx_optimized_model_output_path=paths.onnx_optimized_uri,
            optimization_config=optimization_config,
        )

    elif verbose:  # pragma: no cover
        print(f"Found cached ONNX model in '{paths.onnx_optimized_uri}'.")

    onnxruntime.quantization.quantize_dynamic(
        model_input=paths.onnx_optimized_uri,
        model_output=paths.onnx_quantized_uri,
        weight_type=onnxruntime.quantization.QuantType.QUInt8,
        optimize_model=False,
        per_channel=False,
        extra_options=dict(
            EnableSubgraph=True,
            MatMulConstBOnly=False,
            ForceQuantizeNoInputCheck=True,
        ),
    )

    with open(paths.onnx_config_uri or (paths.output_uri + ".config"), "wb") as f_out:
        pickle.dump(onnx_config, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:  # pragma: no cover
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        print(
            f"Saved quantized BERT (ONNX format) in {c_blu}'{paths.onnx_quantized_uri}'{c_rst}, "
            f"and its configuration file in {c_blu}'{paths.onnx_config_uri}'{c_rst}. "
            "To use it, load a BERT segmenter model as:\n\n"
            f"{__name__}.{models.ONNXBERTSegmenter.__name__}(\n"
            f"   {c_ylw}uri_model={c_blu}'{paths.onnx_quantized_uri}'{c_rst},\n"
            f"   uri_tokenizer='{model.tokenizer.name_or_path}',\n"
            f"   {c_ylw}uri_onnx_config={c_blu}'{paths.onnx_config_uri}'{c_rst},\n"
            "   ...,\n"
            ")"
        )

    return paths


def quantize_lstm_model_as_onnx(
    model: segmenter.LSTMSegmenter,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    intermediary_onnx_optimized_model_name: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    optimization_level: int = 99,
    onnx_opset_version: int = 15,
    check_cached: bool = True,
    verbose: bool = False,
) -> QuantizationOutputONNX:
    """Create a quantized LSTMSegmenter model as ONNX format.

    Models created from this format can be loaded for inference as:

    optimize.ONNXLSTMSegmenter(
        uri_model="[quantized_model_uri]",
        uri_tokenizer=...,
        ...,
    )

    Parameters
    ----------
    model : segmenter.LSTMSegmenter
        LSTMSegmenter model to be quantized.

    quantized_model_filename : str or None, default=None
        Output filename. If None, a long and descriptive name will be derived from model's
        parameters.

    quantized_model_dirpath : str, default='./quantized_models'
        Path to output file directory, which the resulting quantized model will be stored,
        alongside any possible coproducts also generated during the quantization procedure.

    intermediary_onnx_model_name : str or None, default=None
        Name to save intermediary model in ONNX format in `quantized_model_dirpath`. This
        transformation is necessary to perform all necessary optimization and quantization.
        If None, a name will be derived from `quantized_model_filename`.

    intermediary_onnx_optimized_model_name : str or None, default=None
        Name to save intermediary optimized model in ONNX format in `quantized_model_dirpath`.
        This transformation is necessary to perform quantization. If None, a name will be
        derived from `quantized_model_filename`.

    optimization_level : {0, 1, 2, 99}, default=99
        Optimization level for ONNX models. From the ONNX Runtime specification:
        - 0: disable all optimizations;
        - 1: enable only basic optimizations;
        - 2: enable basic and extended optimizations; or
        - 99: enable all optimizations (incluing layer and hardware-specific optimizations).
        See [1]_ for more information.

    onnx_opset_version: int, default=15
        ONNX operator set version. Used only if `model_output_format='onnx'`. Check [2]_ for
        more information.

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    verbose : bool, default=False
        If True, print information regarding the results.

    Returns
    -------
    paths : tuple[str, ...]
        File URIs related from generated files during the quantization procedure. The
        final model URI can be accessed from the `output_uri` attribute.

    References
    ----------
    .. [1] Graph Optimizations in ONNX Runtime. Available at:
       https://onnxruntime.ai/docs/performance/graph-optimizations.html

    .. [2] ONNX Operator Schemas. Available at:
       https://github.com/onnx/onnx/blob/main/docs/Operators.md
    """
    _optional_import_utils.load_required_module("onnxruntime")

    import onnxruntime
    import onnxruntime.quantization

    model_attributes: dict[str, t.Any] = collections.OrderedDict(
        (
            ("hidden_layer_dim", model.lstm_hidden_layer_size),
            ("vocab_size", model.tokenizer.vocab_size),
            ("num_layers", model.lstm_num_layers),
        )
    )

    paths = _build_onnx_default_uris(
        model_name="lstm",
        model_attributes=model_attributes,
        quantized_model_dirpath=quantized_model_dirpath,
        quantized_model_filename=quantized_model_filename,
        intermediary_onnx_model_name=intermediary_onnx_model_name,
        intermediary_onnx_optimized_model_name=intermediary_onnx_optimized_model_name,
        optimization_level=optimization_level,
        include_config_uri=False,
    )

    if check_cached and os.path.isfile(paths.onnx_quantized_uri):
        if verbose:  # pragma: no cover
            print(
                f"Found cached model in '{paths.onnx_quantized_uri}'. "
                "Skipping model quantization.",
            )

        return paths

    pytorch_module = model.model

    if not check_cached or not os.path.isfile(paths.onnx_base_uri):
        torch_sample_input = torch.ones(1, 256, dtype=torch.long)
        torch_sample_input = torch_sample_input.to(model.device)

        torch.onnx.export(
            model=pytorch_module,
            args=(torch_sample_input,),
            f=paths.onnx_base_uri,
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=onnx_opset_version,
            export_params=True,
            dynamic_axes=dict(
                input_ids={0: "batch_axis", 1: "sentence_length"},
                logits={0: "batch_axis", 1: "sentence_length"},
            ),
        )

    elif verbose:  # pragma: no cover
        print(f"Found cached ONNX model in '{paths.onnx_base_uri}'.")

    if not check_cached or not os.path.isfile(paths.onnx_optimized_uri):
        opt_sess_options = onnxruntime.SessionOptions()
        opt_sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel(
            optimization_level
        )
        opt_sess_options.optimized_model_filepath = paths.onnx_optimized_uri

        onnxruntime.InferenceSession(paths.onnx_base_uri, opt_sess_options)

    elif verbose:  # pragma: no cover
        print(f"Found cached ONNX model in '{paths.onnx_optimized_uri}'.")

    onnxruntime.quantization.quantize_dynamic(
        model_input=paths.onnx_optimized_uri,
        model_output=paths.onnx_quantized_uri,
        weight_type=onnxruntime.quantization.QuantType.QUInt8,
        optimize_model=False,
        per_channel=False,
        extra_options=dict(
            EnableSubgraph=True,
            MatMulConstBOnly=False,
            ForceQuantizeNoInputCheck=True,
        ),
    )

    if verbose:  # pragma: no cover
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        print(
            f"Saved quantized Pytorch module (ONNX format) in {c_blu}'{paths.output_uri}'{c_rst}. "
            "To use it, load a LSTM segmenter model as:\n\n"
            f"{__name__}.{models.ONNXLSTMSegmenter.__name__}(\n"
            f"   {c_ylw}uri_model={c_blu}'{paths.output_uri}'{c_rst},\n"
            f"   uri_tokenizer='{model.tokenizer.name_or_path}',\n"
            "   ...,\n"
            ")"
        )

    return paths


def quantize_lstm_model_as_torch(
    model: segmenter.LSTMSegmenter,
    quantized_model_filename: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
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
    """Create a quantized LSTMSegmenter model as Torch format.

    Models created from this format can be loaded for inference as:

    LSTMSegmenter(
        uri_model="[quantized_model_uri]",
        uri_tokenizer=...,
        from_quantized_weights=True,
        ...,
    )

    Parameters
    ----------
    model : segmenter.LSTMSegmenter
        LSTMSegmenter model to be quantized.

    quantized_model_filename : t.Optional[str], default=None
        Output filename. If None, a long and descriptive name will be derived from model's
        parameters.

    quantized_model_dirpath : str, default='./quantized_models'
        Path to output file directory, which the resulting quantized model will be stored,
        alongside any possible coproducts also generated during the quantization procedure.

    modules_to_quantize : tuple[t.Type[torch.nn.Module], ...], \
        default=(torch.nn.Embedding, torch.nn.LSTM, torch.nn.Linear)

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    verbose : bool, default=False
        If True, print information regarding the results.

    Returns
    -------
    paths : tuple[str, ...]
        File URIs related from generated files during the quantization procedure. The
        final model URI can be accessed from the `output_uri` attribute.
    """

    model_attributes: dict[str, t.Any] = collections.OrderedDict(
        (
            ("num_layers", model.lstm_hidden_layer_size),
            ("vocab_size", model.tokenizer.vocab_size),
            ("num_layers", model.lstm_num_layers),
        )
    )

    paths = _build_torch_default_uris(
        model_name="lstm",
        model_attributes=model_attributes,
        quantized_model_dirpath=quantized_model_dirpath,
        quantized_model_filename=quantized_model_filename,
    )

    if check_cached and os.path.isfile(paths.output_uri):
        if verbose:  # pragma: no cover
            print(f"Found cached model in '{paths.output_uri}'. Skipping model quantization.")

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

    if verbose:  # pragma: no cover
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        print(
            f"Saved quantized Pytorch module (Torch format) in {c_blu}'{paths.output_uri}'{c_rst}. "
            "To use it, load a LSTM segmenter model as:\n\n"
            f"LSTMSegmenter(\n"
            f"   {c_ylw}uri_model={c_blu}'{paths.output_uri}'{c_rst},\n"
            f"   uri_tokenizer='{model.tokenizer.name_or_path}',\n"
            f"   {c_ylw}from_quantized_weights={c_blu}True{c_rst},\n"
            "   ...,\n"
            ")"
        )

    return paths


def quantize_model(
    model: t.Union[segmenter.BERTSegmenter, segmenter.LSTMSegmenter],
    quantized_model_filename: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    optimization_level: int = 99,
    model_output_format: t.Literal["onnx", "torch"] = "onnx",
    onnx_opset_version: int = 15,
    check_cached: bool = True,
    verbose: bool = False,
    **kwargs: t.Any,
) -> QuantizationOutput:
    """Generate a quantized segmenter model from an existing segmenter model.

    This function will derive the correct quantization function from the provided
    model type (BERT or LSTM), and the `model_output_format` parameter value. Check
    ``See Also`` section for a list of specific quantization functions.

    Parameters
    ----------
    model : segmenter.BERTSegmenter or segmenter.LSTMSegmenter
        Segmenter model to quantize.

    quantized_model_filename : str or None, default=None
        Output filename. If None, a long and descriptive name will be derived from model's
        parameters.

    quantized_model_dirpath : str, default='./quantized_models'
        Path to output file directory, which the resulting quantized model will be stored,
        alongside any possible coproducts also generated during the quantization procedure.

    optimization_level : {0, 1, 2, 99}, default=99
        Optimization level for ONNX models. From the ONNX Runtime specification:
        - 0: disable all optimizations;
        - 1: enable only basic optimizations;
        - 2: enable basic and extended optimizations; or
        - 99: enable all optimizations (incluing layer and hardware-specific optimizations).
        See [1]_ for more information.

    model_output_format : {'onnx', 'torch'}, default='onnx'
        Output format of quantized model. This option also determines how exactly inference with
        the quantized model will be done. See ``See Also`` section for information about specific
        configuratins of model types and output formats.

    onnx_opset_version: int, default=15
        ONNX operator set version. Used only if `model_output_format='onnx'`. Check [2]_ for
        more information.

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    verbose : bool, default=False
        If True, print information regarding the results.

    **kwargs : dict
        Additional parameters passed to quantization function.

    Returns
    -------
    paths : tuple of str
        Named tuple with all paths of files generated during the full quantization
        procedure.

    See Also
    --------
    quantize_lstm_model_as_onnx : quantize LSTMSegmenter as model_output_format='onnx'.
    quantize_lstm_model_as_torch : quantize LSTMSegmenter as model_output_format='torch'.
    quantize_bert_model_as_onnx : quantize BERTSegmenter as model_output_format='onnx'.

    References
    ----------
    .. [1] Graph Optimizations in ONNX Runtime. Available at:
       https://onnxruntime.ai/docs/performance/graph-optimizations.html

    .. [2] ONNX Operator Schemas. Available at:
       https://github.com/onnx/onnx/blob/main/docs/Operators.md
    """
    if not isinstance(model, (segmenter.BERTSegmenter, segmenter.LSTMSegmenter)):
        raise TypeError(
            f"Unknown segmenter type for quantization: '{type(model)}'. Please "
            "provide either BERTSegmenter or LSTMSegmenter."
        )

    if model_output_format not in {"onnx", "torch"}:
        raise ValueError(
            f"Unsupported '{model_output_format=}'. Please choose either 'onnx' or 'torch'."
        )

    fn_kwargs: dict[str, t.Any] = dict(
        model=model,
        quantized_model_filename=quantized_model_filename,
        quantized_model_dirpath=quantized_model_dirpath,
        check_cached=check_cached,
        verbose=verbose,
    )

    fn_quantization_factory: dict[
        tuple[t.Type[_base.BaseSegmenter], str], t.Callable[..., QuantizationOutput]
    ] = {
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
        fn_kwargs["optimization_level"] = optimization_level
        fn_kwargs["onnx_opset_version"] = onnx_opset_version

    return fn_quantization(**fn_kwargs, **kwargs)

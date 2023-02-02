"""Apply quantization and hardware-specific optimizations in segmenter models."""
import typing as t
import pickle
import os
import pathlib
import collections
import platform
import warnings
import random
import datetime
import shutil

import transformers
import torch
import torch.nn
import torch.onnx

from .. import _base
from .. import segmenter
from . import _optional_import_utils
from . import models


colorama = _optional_import_utils.load_optional_module("colorama")


__all__ = [
    "quantize_bert_model_as_onnx",
    "quantize_bert_model_as_torch",
    "quantize_lstm_model_as_onnx",
    "quantize_lstm_model_as_torch",
    "quantize_model",
]


class QuantizationOutputONNX(t.NamedTuple):
    """Output paths for quantization as ONNX format."""

    onnx_base_uri: str
    onnx_quantized_uri: str
    output_uri: str
    onnx_optimized_uri: t.Optional[str] = None


class QuantizationOutputTorch(t.NamedTuple):
    """Quantization output paths as Torch format."""

    output_uri: str


QuantizationOutput = t.Union[QuantizationOutputONNX, QuantizationOutputTorch]


def _build_onnx_default_uris(
    model_name: str,
    model_attributes: t.Dict[str, t.Any],
    quantized_model_dirpath: str,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
) -> QuantizationOutputONNX:
    """Build default URIs for quantized output in ONNX format."""
    if not intermediary_onnx_model_name:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        intermediary_onnx_model_name = f"segmenter_{attrs_to_name}_{model_name}_model"

    if not quantized_model_filename:
        quantized_model_filename = f"q_{intermediary_onnx_model_name}"

    if not intermediary_onnx_model_name.endswith(".onnx"):
        intermediary_onnx_model_name += ".onnx"

    if not quantized_model_filename.endswith(".onnx"):
        quantized_model_filename += ".onnx"

    pathlib.Path(quantized_model_dirpath).mkdir(exist_ok=True, parents=True)

    onnx_base_uri = os.path.join(quantized_model_dirpath, intermediary_onnx_model_name)
    onnx_quantized_uri = os.path.join(quantized_model_dirpath, quantized_model_filename)

    paths_dict: t.Dict[str, str] = {
        "onnx_base_uri": onnx_base_uri,
        "onnx_quantized_uri": onnx_quantized_uri,
        "output_uri": onnx_quantized_uri,
    }

    paths = QuantizationOutputONNX(**paths_dict)

    all_path_set = {paths.onnx_base_uri, paths.onnx_quantized_uri}
    num_distinct_paths = len(all_path_set)

    if num_distinct_paths < 2:
        raise ValueError(
            f"{2 - num_distinct_paths} URI for ONNX models (including intermediary models) "
            "are the same, which will cause undefined behaviour while quantizing the model. "
            "Please provide distinct filenames for ONNX files."
        )

    return paths


def _build_torch_default_uris(
    model_name: str,
    model_attributes: t.Dict[str, t.Any],
    quantized_model_dirpath: str,
    quantized_model_filename: t.Optional[str] = None,
) -> QuantizationOutputTorch:
    """Build default URIs for quantized output in Torch format."""
    if not quantized_model_filename:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        quantized_model_filename = f"q_segmenter_{attrs_to_name}_{model_name}_model.pt"

    pathlib.Path(quantized_model_dirpath).mkdir(exist_ok=True, parents=True)
    output_uri = os.path.join(quantized_model_dirpath, quantized_model_filename)

    paths = QuantizationOutputTorch(output_uri=output_uri)

    return paths


def _gen_dummy_inputs_for_tracing(
    batch_size: int, vocab_size: int, seq_length: int
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate dummy inputs for Torch JIT tracing."""
    dummy_input_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_length), dtype=torch.long
    )
    dummy_attention_mask = torch.randint(
        low=0, high=2, size=(batch_size, seq_length), dtype=torch.long
    )
    dummy_token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    return dummy_input_ids, dummy_attention_mask, dummy_token_type_ids


def quantize_bert_model_as_onnx(
    model: segmenter.BERTSegmenter,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    check_cached: bool = True,
    verbose: bool = False,
) -> QuantizationOutputONNX:
    """Create a quantized BERTSegmenter model as ONNX format.

    Models created from this format can be loaded for inference as:

    >>> optimize.ONNXBERTSegmenter(  # doctest: +SKIP
    ...     uri_model='<quantized_model_uri>',
    ...     uri_tokenizer=...,
    ...     ...,
    ... )

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

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    verbose : bool, default=False
        If True, print information regarding the results.

    Returns
    -------
    paths : t.Tuple[str, ...]
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

    model_config: transformers.BertConfig = model.model.config  # type: ignore
    is_pruned = bool(model_config.pruned_heads)

    if is_pruned:
        raise RuntimeError(
            "BERT with pruned attention heads will not work in ONNX format. Please use Torch "
            "format instead (with quantize_bert_model_as_torch(...) function or by using "
            "quantize_model(..., model_output_format='torch_jit')."
        )

    model_attributes: t.Dict[str, t.Any] = collections.OrderedDict(
        (
            ("num_layers", model_config.num_hidden_layers),
            ("vocab_size", model.tokenizer.vocab_size),
            ("pruned", is_pruned),
        )
    )

    paths = _build_onnx_default_uris(
        model_name="bert",
        model_attributes=model_attributes,
        quantized_model_dirpath=quantized_model_dirpath,
        quantized_model_filename=quantized_model_filename,
        intermediary_onnx_model_name=intermediary_onnx_model_name,
    )

    quantized_model_uri = paths.output_uri.replace(".onnx", "_onnx")

    paths = QuantizationOutputONNX(
        onnx_base_uri=quantized_model_uri,
        onnx_quantized_uri=quantized_model_uri,
        output_uri=quantized_model_uri,
    )

    if check_cached and os.path.exists(paths.onnx_quantized_uri):
        if verbose:  # pragma: no cover
            print(
                f"Found cached model in '{paths.onnx_quantized_uri}'.",
                "Skipping model quantization.",
            )

        return paths

    optimization_config = optimum_onnxruntime.OptimizationConfig(
        optimization_level=99,
        enable_transformers_specific_optimizations=True,
        disable_gelu_fusion=False,
        disable_embed_layer_norm_fusion=False,
        disable_attention_fusion=False,
        disable_skip_layer_norm_fusion=False,
        disable_bias_skip_layer_norm_fusion=False,
        disable_bias_gelu_fusion=False,
        enable_gelu_approximation=True,
        optimize_for_gpu=torch.device(model.model.device).type == "cuda",  # type: ignore
    )

    quantization_config = optimum_onnxruntime.configuration.QuantizationConfig(
        is_static=False,
        format=optimum_onnxruntime.quantization.QuantFormat.QOperator,
        mode=optimum_onnxruntime.quantization.QuantizationMode.IntegerOps,
        activations_dtype=optimum_onnxruntime.quantization.QuantType.QUInt8,
        weights_dtype=optimum_onnxruntime.quantization.QuantType.QInt8,
        per_channel=True,
        operators_to_quantize=["MatMul", "Add", "Gather", "EmbedLayerNormalization", "Attention"],
    )

    ort_model = optimum_onnxruntime.ORTModelForTokenClassification.from_pretrained(
        model.model.name_or_path,
        from_transformers=True,
        local_files_only=True,
    )

    optimizer = optimum_onnxruntime.ORTOptimizer.from_pretrained(ort_model)

    temp_optimized_model_uri = "_".join(
        [
            "temp_optimized_ulysses_segmenter_model",
            datetime.datetime.utcnow().strftime("%Y_%m_%d__%H_%M_%S"),
            hex(random.getrandbits(128))[2:],
        ]
    )
    temp_optimized_model_uri = os.path.join(quantized_model_dirpath, temp_optimized_model_uri)

    optimizer.optimize(
        save_dir=temp_optimized_model_uri,
        file_suffix="",
        optimization_config=optimization_config,
    )

    try:
        ort_model = optimum_onnxruntime.ORTModelForTokenClassification.from_pretrained(
            temp_optimized_model_uri,
            local_files_only=True,
        )

        quantizer = optimum_onnxruntime.ORTQuantizer.from_pretrained(ort_model)

        quantizer.quantize(
            save_dir=paths.onnx_quantized_uri,
            file_suffix="quantized",
            quantization_config=quantization_config,
        )

    finally:
        shutil.rmtree(temp_optimized_model_uri)

    if verbose:  # pragma: no cover
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        module_name = ".".join(__name__.split(".")[:-1])

        print(
            f"Saved quantized BERT (ONNX format) in {c_blu}'{paths.onnx_quantized_uri}'{c_rst}. "
            "To use it, load a BERT segmenter model as:\n\n"
            f"{module_name}.{models.ONNXBERTSegmenter.__name__}(\n"
            f"   {c_ylw}uri_model={c_blu}'{paths.onnx_quantized_uri}'{c_rst},\n"
            f"   uri_tokenizer='{model.tokenizer.name_or_path}',\n"
            "   ...,\n"
            ")"
        )

    return paths


def quantize_lstm_model_as_onnx(
    model: segmenter.LSTMSegmenter,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    onnx_opset_version: int = 17,
    check_cached: bool = True,
    verbose: bool = False,
) -> QuantizationOutputONNX:
    """Create a quantized LSTMSegmenter model as ONNX format.

    Models created from this format can be loaded for inference as:

    >>> optimize.ONNXLSTMSegmenter(  # doctest: +SKIP
    ...     uri_model='<quantized_model_uri>',
    ...     uri_tokenizer=...,
    ...     ...,
    ... )

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

    onnx_opset_version: int, default=17
        ONNX operator set version. Used only if `model_output_format='onnx'`. Check [2]_ for
        more information.

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    verbose : bool, default=False
        If True, print information regarding the results.

    Returns
    -------
    paths : t.Tuple[str, ...]
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

    model_attributes: t.Dict[str, t.Any] = collections.OrderedDict(
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
            do_constant_folding=True,
            dynamic_axes={
                "input_ids": {0: "batch_axis", 1: "sentence_length"},
                "logits": {0: "batch_axis", 1: "sentence_length"},
            },
        )

    elif verbose:  # pragma: no cover
        print(f"Found cached ONNX model in '{paths.onnx_base_uri}'.")

    onnxruntime.quantization.quantize_dynamic(
        model_input=paths.onnx_base_uri,
        model_output=paths.onnx_quantized_uri,
        weight_type=onnxruntime.quantization.QuantType.QUInt8,
        optimize_model=True,
        per_channel=True,
    )

    if verbose:  # pragma: no cover
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        module_name = ".".join(__name__.split(".")[:-1])

        print(
            f"Saved quantized Pytorch module (ONNX format) in {c_blu}'{paths.output_uri}'{c_rst}. "
            "To use it, load a LSTM segmenter model as:\n\n"
            f"{module_name}.{models.ONNXLSTMSegmenter.__name__}(\n"
            f"   {c_ylw}uri_model={c_blu}'{paths.output_uri}'{c_rst},\n"
            f"   uri_tokenizer='{model.tokenizer.name_or_path}',\n"
            "   ...,\n"
            ")"
        )

    return paths


def quantize_bert_model_as_torch(
    model: segmenter.BERTSegmenter,
    quantized_model_filename: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    modules_to_quantize: t.Union[
        t.Set[t.Type[torch.nn.Module]], t.Tuple[t.Type[torch.nn.Module], ...]
    ] = (
        torch.nn.Embedding,
        torch.nn.Linear,
    ),
    check_cached: bool = True,
    verbose: bool = False,
) -> QuantizationOutputTorch:
    """Create a quantized BERTSegmenter model as Torch format.

    Models created from this format can be loaded for inference as:

    >>> optimize.TorchJITBERTSegmenter(  # doctest: +SKIP
    ...     uri_model='<quantized_model_uri>',
    ...     ...,
    ... )

    Parameters
    ----------
    model : segmenter.BERTSegmenter
        BERTSegmenter model to be quantized.

    quantized_model_filename : t.Optional[str], default=None
        Output filename. If None, a long and descriptive name will be derived from model's
        parameters.

    quantized_model_dirpath : str, default='./quantized_models'
        Path to output file directory, which the resulting quantized model will be stored,
        alongside any possible coproducts also generated during the quantization procedure.

    modules_to_quantize : t.Tuple[t.Type[torch.nn.Module], ...], \
        default=(torch.nn.Embedding, torch.nn.Linear)

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    verbose : bool, default=False
        If True, print information regarding the results.

    Returns
    -------
    paths : t.Tuple[str, ...]
        File URIs related from generated files during the quantization procedure. The
        final model URI can be accessed from the `output_uri` attribute.
    """
    model_config: transformers.BertConfig = model.model.config  # type: ignore

    model_attributes: t.Dict[str, t.Any] = collections.OrderedDict(
        (
            ("num_layers", model_config.num_hidden_layers),
            ("vocab_size", model.tokenizer.vocab_size),
            ("pruned", bool(model_config.pruned_heads)),
        )
    )

    paths = _build_torch_default_uris(
        model_name="bert",
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
        embedding_modules = [
            pytorch_module.get_submodule("bert.embeddings.word_embeddings"),
            pytorch_module.get_submodule("bert.embeddings.position_embeddings"),
            pytorch_module.get_submodule("bert.embeddings.token_type_embeddings"),
        ]

        for module in embedding_modules:
            module.qconfig = torch.quantization.float_qparams_weight_only_qconfig  # type: ignore

    quantized_pytorch_module = torch.quantization.quantize_dynamic(
        pytorch_module,
        set(modules_to_quantize),
        dtype=torch.qint8,
    )

    dummy_inputs = _gen_dummy_inputs_for_tracing(
        batch_size=1,
        vocab_size=model_config.vocab_size,
        seq_length=model_config.max_position_embeddings,
    )

    jit_traced_model = torch.jit.trace(quantized_pytorch_module, dummy_inputs, strict=False)
    pickled_tokenizer = pickle.dumps(model.tokenizer, protocol=pickle.HIGHEST_PROTOCOL)

    torch.jit.save(
        m=jit_traced_model,
        f=paths.output_uri,
        _extra_files={"tokenizer": pickled_tokenizer},
    )

    if verbose:  # pragma: no cover
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        module_name = ".".join(__name__.split(".")[:-1])

        print(
            "Saved quantized Pytorch module (Torch JIT format) in "
            f"{c_blu}'{paths.output_uri}'{c_rst}. "
            "To use it, load a BERT segmenter model as:\n\n"
            f"{module_name}.{models.TorchJITBERTSegmenter.__name__}(\n"
            f"   {c_ylw}uri_model={c_blu}'{paths.output_uri}'{c_rst},\n"
            "   ...,\n"
            ")"
        )

    return paths


def quantize_lstm_model_as_torch(
    model: segmenter.LSTMSegmenter,
    quantized_model_filename: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    modules_to_quantize: t.Union[
        t.Set[t.Type[torch.nn.Module]], t.Tuple[t.Type[torch.nn.Module], ...]
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

    >>> optimize.TorchJITLSTMSegmenter(  # doctest: +SKIP
    ...     uri_model='<quantized_model_uri>',
    ...     ...,
    ... )

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

    modules_to_quantize : t.Tuple[t.Type[torch.nn.Module], ...], \
        default=(torch.nn.Embedding, torch.nn.LSTM, torch.nn.Linear)

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    verbose : bool, default=False
        If True, print information regarding the results.

    Returns
    -------
    paths : t.Tuple[str, ...]
        File URIs related from generated files during the quantization procedure. The
        final model URI can be accessed from the `output_uri` attribute.
    """

    model_attributes: t.Dict[str, t.Any] = collections.OrderedDict(
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

    dummy_input_ids, _, _ = _gen_dummy_inputs_for_tracing(
        batch_size=3,
        vocab_size=model.tokenizer.vocab_size,
        seq_length=512,
    )

    dummy_input_ids = dummy_input_ids.unsqueeze(1)

    jit_traced_model = torch.jit.trace(
        func=torch.jit.script(quantized_pytorch_module),
        example_inputs=(dummy_input_ids[0],),
        check_inputs=[(dummy_input_ids[1],), (dummy_input_ids[2],)],
        strict=True,
    )
    pickled_tokenizer = pickle.dumps(model.tokenizer, protocol=pickle.HIGHEST_PROTOCOL)

    torch.jit.save(
        m=jit_traced_model,
        f=paths.output_uri,
        _extra_files={"tokenizer": pickled_tokenizer},
    )

    if verbose:  # pragma: no cover
        c_ylw = colorama.Fore.YELLOW if colorama else ""
        c_blu = colorama.Fore.BLUE if colorama else ""
        c_rst = colorama.Style.RESET_ALL if colorama else ""

        module_name = ".".join(__name__.split(".")[:-1])

        print(
            "Saved quantized Pytorch module (Torch JIT format) in "
            f"{c_blu}'{paths.output_uri}'{c_rst}. "
            "To use it, load a LSTM segmenter model as:\n\n"
            f"{module_name}.{models.TorchJITLSTMSegmenter.__name__}(\n"
            f"   {c_ylw}uri_model={c_blu}'{paths.output_uri}'{c_rst},\n"
            "   ...,\n"
            ")"
        )

    return paths


def quantize_model(
    model: t.Union[segmenter.BERTSegmenter, segmenter.LSTMSegmenter],
    quantized_model_filename: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    model_output_format: str = "onnx",
    onnx_opset_version: int = 17,
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

    model_output_format : {'onnx', 'torch_jit'}, default='onnx'
        Output format of quantized model. This option also determines how exactly inference with
        the quantized model will be done. See ``See Also`` section for information about specific
        configuratins of model types and output formats.

    onnx_opset_version: int, default=17
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
    paths : t.Tuple[str, ...]
        Named tuple with all paths of files generated during the full quantization
        procedure.

    See Also
    --------
    quantize_lstm_model_as_onnx : quantize LSTMSegmenter as model_output_format='onnx'.
    quantize_lstm_model_as_torch : quantize LSTMSegmenter as model_output_format='torch_jit'.
    quantize_bert_model_as_onnx : quantize BERTSegmenter as model_output_format='onnx'.
    quantize_bert_model_as_torch : quantize BERTSegmenter as model_output_format='torch_jit'.

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

    if model_output_format not in {"onnx", "torch_jit"}:
        raise ValueError(
            f"Unsupported 'model_output_format={model_output_format}'. "
            "Please choose either 'onnx' or 'torch_jit'."
        )

    fn_kwargs: t.Dict[str, t.Any] = {
        "model": model,
        "quantized_model_filename": quantized_model_filename,
        "quantized_model_dirpath": quantized_model_dirpath,
        "check_cached": check_cached,
        "verbose": verbose,
    }

    fn_quantization_factory: t.Dict[
        t.Tuple[t.Type[_base.BaseSegmenter], str], t.Callable[..., QuantizationOutput]
    ] = {
        (segmenter.BERTSegmenter, "torch_jit"): quantize_bert_model_as_torch,
        (segmenter.BERTSegmenter, "onnx"): quantize_bert_model_as_onnx,
        (segmenter.LSTMSegmenter, "torch_jit"): quantize_lstm_model_as_torch,
        (segmenter.LSTMSegmenter, "onnx"): quantize_lstm_model_as_onnx,
    }

    try:
        fn_quantization = fn_quantization_factory[(type(model), model_output_format)]

    except KeyError as e_key:
        raise ValueError(
            f"Unsupported 'model_output_format={model_output_format}' for segmenter "
            f"type={type(model)}."
        ) from e_key

    if model_output_format == "onnx" and isinstance(model, segmenter.LSTMSegmenter):
        v_major, v_minor, _ = platform.python_version_tuple()
        fn_kwargs["onnx_opset_version"] = onnx_opset_version

        if 100 * int(v_major) + int(v_minor) < 310 and onnx_opset_version > 15:
            warnings.warn(
                f"Unsupported onnx_opset_version={onnx_opset_version} for Python version < 3.10 "
                f"(detected '{v_major}.{v_minor}'). Setting it to '15'.",
                UserWarning,
            )
            fn_kwargs["onnx_opset_version"] = 15

    return fn_quantization(**fn_kwargs, **kwargs)

"""Handle optional imports that are required only in certain situations."""
import typing as t
import importlib
import warnings
import types


def load_optional_module(module_name: str) -> t.Optional[types.ModuleType]:
    """Load an optional module, preventing any ImportErrors."""
    try:
        return MAP_FACTORY[module_name]()

    except ImportError:
        return None


def load_required_module(module_name: str) -> types.ModuleType:
    """Load the requested module."""
    return MAP_FACTORY_REQUIRED[module_name]()


def import_optimum_onnxruntime() -> types.ModuleType:
    """Load optimum.onnxruntime (from optimum[onnxruntime])."""
    try:
        return importlib.import_module("optimum.onnxruntime")

    except ImportError as e_import:
        raise ImportError(
            "Optinal dependency 'optimum.onnxruntime' not found, which is necessary to "
            "produce ONNX BERT segmenter models. Please install it with the following "
            "command:\n\n"
            "python -m pip install optimum[onnxruntime]\n\n"
            "See https://huggingface.co/docs/optimum/index for more information."
        ) from e_import


def import_onnxruntime() -> types.ModuleType:
    """Load onnxruntime."""
    try:
        return importlib.import_module("onnxruntime")

    except ImportError as e_import:
        raise ImportError(
            "Optinal dependency 'onnxruntime' not found, which is necessary to "
            "produce ONNX LSTM segmenter models. Please install it with the following "
            "command:\n\n"
            "python -m pip install onnxruntime\n\n"
            "See https://onnxruntime.ai/ for more information."
        ) from e_import


def import_colorama() -> t.Optional[types.ModuleType]:
    """Try to load colorama."""
    try:
        return importlib.import_module("colorama")

    except ImportError:
        warnings.warn(
            message=(
                "Optional dependency 'colorama' not found. The quantization output will be "
                "colorless. In order to (optionally) fix this issue, use the following command:"
                "\n\n"
                "python -m pip install colorama\n\n"
                "See https://pypi.org/project/colorama/ for more information."
            ),
            category=ImportWarning,
        )
        return None


MAP_FACTORY_REQUIRED: t.Dict[str, t.Callable[[], types.ModuleType]] = {
    "optimum.onnxruntime": import_optimum_onnxruntime,
    "onnxruntime": import_onnxruntime,
}

MAP_FACTORY: t.Dict[str, t.Callable[[], t.Optional[types.ModuleType]]] = {
    "colorama": import_colorama,
    **MAP_FACTORY_REQUIRED,
}

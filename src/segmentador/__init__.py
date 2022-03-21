# pylint: disable='missing-module-docstring'
from .segmenter import *

try:
    import importlib.metadata as importlib_metadata

except ModuleNotFoundError:
    import importlib_metadata  # type: ignore


try:
    __version__ = importlib_metadata.version(__name__)

except importlib_metadata.PackageNotFoundError:
    __version__ = "0.1.0"

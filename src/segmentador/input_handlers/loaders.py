"""TODO."""
import os

import buscador


__all__ = [
    "download_model",
    "get_model_uri_if_local_file",
]


def download_model(model_name: str, output_dir: str, show_progress_bar: bool = True) -> bool:
    """TODO."""
    try:
        download_has_succeed = buscador.download_model(
            task_name="legal_text_segmentation",
            model_name=model_name,
            output_dir=output_dir,
            show_progress_bar=show_progress_bar,
            check_cached=True,
            clean_zip_files=True,
            check_model_hash=True,
            timeout_limit_seconds=10,
        )
        return bool(download_has_succeed)

    except ValueError:
        return False


def get_model_uri_if_local_file(
    model_name: str, download_dir: str, file_extension: str = ""
) -> str:
    """TODO"""
    uri_model = str(model_name)
    uri_model = os.path.join(download_dir, uri_model)
    uri_model = os.path.normpath(uri_model)
    uri_model = os.path.join(".", uri_model)

    if file_extension and not file_extension.startswith("."):
        file_extension = f".{file_extension}"

    if file_extension and not uri_model.endswith(file_extension):
        uri_model += file_extension

    if not os.path.exists(uri_model):
        return model_name

    return uri_model

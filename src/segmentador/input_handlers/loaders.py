"""TODO."""
import os

import buscador


__all__ = [
    "download_model",
]


def download_model(model_name: str, output_dir: str, show_progress_bar: bool = True) -> str:
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

    except ValueError:
        download_has_succeed = False

    uri_model = model_name

    if download_has_succeed:
        uri_model = os.path.join(output_dir, uri_model)
        uri_model = os.path.normpath(uri_model)
        uri_model = os.path.join(".", uri_model)

    return uri_model

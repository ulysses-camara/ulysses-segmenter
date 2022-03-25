"""Jupyter Notebook API to the interactive label refinery.

Send data between Jupyter Notebook and front-end interface.
"""
import os
import typing as t
import json as json_package
import collections
import warnings

import numpy as np
import numpy.typing as npt
import requests


FLASK_PORT = os.environ.get("FLASK_PORT", 5000)
DATA_WAS_SENT = False


# pylint: disable=global-statement


def _compute_margin(
    logits: npt.NDArray[np.float64],
    apply_softmax_to_logits: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute token-wise class margins (largest activation minus second largest activation)."""

    def softmax(vals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        vals = np.array(vals, dtype=np.float64)
        vals -= np.max(vals, axis=-1, keepdims=True)
        vals -= np.log(np.sum(np.exp(vals), axis=-1, keepdims=True))
        return np.exp(vals)

    activations = softmax(logits) if apply_softmax_to_logits else logits
    rank_1, rank_2 = np.quantile(activations, q=(1.0, 0.99), method="lower", axis=-1)

    margins = np.asfarray(rank_1 - rank_2).squeeze()
    margins = np.atleast_1d(margins)

    return margins


def open_example(
    tokens: list[str],
    labels: list[t.Union[int, str]],
    logits: t.Optional[npt.NDArray[np.float64]] = None,
    apply_softmax_to_logits: bool = True,
) -> None:
    """Send example from Jupyter Notebook to interactive front-end.

    Parameters
    ----------
    tokens : list of str
        Subword input text tokens.

    labels : list of t.Union[int, str]
        Token-wise labels.

    logits : npt.NDArray[np.float64] or None, default=None
        Token-wise logits from a pivot model. Used to compute class margins, which
        in turn are translated as a token-wise heatmap in front-end, highlighting
        tokens that the pivot model is unsure about its classification.

    apply_softmax_to_logits : bool, default=True
        If False, assume that `logits` are class activations (softmaxed logits) instead
        of proper logits.

    Raises
    ------
    ConnectionError
        If connection with flask application was not successful.
    """
    data = [{"token": tok, "label": lab} for tok, lab in zip(tokens, labels)]

    if logits is not None:
        margins = _compute_margin(logits, apply_softmax_to_logits=apply_softmax_to_logits)
        for item, margin in zip(data, margins):
            item["margin"] = margin

    rep = requests.post(
        os.path.join(f"http://localhost:{FLASK_PORT}/", "refinery-data-transfer"),
        json=data,
    )

    if rep.status_code != 200:
        raise ConnectionError(
            "Something went wrong while connecting to front-end "
            f"(status code: {rep.status_code})"
        )

    global DATA_WAS_SENT
    DATA_WAS_SENT = True

    rep = requests.post(
        os.path.join(f"http://localhost:{FLASK_PORT}/", "call-for-refresh"),
        json=data,
    )

    if rep.status_code != 200:
        warnings.warn(
            message=(
                "Data was sent to front-end, but auto-refresh could not be triggered. "
                "You need to reload manually."
            ),
            category=UserWarning,
        )


def retrieve_refined_example(
    return_modified_list: bool = True,
) -> dict[str, list[t.Union[bool, str, int]]]:
    """Recover refined data from front-end to Jupyter Notebook.

    This function must be invoked after `open_example`.

    Parameters
    ----------
    return_modified_list : bool, default=True
        If True, include the `modified` boolean value in return dictionary,
        indicating which labels where effectively modified.

    Returns
    -------
    refined_example : dict of list
        Dictionary containing refined example data. Will have the keys `tokens` and `labels`
        and, optionally, the key `modified` if `return_modified_list=True`.

    Raises
    ------
    ValueError
        If this function is called before `open_example`.

    ConnectionError
        If connection with flask application was not successful.
    """
    if not DATA_WAS_SENT:
        raise ValueError(
            "No data was not sent to front-end. Please use 'open_example' "
            "function before retrieving results."
        )

    rep = requests.get(f"http://localhost:{FLASK_PORT}/refinery-data-transfer")

    if rep.status_code != 200:
        raise ConnectionError(
            "Something went wrong while connecting to front-end "
            f"(status code: {rep.status_code})"
        )

    json = json_package.loads(rep.text)

    ret = collections.defaultdict(list)

    for item in json:
        ret["tokens"].append(item["token"])
        ret["labels"].append(int(item["label"]))

        if return_modified_list:
            ret["modified"].append(item.get("modified", False))

    return ret

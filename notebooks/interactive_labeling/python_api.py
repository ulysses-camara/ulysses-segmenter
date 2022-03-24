import os
import typing as t
import json as json_package
import collections

import numpy as np
import numpy.typing as npt
import requests


FLASK_PORT = 5000
DATA_WAS_SENT = False


def _compute_margin(
    logits: npt.NDArray[np.float64],
    apply_softmax_to_logits: bool = True,
) -> npt.NDArray[np.float64]:
    def softmax(x):
        x = np.array(x, dtype=np.float64)
        x -= np.max(x, axis=-1, keepdims=True)
        x -= np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
        return np.exp(x)

    activations = softmax(logits) if apply_softmax_to_logits else logits
    rank_1, rank_2 = np.quantile(activations, q=(1.0, 0.75), interpolation="nearest", axis=-1)
    return (rank_1 - rank_2).squeeze()


def open_example(
    tokens: list[str],
    labels: list[t.Union[int, str]],
    logits: t.Optional[npt.NDArray[np.float64]] = None,
    apply_softmax_to_logits: bool = True,
) -> None:
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


def retrieve_refined_example(return_modified_list: bool = True) -> dict[str, list]:
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

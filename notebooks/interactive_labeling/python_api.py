import typing as t
import json as json_package
import collections

import requests

FLASK_PORT = 5000
DATA_WAS_SENT = False


def open_example(tokens: list[str], labels: list[t.Union[int, str]]) -> None:
    data = [{"token": tok, "label": lab} for tok, lab in zip(tokens, labels)]

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

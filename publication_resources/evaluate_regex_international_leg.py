import itertools

import regex as re
import pandas as pd
import tqdm
import sklearn.metrics
import datasets
import numpy as np

import weak_supervision_data_preparation


re_quotes = re.compile(r"(?:[““]|`\s*`|'\s*')+")


CLS_TOKEN_ID = 2
SEG_START_CLS_ID = 1


def run(df, dt, language: str):
    segs_true: list[str] = []
    segs_regex_preds: list[str] = []

    docs_true = []
    cur_inst = []

    # NOTE: if True, then dataset has only a single document.
    include_cls_token = bool(int(dt["input_ids"][0][0]) != CLS_TOKEN_ID)

    for inst in dt["input_ids"]:
        if inst[0] == CLS_TOKEN_ID:
            docs_true.append(weak_supervision_data_preparation.seg_model.tokenizer.convert_ids_to_tokens(cur_inst))
            cur_inst = []
        cur_inst.extend(inst)

    if docs_true:
        docs_true.pop(0)
        docs = " ".join(df).replace("[SEP]", " ").strip().split("[CLS]")[1:]
    else:
        docs = [" ".join(df)]

    if cur_inst:
        docs_true.append(weak_supervision_data_preparation.seg_model.tokenizer.convert_ids_to_tokens(cur_inst))

    assert len(docs_true) == len(docs), (len(docs_true), len(docs))

    for i, text in enumerate(tqdm.tqdm(docs)):
        text = text.strip()
        text = text.replace("cannot", "_cannot_")  # NOTE: prevent splitting "cannot" into "can" + "not"
        text = weak_supervision_data_preparation.preprocess_instance({"text": text}, ind=0, language=language)
        text["tokens"] = [re_quotes.sub('"', item) for item in text["tokens"]]
        text["tokens"] = [item if item != "_cannot_" else "cannot" for item in text["tokens"]]
        text = weak_supervision_data_preparation.tokenize_and_align_labels(
            {k: [v] for k, v in text.items()}, max_tokens_per_inst=90000000000, truncation=False
        )
        segs_regex_preds.extend(text["labels"][0])

    segs_true = list(itertools.chain(*dt["labels"]))

    if include_cls_token:
        segs_true.insert(0, -100)
        segs_true.append(-100)

    assert len(segs_true) == len(segs_regex_preds), (len(segs_true), len(segs_regex_preds))

    segs_regex_preds = [int(item) if item in {0, 1} else 0 for i, item in enumerate(segs_regex_preds) if segs_true[i] != -100]
    segs_true = [int(item) for item in segs_true if item != -100]

    precision = sklearn.metrics.precision_score(segs_true, segs_regex_preds, average="binary")
    recall = sklearn.metrics.recall_score(segs_true, segs_regex_preds, average="binary")

    print("Precision (c_1, %)", (100.0 * precision).round(2))
    print("Recall    (c_1, %)", (100.0 * recall).round(2))
    print("F1-score  (c_1, %)", (100.0 * 2 * precision * recall / (1e-12 + precision + recall)).round(2))
    print("Gavg      (c_1, %)", (100.0 * np.sqrt(precision * recall)).round(2))


def load_data(dataset_uri: str):
    dt = datasets.Dataset.load_from_disk(dataset_uri.removesuffix(".tsv"))
    df = pd.read_csv(dataset_uri, sep="\t", index_col=0).squeeze().tolist()
    return (dt, df)


def run_us():
    # NOTE: Data available in:
    # https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
    dt, df = load_data("../data/scraping_other_langs/publication_data/us_congressional_bills.tsv")
    print("US:")
    run(df, dt, language="english")
    print(end="\n\n")


def run_italian():
    # NOTE: Data available in:
    # https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
    dt, df = load_data("../data/scraping_other_langs/publication_data/italian_legislative_decrees.tsv")
    print("ITALIAN:")
    run(df, dt, language="italian")
    print(end="\n\n")


def run_german():
    # NOTE: Data available in:
    # https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
    dt, df = load_data("../data/scraping_other_langs/publication_data/german_laws_and_regulations.tsv")
    print("GERMAN:")
    run(df, dt, language="german")
    print(end="\n\n")


def run_french():
    # NOTE: Data available in:
    # https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
    dt, df = load_data("../data/scraping_other_langs/publication_data/french_code_civil.tsv")
    print("FRENCH:")
    run(df, dt, language="french")
    print(end="\n\n")


if __name__ == "__main__":
    # NOTE: Data available in:
    # https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
    # run_us()
    # run_italian()
    # run_german()
    run_french()

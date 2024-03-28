import functools
import itertools
import os

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import datasets
import torch
import torch.nn
import torch.utils
import segmentador
import tqdm
import sklearn
import sklearn.metrics
import numpy as np
import pandas as pd


def fn_pad_and_truncate(X, max_length: int = 1024):
    for k, v in X.items():
        X[k] = v[:max_length]

    rem = max_length - len(X["labels"])
    right = (rem + 1) // 2
    left = rem // 2

    assert right + left == rem

    X["labels"] = left * [-100] + X["labels"] + right * [-100]
    X["input_ids"] = left * [0] + X["input_ids"] + right * [0]
    X["token_type_ids"] = left * [0] + X["token_type_ids"] + right * [0]
    X["attention_mask"] = left * [0] + X["attention_mask"] + right * [0]

    return X


def fn_compute_metrics(labels, preds):
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)

    assert labels.ndim == 1
    assert preds.ndim == 1
    assert labels.size == preds.size, (labels.size, preds.size)

    preds = [v for i, v in enumerate(preds) if labels[i] != -100]
    labels = [v for v in labels if v != -100]

    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, preds, average="macro")
    cls_precision, cls_recall, cls_f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, preds, average=None)
    acc = sklearn.metrics.accuracy_score(labels, preds)

    def add_cls_ind(vals, key):
        return {f"cls_{i}_{key}": val for i, val in enumerate(vals)}

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        **add_cls_ind(cls_precision, "precision"),
        **add_cls_ind(cls_recall, "recall"),
        **add_cls_ind(cls_f1, "f1"),
    }


class MaskedBertSelfAttention(torch.nn.Module):
    def __init__(self, attention_module, attention_mask: torch.Tensor):
        super().__init__()
        self.attention_module = attention_module
        self.attention_mask = attention_mask

    def __call__(self, *args, **kwargs):
        (
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        ) = args

        encoder_hidden_states = hidden_states
        encoder_attention_mask = self.attention_mask.to("cuda:0")

        return self.attention_module(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )


def inject_attention_mask_(model, size: int) -> None:
    n = 1024
    (left, right) = size

    attention_mask = torch.zeros(n, n)
    attention_mask[*torch.triu_indices(n, n, offset=right)] = -torch.inf
    attention_mask[*torch.tril_indices(n, n, offset=-left)] = -torch.inf
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    for i in range(len(model.bert.encoder.layer)):
        model.bert.encoder.layer[i].attention.self = MaskedBertSelfAttention(
            model.bert.encoder.layer[i].attention.self,
            attention_mask=attention_mask,
        )


def eval_model(dt, size: tuple[int, int], batch_size: int = 32):
    # NOTE: Models info + download link:
    # https://github.com/ulysses-camara/ulysses-segmenter/tree/master?tab=readme-ov-file#available-models
    model = segmentador.BERTSegmenter(
        uri_model="4_6000_layer_model_v2",
        device="cuda:0",
    )

    model = model.model

    inject_attention_mask_(model=model, size=size)

    y_preds = []
    y_true = []

    dt = dt.map(fn_pad_and_truncate)
    dt.set_format("torch")
    dl = torch.utils.data.DataLoader(dt, batch_size=batch_size)

    for batch in tqdm.tqdm(dl):
        labels = batch.pop("labels")

        batch = {k: v.to("cuda:0") for k, v in batch.items()}
        cur_logits = model(**batch)["logits"]

        cur_y_preds = cur_logits.detach().argmax(axis=-1).cpu()
        cur_y_true = labels.cpu()

        y_preds.extend(cur_y_preds.view(-1).tolist())
        y_true.extend(cur_y_true.view(-1).tolist())

    assert len(y_preds) == len(y_true)

    res = fn_compute_metrics(y_true, y_preds)
    return res


def test():
    # NOTE: Data info + download link:
    # https://github.com/ulysses-camara/ulysses-segmenter/tree/master?tab=readme-ov-file#train-and-evaluation-data
    dt = datasets.DatasetDict.load_from_disk("data/dataset_ulysses_segmenter_v2_active_learning_curated_only")
    dt = dt["test"]
    print(dt)
    context_sizes = [1, 16, 32, 64, 128, 256, 512, 1024][::-1]

    (scores_c1, scores_macro) = np.zeros((2, len(context_sizes), len(context_sizes)), dtype=float)

    for i, j in itertools.product(range(len(context_sizes)), range(len(context_sizes))):
        print("bert", (i, j))
        with torch.no_grad():
            res = eval_model(dt, (context_sizes[i], context_sizes[j]), batch_size=8)

        scores_c1[i, j] = float(res["cls_1_f1"])
        scores_macro[i, j] = float(res["macro_f1"])

        print(scores_c1)
        print(scores_macro)

    print(scores_c1)
    print(scores_macro)

    df_c1 = pd.DataFrame(scores_c1, index=context_sizes, columns=context_sizes)
    df_c1.to_csv("results/shrunken_windows/scores_c1.csv")

    df_macro = pd.DataFrame(scores_macro, index=context_sizes, columns=context_sizes)
    df_macro.to_csv("results/shrunken_windows/scores_macro.csv")


def run():
    if not os.path.exists("results/shrunken_windows/scores_c1.csv"):
        test()


if __name__ == "__main__":
    run()

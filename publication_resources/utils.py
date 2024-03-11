import functools
import warnings

import torch
import torch.nn
import transformers
import datasets
import sklearn.metrics
import numpy as np
import segmentador


def fn_compute_metrics(labels, logits):
    assert labels.size
    assert logits.size

    preds = logits.argmax(-1).astype(int, copy=False)

    assert labels.ndim == 1
    assert preds.ndim == 1
    assert labels.size == preds.size, (labels.size, preds.size)

    preds = [v for i, v in enumerate(preds) if labels[i] != -100]
    labels = [v for v in labels if v != -100]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def segments_to_dict(segments: list[str]) -> dict[str, list[list[int]]]:
    tokenizer = transformers.AutoTokenizer.from_pretrained("../../cache/tokenizers/6000_subword_tokenizer/")
    output = tokenizer(segments, add_special_tokens=False, truncation=False)
    labels = []

    for seg in segments:
        tokens = tokenizer.tokenize(seg, add_special_tokens=False, truncation=False)
        labels.append([-100 if tok.startswith("##") else 0 for tok in tokens])
        labels[-1][0] = 1

    output["labels"] = labels

    return output


def flatten_dict(d: dict[str, list[list[int]]], *, group_ids=None) -> dict[str, list[int]]:
    new_dict = {}
    if group_ids is not None:
        group_ids_flatten = []

    for k, v in d.items():
        new_dict[k] = []
        for i, vi in enumerate(v):
            new_dict[k].extend(vi)
            if group_ids is not None and k == "input_ids":
                group_ids_flatten.extend(len(vi) * [group_ids[i]])

    if group_ids is not None:
        assert len(new_dict["input_ids"]) == len(group_ids_flatten)
        return new_dict, np.asarray(group_ids_flatten)

    return new_dict


def split_train_test(input_, m: int, random_state: int, shifts: int):
    n = len(input_["input_ids"])
    rng = np.random.RandomState(random_state)

    train_start_inds = set(rng.choice(n, size=m, replace=False))
    train_inds = set()
    split_train = {k: [] for k in input_.keys()}

    for tsi in train_start_inds:
        for k in split_train.keys():
            split_train[k].append([])

        for shift in range(shifts):
            i = tsi + shift

            if len(split_train["input_ids"][-1]) >= 1024:
                break
            if i >= n:
                break

            train_inds.add(i)

            for k, v in input_.items():
                vi = v[i]
                split_train[k][-1].extend(vi if isinstance(vi, list) else vi.tolist())

    split_test = {k: [vi for i, vi in enumerate(v) if i not in train_inds] for k, v in input_.items()}

    return (split_train, split_test)


def fn_pad_and_truncate(X, max_length: int):
    for k, v in X.items():
        X[k] = v[:max_length]

    rem = max_length - len(X["labels"])
    X["labels"] = X["labels"] + rem * [-100]
    X["input_ids"] = X["input_ids"] + rem * [0]
    X["token_type_ids"] = X["token_type_ids"] + rem * [0]
    X["attention_mask"] = X["attention_mask"] + rem * [0]

    return X


def train(
    segmenter_name: str | segmentador.BERTSegmenter,
    split_train: dict[str, list[list[int]]],
    pbar,
    *,
    random_init: bool = False,
    n_epochs: int = 15,
):
    if isinstance(segmenter_name, str):
        seg_model = segmentador.BERTSegmenter(
            uri_model=segmenter_name,
            device="cuda:0",
            init_from_pretrained_weights=not random_init,
        )
    else:
        seg_model = segmenter_name

    n = len(split_train["labels"])

    if n == 0:
        return seg_model

    model = seg_model.model

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.80)

    m = min(1024, max(len(x) for x in split_train["labels"]))

    if not isinstance(split_train, datasets.Dataset):
        split_train = datasets.Dataset.from_dict(split_train)

    split_train = split_train.map(functools.partial(fn_pad_and_truncate, max_length=m))
    split_train.set_format("torch")

    true_batch_size = 10
    grad_acc_its = true_batch_size // 2
    dl_train = torch.utils.data.DataLoader(split_train, shuffle=True, batch_size=2, drop_last=False)

    model.train()

    for _ in range(n_epochs):
        mov_loss = None
        optim.zero_grad()

        for j, batch in enumerate(dl_train, 1):
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            y_true = batch.pop("labels").view(-1)

            output = model(**batch)
            logits = output["logits"].view(-1, 4)

            assert logits.shape[0] == y_true.numel(), (logits.shape, y_true.shape)

            loss = loss_fn(logits, y_true)
            loss = loss / true_batch_size
            loss.backward()

            loss_val = float(loss.detach().item())
            mov_loss = 0.95 * mov_loss + 0.05 * loss_val if mov_loss is not None else loss_val

            if j % grad_acc_its == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad()

        pbar.set_description(f"{mov_loss = :.6f}")
        lr_scheduler.step()

    model.eval()
    seg_model._model = model

    return seg_model

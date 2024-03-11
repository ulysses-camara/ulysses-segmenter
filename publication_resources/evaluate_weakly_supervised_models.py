import datasets
import torch.utils
import segmentador
import tqdm
import sklearn
import sklearn.metrics
import numpy as np


def fn_pad_and_truncate(X, max_length: int = 1024):
    for k, v in X.items():
        X[k] = v[:max_length]

    rem = max_length - len(X["labels"])
    X["labels"] = X["labels"] + rem * [-100]
    X["input_ids"] = X["input_ids"] + rem * [0]
    X["token_type_ids"] = X["token_type_ids"] + rem * [0]
    X["attention_mask"] = X["attention_mask"] + rem * [0]

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


def eval_model(dt, model_type: str, size: int, batch_size: int = 32):
    if model_type == "lstm":
        # Models info + download link: https://github.com/ulysses-camara/ulysses-segmenter/tree/master?tab=readme-ov-file#available-models
        model = segmentador.LSTMSegmenter(
            uri_model=f"../segmenter_checkpoint_v1/{size}_6000_1_lstm/checkpoints/{size}_hidden_dim_6000_vocab_size_1_layer_lstm.pt",
            uri_tokenizer="tokenizers/6000_subwords",
            device="cuda:0",
        )

    else:
        # Models info + download link: https://github.com/ulysses-camara/ulysses-segmenter/tree/master?tab=readme-ov-file#available-models
        model = segmentador.BERTSegmenter(
            uri_model=f"../segmenter_checkpoint_v1/{size}_6000_layer_model",
            device="cuda:0",
        )

    model = model.model

    y_preds = []
    y_true = []

    dt = dt.map(fn_pad_and_truncate)
    dt.set_format("torch")
    dl = torch.utils.data.DataLoader(dt, batch_size=batch_size)

    for batch in tqdm.tqdm(dl):
        labels = batch.pop("labels")

        if model_type == "lstm":
            cur_logits = model(batch["input_ids"].to("cuda:0"))["logits"]
        else:
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            cur_logits = model(**batch)["logits"]

        cur_y_preds = cur_logits.detach().argmax(axis=-1).cpu()
        cur_y_true = labels.cpu()

        y_preds.extend(cur_y_preds.view(-1).tolist())
        y_true.extend(cur_y_true.view(-1).tolist())

    assert len(y_preds) == len(y_true)

    res = fn_compute_metrics(y_true, y_preds)
    return res


def run():
    # Data info + download link: https://github.com/ulysses-camara/ulysses-segmenter/tree/master?tab=readme-ov-file#train-and-evaluation-data
    dt = datasets.DatasetDict.load_from_disk("data/dataset_ulysses_segmenter_v1_weak_supervision")
    print(dt)

    for hidden_dim in [512, 256, 128]:
        print("lstm", hidden_dim)
        with torch.no_grad():
            res = eval_model(dt["test"], "lstm", hidden_dim)
        print(res)
        for k, v in res.items():
            print(k, v)
        print(end="\n\n")

    for n_layers in [2, 4, 6]:
        print("bert", n_layers)
        with torch.no_grad():
            res = eval_model(dt["test"], "bert", n_layers, batch_size=8)
        print(res)
        for k, v in res.items():
            print(k, v)
        print(end="\n\n")


if __name__ == "__main__":
    run()

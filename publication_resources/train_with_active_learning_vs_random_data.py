import copy
import os
import collections
import gc

import torch
import torch.nn
import tqdm.auto
import torchmetrics
import transformers
import segmentador
import pandas as pd
import datasets


output_dir = os.path.abspath("./results/training_with_random_vs_active_data_logs")
os.makedirs(output_dir, exist_ok=True)


def eval_model(model, dl_test, device="cuda:0"):
    try:
        model.eval()
    except AttributeError:
        pass

    if device and (not hasattr(model, "device") or torch.device(device) != model.device):
        try:
            model.to(device)
        except Exception:
            device = "cpu"

    fn_precision = torchmetrics.classification.Precision(task="multiclass", num_classes=4, average="macro").to(device)
    fn_recall = torchmetrics.classification.Recall(task="multiclass", num_classes=4, average="macro").to(device)
    fn_f1 = torchmetrics.classification.f_beta.F1Score(task="multiclass", num_classes=4, average="macro").to(device)

    fn_precision_per_cls = torchmetrics.classification.Precision(task="multiclass", num_classes=4, average=None).to(device)
    fn_recall_per_cls = torchmetrics.classification.Recall(task="multiclass", num_classes=4, average=None).to(device)
    fn_f1_per_cls = torchmetrics.classification.f_beta.F1Score(task="multiclass", num_classes=4, average=None).to(device)

    preds = []
    targets = []

    for batch in dl_test:
        with torch.no_grad():
            if isinstance(model, transformers.BertForTokenClassification):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits

            else:
                outputs = model(batch["input_ids"].to(device))
                logits = outputs["logits"].view(-1, 4)

        predictions = torch.argmax(logits, dim=-1)

        preds.append(predictions.to("cpu"))
        targets.append(batch["labels"].to("cpu"))

    try:
        preds = torch.vstack(preds).view(-1)

    except RuntimeError:
        preds = torch.concat(preds)

    targets = torch.vstack(targets).view(-1)

    preds = torch.tensor([p for i, p in enumerate(preds) if targets[i] != -100], dtype=torch.long).to(device)
    targets = torch.tensor([tg for tg in targets if tg != -100], dtype=torch.long).to(device)

    precision = fn_precision(preds, targets)
    recall = fn_recall(preds, targets)
    f1 = fn_f1(preds, targets)

    precision_cls_1 = float(fn_precision_per_cls(preds, targets)[1])
    recall_cls_1 = float(fn_recall_per_cls(preds, targets)[1])
    f1_cls_1 = float(fn_f1_per_cls(preds, targets)[1])

    return (
        float(recall.to("cpu")),
        float(precision.to("cpu")),
        float(f1.to("cpu")),
        recall_cls_1,
        precision_cls_1,
        f1_cls_1,
    )


def fn_pad(X):
    rem = 1024 - len(X["labels"])
    X["labels"] = X["labels"] + rem * [-100]
    X["input_ids"] = X["input_ids"] + rem * [0]
    X["token_type_ids"] = X["token_type_ids"] + rem * [0]
    X["attention_mask"] = X["attention_mask"] + rem * [0]
    return X


def filter_leakage(dt_a, dt_b, show_progress_bar: bool = True):
    tokenizer = segmentador.LSTMSegmenter(
        uri_model="./models/lstm_256_v1/checkpoints/256_hidden_dim_6000_vocab_size_1_layer_lstm.pt",
        device="cpu",
    ).tokenizer

    texts_a = tokenizer.batch_decode(dt_a["input_ids"])
    texts_b = frozenset(tokenizer.batch_decode(dt_b["input_ids"]))

    visited = set()
    keep_inds = set()
    n = len(texts_a)

    for i, text in enumerate(tqdm.auto.tqdm(texts_a[::-1], disable=not show_progress_bar)):
        if text in visited:
            continue

        visited.add(text)

        if text not in texts_b:
            keep_inds.add(n - 1 - i)

    print("non leakage  :", len(dt_a))
    print("leakage prop :", 1.0 - len(keep_inds) / len(dt_a))

    dt_a = dt_a.filter(lambda _, k: k in keep_inds, with_indices=True)

    return dt_a


def train(logit_model, dl_train, n):
    logit_model_copy = copy.deepcopy(logit_model.model)
    torch.cuda.empty_cache()

    # Setup for final version:
    optim = torch.optim.Adam(logit_model_copy.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.8)
    num_epochs = 3
    num_training_steps = num_epochs * len(dl_train)
    grad_acc_steps = 1
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    pbar = tqdm.auto.tqdm(range(num_training_steps), total=num_training_steps)

    logit_model_copy.to("cuda:0")
    mv_avg = 0.0

    for epoch in range(num_epochs):
        logit_model_copy.train()
        optim.zero_grad()

        for i, batch in enumerate(dl_train, 1):
            if isinstance(logit_model_copy, transformers.BertForTokenClassification):
                batch = {k: v.to("cuda:0") for k, v in batch.items()}
                outputs = logit_model_copy(**batch)
                loss = outputs.loss

            else:
                outputs = logit_model_copy(batch["input_ids"].to("cuda:0"))
                logits = outputs["logits"].view(-1, 4)
                true = batch["labels"].to("cuda:0").view(-1)
                loss = loss_fn(logits, true)

            loss.backward()
            mv_avg = 0.95 * mv_avg + 0.05 * float(loss.detach().item())
            pbar.set_description(f"{n = }: {mv_avg:.6f}")

            if i % grad_acc_steps == 0 or i == len(dl_train):
                optim.step()
                optim.zero_grad()

            pbar.update(1)

        lr_scheduler.step()

    return logit_model_copy


def load_model(uri_model):
    if "bert" in uri_model:
        logit_model = segmentador.BERTSegmenter(uri_model=uri_model, device="cpu")
    else:
        logit_model = segmentador.LSTMSegmenter(uri_model=uri_model, device="cpu")

    return logit_model


def main():
    # NOTE: data info + download links available in:
    # https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
    dt = datasets.DatasetDict.load_from_disk("data/dataset_ulysses_segmenter_v2_active_learning_curated_only")
    dt_train_v1 = datasets.DatasetDict.load_from_disk("data/dataset_ulysses_segmenter_v1_weak_supervision")
    dt_train_v1 = dt_train_v1["train"]

    dt_curated_train = dt["train"]
    dt_curated_eval = dt["eval"]
    dt_curated_test = dt["test"]

    print("Non-filtered:")
    print(dt_curated_train)
    print(dt_curated_eval)
    print(dt_curated_test)
    print()

    dt_curated_eval = filter_leakage(dt_curated_eval, dt_curated_test)

    print("Filtered:")
    print(dt_curated_train)
    print(dt_curated_eval)
    print(dt_curated_test)
    print()

    dt_curated_train = dt_curated_train.map(fn_pad)
    dt_curated_eval = dt_curated_eval.map(fn_pad)
    dt_curated_test = dt_curated_test.map(fn_pad)

    dt_curated_train.set_format("torch")
    dt_curated_eval.set_format("torch")
    dt_curated_test.set_format("torch")

    min_ = 150
    max_ = min(len(dt_curated_train), len(dt_curated_eval))
    step = 100
    extra_test_insts = len(dt_curated_test)

    print(extra_test_insts)

    # NOTE: Model info + download links available in:
    # https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#available-models
    uris_model = (
        "256_hidden_dim_6000_vocab_size_1_layer_lstm.pt",
        "512_hidden_dim_6000_vocab_size_1_layer_lstm.pt",
        "2_layer_6000_vocab_size_bert_v1",
        "4_layer_6000_vocab_size_bert_v1",
    )

    batch_size = 3
    dl_test = torch.utils.data.DataLoader(dt_curated_test, batch_size=batch_size)

    for k in range(5):
        data = []
        for uri_model in uris_model:
            uri_model = os.path.abspath(uri_model)

            print(f"{uri_model = }")

            for n in range(min_, max_ + 1, step):
                cur_dt_train = datasets.Dataset.from_dict(dt_curated_train[:n])
                cur_dt_eval = datasets.Dataset.from_dict(dt_curated_eval[:n])
                cur_dt_mix = datasets.concatenate_datasets(
                    [
                        datasets.Dataset.from_dict(dt_curated_eval[:n]).train_test_split(
                            train_size=n // 2, shuffle=True, seed=(38278 + max_) * k + n + 1
                        )["train"],
                        datasets.Dataset.from_dict(dt_curated_train[:n]).train_test_split(
                            train_size=n // 2, shuffle=True, seed=(40331 + max_) * k + n + 2
                        )["train"],
                    ]
                )
                cur_dt_test_extra = datasets.Dataset.from_dict(dt_curated_train[n : n + extra_test_insts])

                cur_dt_test_extra = filter_leakage(cur_dt_test_extra, cur_dt_train, show_progress_bar=False)
                cur_dt_test_extra = filter_leakage(cur_dt_test_extra, cur_dt_eval, show_progress_bar=False)

                cur_dt_train.set_format("torch")
                cur_dt_eval.set_format("torch")
                cur_dt_mix.set_format("torch")
                cur_dt_test_extra.set_format("torch")

                assert len(cur_dt_train) == len(cur_dt_eval)
                assert len(cur_dt_train) == n
                assert len(cur_dt_test_extra) >= extra_test_insts * 0.80

                dl_train = torch.utils.data.DataLoader(cur_dt_train, batch_size=batch_size, shuffle=True)
                dl_eval = torch.utils.data.DataLoader(cur_dt_eval, batch_size=batch_size, shuffle=True)
                dl_mix = torch.utils.data.DataLoader(cur_dt_mix, batch_size=batch_size, shuffle=True)
                dl_test_ext = torch.utils.data.DataLoader(cur_dt_test_extra, batch_size=batch_size, shuffle=False)

                model_name = os.path.basename(uri_model)

                logit_model = load_model(uri_model)
                logit_model_m = train(logit_model, dl_mix, n=n)
                data.append([model_name, n, "mix", "test", *eval_model(logit_model_m, dl_test, device="cuda:0")])
                data.append([model_name, n, "mix", "test_ext", *eval_model(logit_model_m, dl_test_ext, device="cuda:0")])
                del logit_model_m
                torch.cuda.empty_cache()
                gc.collect()

                logit_model = load_model(uri_model)
                logit_model_a = train(logit_model, dl_train, n=n)
                data.append([model_name, n, "active", "test", *eval_model(logit_model_a, dl_test, device="cuda:0")])
                data.append([model_name, n, "active", "test_ext", *eval_model(logit_model_a, dl_test_ext, device="cuda:0")])
                del logit_model_a
                torch.cuda.empty_cache()
                gc.collect()

                logit_model = load_model(uri_model)
                logit_model_r = train(logit_model, dl_eval, n=n)
                data.append([model_name, n, "random", "test", *eval_model(logit_model_r, dl_test, device="cuda:0")])
                data.append([model_name, n, "random", "test_ext", *eval_model(logit_model_r, dl_test_ext, device="cuda:0")])
                del logit_model_r
                torch.cuda.empty_cache()
                gc.collect()

            del logit_model
            torch.cuda.empty_cache()
            gc.collect()
            print()

            out = pd.DataFrame.from_records(
                data,
                columns=[
                    "model",
                    "n",
                    "method",
                    "split",
                    "recall_macro",
                    "precision_macro",
                    "f1_macro",
                    "recall_c1",
                    "precision_c1",
                    "f1_c1",
                ],
            )
            out.to_csv(os.path.join(output_dir, f"avr_log_{k}.csv"), sep=",")


if __name__ == "__main__":
    main()

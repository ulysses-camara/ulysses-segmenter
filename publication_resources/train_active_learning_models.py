import typing as t
import argparse
import os
import datetime
import functools

import transformers
import torch
import torch.nn
import datasets
import tqdm

import segmentador

import utils


def train(
    segmenter_input_uri: str,
    segmenter_output_uri: str,
    train_data_uri: str,
    n_epochs: int,
    device: str,
    *,
    tokenizer_uri: t.Optional[str] = None,
) -> None:
    if os.path.isdir(segmenter_input_uri):
        segmenter = segmentador.BERTSegmenter(
            uri_model=segmenter_input_uri,
            device=device,
            local_files_only=True,
        )

    else:
        if tokenizer_uri is None:
            raise ValueError(
                "Tokenizer path must be provded for LSTM models (provide --tokenizer-uri=path/to/tokenizer argument)."
            )

        segmenter = segmentador.LSTMSegmenter(
            uri_model=segmenter_input_uri,
            uri_tokenizer=tokenizer_uri,
            device=device,
            local_files_only=True,
        )

    logit_model = segmenter.model

    optim = torch.optim.Adam(logit_model.parameters(), lr=5e-5)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.8)
    batch_size = 3
    batch_size_total = 3  # NOTE: useful if gradient accumulation is necessary.
    grad_acc_steps = batch_size_total // batch_size
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    dt = datasets.Dataset.load_from_disk(train_data_uri)
    dt = dt.map(functools.partial(utils.fn_pad_and_truncate, max_length=1024))
    dt.set_format("torch")

    dl = torch.utils.data.DataLoader(
        dt,
        drop_last=False,
        batch_size=batch_size,
    )

    pbar = tqdm.tqdm(range(n_epochs * len(dl)))
    logit_model.to(device)
    logit_model.train()

    for _ in range(n_epochs):
        for i, batch in enumerate(dl, 1):
            if isinstance(logit_model, transformers.BertForTokenClassification):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = logit_model(**batch)
                loss = outputs.loss

            else:
                outputs = logit_model(batch["input_ids"].to(device))
                logits = outputs["logits"].view(-1, 4)
                true_labels = batch["labels"].to(device).view(-1)
                loss = loss_fn(logits, true_labels)

            loss.backward()
            loss_val = float(loss.detach().item())
            pbar.set_description(f"{loss_val:.6f}")

            if i % grad_acc_steps == 0 or i == len(dl):
                optim.step()
                optim.zero_grad()

            pbar.update(1)

        lr_scheduler.step()

    torch.save(logit_model, segmenter_output_uri)
    print(f"Saved fine-tuned parameters (.pt) at '{segmenter_output_uri}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "segmenter_input_uri",
        help=(
            "URI to the pretrained segmenter model. Provide a path to a directory for BERT, and a path "
            "to the .pt for LSTM models."
        ),
    )
    parser.add_argument("train_data_uri")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device for training.")

    # NOTE: we used n_epochs=10 for final version, and n_epochs=2 during weak supervision data curation.
    parser.add_argument("--n-epochs", default=2)

    now = datetime.datetime.now()
    now = now.isoformat().split(".")[0]
    parser.add_argument(
        "--segmenter-output-uri",
        default=f"finetuned_segmenter_{now}.pt",
        type=str,
        help="Output URI for fine-tuned parameters.",
    )

    parser.add_argument("--tokenizer-uri", default=None, type=str, help="Tokenizer path. Only required for LSTM models.")

    args = parser.parse_args()
    train(**vars(args))

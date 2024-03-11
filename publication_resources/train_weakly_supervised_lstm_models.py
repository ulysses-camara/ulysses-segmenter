import os
import pathlib
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as tF
import pytorch_lightning as pl
import tokenizers
import datasets


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hfdf):
        self.hfdf = hfdf

    def __getitem__(self, idx):
        return self.hfdf[idx]

    def __len__(self):
        return len(self.hfdf)


class LitSegmenterBaseline(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        tokenizer_uri: str,
        dataset_uri: str,
        batch_size: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        num_classes: int = 4,
        pad_token: str = "[PAD]",
    ):
        super(LitSegmenterBaseline, self).__init__()

        self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_uri)

        self.batch_size = batch_size
        self.pad_id = self.tokenizer.get_vocab().get(pad_token, 0)

        def fn_pad_sequences(batch):
            X = [torch.tensor(x_i["input_ids"], dtype=torch.int) for x_i in batch]
            y = [torch.tensor(y_i["labels"]) for y_i in batch]

            X = nn.utils.rnn.pad_sequence(X, padding_value=self.pad_id, batch_first=True)
            y = nn.utils.rnn.pad_sequence(y, padding_value=-100, batch_first=True)

            return X, y

        self.fn_pad_sequences = fn_pad_sequences

        if isinstance(dataset_uri, str):
            self.hfdf = datasets.load_from_disk(dataset_uri)

        else:
            dfs = []
            for uri in dataset_uri:
                dfs.append(datasets.load_from_disk(uri))

            hfdf = {}
            for key in dfs[0].keys():
                hfdf[key] = datasets.concatenate_datasets([df[key] for df in dfs])

            self.hfdf = datasets.DatasetDict(hfdf)

        self.embeddings = nn.Embedding(
            num_embeddings=self.tokenizer.get_vocab_size(),
            embedding_dim=768,
            padding_idx=self.pad_id,
        )

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1,
            bidirectional=bidirectional,
            proj_size=0,
        )

        self.lin_out = nn.Linear(
            (1 + int(bidirectional)) * hidden_size,
            num_classes,
        )

    def forward(self, X):
        out = X

        if isinstance(out, str):
            out = self.tokenizer(out, return_tensors="pt")
            out = out["input_ids"]

        out = self.embeddings(out)
        out, *_ = self.lstm(out)
        out = self.lin_out(out)

        return out

    @staticmethod
    def _compute_pred_metrics(y_preds, y, phase: str) -> dict[str, float]:
        y_preds = y_preds.view(-1, y_preds.shape[-1])
        y = y.view(-1).squeeze()

        loss = F.cross_entropy(input=y_preds, target=y, ignore_index=-100)

        non_pad_inds = [i for i, cls_i in enumerate(y) if cls_i != -100]

        per_cls_recall = tF.recall(
            preds=y_preds[non_pad_inds, ...],
            target=y[non_pad_inds],
            num_classes=4,
            average=None,
        )

        per_cls_precision = tF.precision(
            preds=y_preds[non_pad_inds, ...],
            target=y[non_pad_inds],
            num_classes=4,
            average=None,
        )

        macro_precision = float(per_cls_precision.mean().item())
        macro_recall = float(per_cls_recall.mean().item())
        macro_f1_score = 2.0 * macro_precision * macro_recall / (1e-8 + macro_precision + macro_recall)

        out = {
            f"{(phase + '_') if phase != 'train' else ''}loss": loss,
            **{f"{phase}_cls_{i}_precision": float(val) for i, val in enumerate(per_cls_precision)},
            **{f"{phase}_cls_{i}_recall": float(val) for i, val in enumerate(per_cls_recall)},
            f"{phase}_macro_precision": macro_precision,
            f"{phase}_macro_recall": macro_recall,
            f"{phase}_macro_f1_score": macro_f1_score,
        }

        return out

    @staticmethod
    def _agg_stats(step_outputs):
        out = {}
        agg_items = collections.defaultdict(list)

        for items in step_outputs:
            for key, val in items.items():
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val)

                agg_items[key].append(val)

        for key, vals in agg_items.items():
            avg_vals = float(torch.stack(vals).mean().item())
            out[f"avg_{key}"] = avg_vals

        return out

    def training_step(self, batch, batch_idx: int):
        X, y = batch
        y_preds = self.forward(X)

        out = self._compute_pred_metrics(y_preds, y, phase="train")

        self.log_dict(
            out,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return out

    def training_epoch_end(self, training_step_outputs):
        out = self._agg_stats(training_step_outputs)

        self.log_dict(
            out,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx: int):
        X, y = batch
        y_preds = self.forward(X)

        out = self._compute_pred_metrics(y_preds, y, phase="val")

        self.log_dict(
            out,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        return out

    def validation_epoch_end(self, validation_step_outputs):
        out = self._agg_stats(validation_step_outputs)

        self.log_dict(
            out,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx: int):
        X, y = batch
        y_preds = self.forward(X)

        out = self._compute_pred_metrics(y_preds, y, phase="test")

        self.log_dict(
            out,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        return out

    def test_epoch_end(self, test_step_outputs):
        out = self._agg_stats(test_step_outputs)

        self.log_dict(
            out,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        df_train = HFDataset(self.hfdf["train"])

        train_dataloader = torch.utils.data.DataLoader(
            dataset=df_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=self.fn_pad_sequences,
        )

        return train_dataloader

    def val_dataloader(self):
        df_eval = HFDataset(self.hfdf["eval"])

        eval_dataloader = torch.utils.data.DataLoader(
            dataset=df_eval,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.fn_pad_sequences,
        )

        return eval_dataloader

    def test_dataloader(self):
        df_test = HFDataset(self.hfdf["test"])

        test_dataloader = torch.utils.data.DataLoader(
            dataset=df_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.fn_pad_sequences,
        )

        return test_dataloader


def main(args):
    configs = [
        (512, 32),
        (256, 32),
        (128, 64),
    ]

    for hidden_size, batch_size in configs:
        accumulate_grad_batches = 128 // batch_size

        model = LitSegmenterBaseline(
            hidden_size=hidden_size,
            batch_size=batch_size,
            tokenizer_uri="tokenizers/6000_subwords",
            dataset_uri=["data/df_tokenized_split_0_120000_6000"],
        )

        trainer = pl.Trainer.from_argparse_args(
            args,
            overfit_batches=0.0,
            accumulate_grad_batches=accumulate_grad_batches,
        )

        trainer.fit(model)
        trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args(
        """
        --gpu 1
        --max_epochs 4
        --log_every_n_steps 100
        --precision 32
    """.split()
    )

    main(args)

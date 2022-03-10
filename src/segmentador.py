"""Legal text segmenter."""
import typing as t
import regex

import transformers
import torch
import torch.nn
import numpy as np


class Segmenter:
    """TODO."""

    RE_BLANK_SPACES = regex.compile(r"\s+")
    RE_JUSTIFICATIVA = regex.compile(
        "|".join(
            (
                r"\s*".join("JUSTIFICATIVA"),
                r"\s*".join([*"JUSTIFICA", "[CÇ]", "[AÁÀÃÃ]", "O"]),
                r"\s*".join("ANEXOS"),
            )
        )
    )

    def __init__(
        self,
        uri_model: str = "neuralmind/bert-base-portuguese-cased",
        uri_tokenizer: t.Optional[str] = None,
        num_labels: int = 4,
        local_files_only: bool = True,
        device: str = "cuda",
        init_from_pretrained_weights: bool = False,
        config: t.Optional[transformers.BertConfig] = None,
        num_hidden_layers: int = 6,
        cache_dir: str = "../cache/models",
    ):
        self.local_files_only = bool(local_files_only)

        if config is None:
            config = transformers.BertConfig.from_pretrained(uri_model)
            config.max_position_embeddings = 1024
            config.num_hidden_layers = num_hidden_layers
            config.num_labels = num_labels

        if uri_tokenizer is None:
            uri_tokenizer = uri_model

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        if init_from_pretrained_weights:
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                uri_model,
                local_files_only=self.local_files_only,
                cache_dir=cache_dir,
            )

        else:
            model = transformers.AutoModelForTokenClassification.from_config(config)

        """
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            uri_model,
            num_labels=num_labels,
            local_files_only=self.local_files_only,
            cache_dir=cache_dir,
        )
        """

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            uri_tokenizer,
            local_files_only=self.local_files_only,
            cache_dir="../cache/tokenizers",
        )

        model.resize_token_embeddings(tokenizer.vocab_size)

        self._tokenizer = tokenizer
        self._model = model.to(device)

        self.pipeline = transformers.pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
        )

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def save_pretrained(self, save_directory: str) -> None:
        self.pipeline.save_pretrained(save_directory)

    @classmethod
    def preprocess_legal_text(
        cls, text: str, return_justificativa: bool = False
    ) -> t.Union[str, tuple[str, list[str]]]:
        """Apply minimal legal text preprocessing."""
        text = cls.RE_BLANK_SPACES.sub(" ", text)
        text = text.strip()
        text, *justificativa = cls.RE_JUSTIFICATIVA.split(text)

        if return_justificativa:
            return text, justificativa

        return text

    def segment_legal_text(self, text: str, return_justificativa: bool = False):
        """Segment `text`."""
        self._model.eval()

        text = self.preprocess_legal_text(
            text, return_justificativa=return_justificativa
        )

        if isinstance(text, tuple):
            text, justificativa = text

        tokens = self._tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors="pt",
            return_length=True,
        )
        num_tokens = tokens.pop("length")

        preds = []

        for i in range(0, num_tokens, 1024):
            subset = {}

            for key, vals in tokens.items():
                slice_ = vals[..., i : i + 1024]
                slice_ = slice_.to(self._model.device)
                subset[key] = slice_

            with torch.no_grad():
                model_out = self._model(**subset)

            model_out = model_out["logits"]
            model_out = model_out.cpu().numpy()
            model_out = model_out.argmax(axis=-1)
            preds.extend(model_out)

        preds = np.hstack(preds).astype(int, copy=False)
        segment_inds = np.hstack((0, np.flatnonzero(preds == 1), num_tokens))

        segs: list[str] = []

        for i, i_next in zip(segment_inds[:-1], segment_inds[1:]):
            split_ = tokens["input_ids"].numpy().ravel()[i:i_next]
            seg = self._tokenizer.decode(split_, skip_special_tokens=True)
            if seg:
                segs.append(seg)

        if return_justificativa:
            return segs, justificativa

        return segs

    def __call__(self, *args, **kwargs) -> str:
        return self.segment_legal_text(*args, **kwargs)


def main():
    seg = Segmenter()


if __name__ == "__main__":
    main()

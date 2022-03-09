"""Legal text segmenter."""
import typing as t
import re

import transformers
import torch
import torch.nn


class Segmenter:
    """TODO."""

    RE_BLANK_SPACES = re.compile(r"\s+")

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
    def preprocess_legal_text(cls, text: str) -> str:
        """Apply minimal legal text preprocessing."""
        text = cls.RE_BLANK_SPACES.sub(" ", text)
        text = text.strip()
        return text

    def segment_legal_text(self, text: str):
        """Segment `text`."""
        text = self.preprocess_legal_text(text)
        return self.pipeline(text)

    def __call__(self, text: str) -> str:
        return self.segment_legal_text(text)


def main():
    seg = Segmenter()


if __name__ == "__main__":
    main()

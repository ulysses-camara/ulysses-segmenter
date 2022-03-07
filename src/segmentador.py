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
        num_hidden_layers: int = 12,
        device: str = "cuda",
    ):
        self.local_files_only = bool(local_files_only)

        if uri_tokenizer is None:
            uri_tokenizer = uri_model

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        config = transformers.BertConfig.from_pretrained(uri_model)
        config.max_position_embeddings = 1024
        config.num_hidden_layers = num_hidden_layers
        config.num_labels = num_labels

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            uri_tokenizer,
            local_files_only=self.local_files_only,
            cache_dir="../cache/tokenizers",
        )

        model = transformers.AutoModelForTokenClassification.from_config(config)

        model = model.to(device)

        """
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            uri_model,
            num_labels=num_labels,
            local_files_only=self.local_files_only,
            cache_dir="../cache/models",
            num_hidden_layers=num_hidden_layers,
        )

        emb = model.bert.embeddings.position_embeddings
        emb = torch.nn.Embedding.from_pretrained(
            torch.cat(
                [
                    emb.weight,
                    torch.zeros(
                        1024 - emb.num_embeddings, 768, device=emb.weight.device
                    ),
                ],
                axis=0,
            )
        )
        tokenizer.model_max_length = 1024
        model.config.max_position_embeddings = 1024
        model.bert.embeddings.position_embeddings = emb
        """

        self.pipeline = transformers.pipeline(
            "token-classification", model=model, tokenizer=tokenizer
        )

    @property
    def model(self):
        return self.pipeline.model

    @property
    def tokenizer(self):
        return self.pipeline.tokenizer

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

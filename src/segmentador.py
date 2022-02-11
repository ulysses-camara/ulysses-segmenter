"""Legal text segmenter."""
import typing as t
import re

import transformers


class Segmenter:
    """TODO."""

    RE_BLANK_SPACES = re.compile(r"\s+")

    def __init__(
        self,
        uri_model: str = "neuralmind/bert-base-portuguese-cased",
        uri_tokenizer: t.Optional[str] = None,
        num_labels: int = 4,
        local_files_only: bool = True,
    ):
        self.local_files_only = bool(local_files_only)

        if uri_tokenizer is None:
            uri_tokenizer = uri_model

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            uri_tokenizer, local_files_only=self.local_files_only
        )

        model = transformers.AutoModelForTokenClassification.from_pretrained(
            uri_model, num_labels=num_labels, local_files_only=self.local_files_only
        )

        self.pipeline = transformers.pipeline(
            "token-classification", model=model, tokenizer=tokenizer
        )

    @property
    def model(self):
        return self.pipeline.model

    @property
    def tokenizer(self):
        return self.pipeline.tokenizer

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

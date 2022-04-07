import typing as t
import abc

import transformers
import datasets
import torch
import numpy as np
import numpy.typing as npt
import regex


InputHandlerOutputType = tuple[
    transformers.tokenization_utils_base.BatchEncoding, t.Optional[list[str]], int
]


class _BaseInputHandler(abc.ABC):
    pass


class InputHandlerString(_BaseInputHandler):
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

    @classmethod
    def setup_regex_justificativa(
        cls,
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ) -> regex.Pattern:
        """Compile or set default 'JUSTIFICATIVA' block regex.

        If the provided regex is already compiled, this function simply returns its own
        argument.
        """
        if regex_justificativa is None:
            regex_justificativa = cls.RE_JUSTIFICATIVA

        if isinstance(regex_justificativa, str):
            regex_justificativa = regex.compile(regex_justificativa)

        return regex_justificativa

    @classmethod
    def preprocess_legal_text(
        cls,
        text: str,
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ) -> tuple[str, list[str]]:
        """Apply minimal legal text preprocessing.

        The preprocessing steps are:
        1. Coalesce all blank spaces in text;
        2. Remove all trailing and leading blank spaces; and
        3. Pre-segment text into legal text content and `justificativa`.

        Parameters
        ----------
        text : str
            Text to be preprocessed.

        regex_justificativa : str, regex.Pattern or None, default=None
            Regular expression specifying how the `justificativa` portion from legal
            documents should be detected. If None, will use the pattern predefined in
            `Segmenter.RE_JUSTIFICATIVA` class attribute.

        Returns
        -------
        preprocessed_text : str
            Content from `text` after the preprocessing steps.

        justificativa_block : list[str]
            Detected legal text `justificativa` blocks.
        """
        text = cls.RE_BLANK_SPACES.sub(" ", text)
        text = text.strip()

        regex_justificativa = cls.setup_regex_justificativa(regex_justificativa)
        text, *justificativa = regex_justificativa.split(text)

        return text, justificativa

    @classmethod
    def tokenize(
        cls,
        text: str,
        tokenizer: transformers.models.bert.tokenization_bert_fast.BertTokenizerFast,
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
        *args,
        **kwargs,
    ) -> InputHandlerOutputType:
        text, justificativa = cls.preprocess_legal_text(
            text,
            regex_justificativa=regex_justificativa,
        )

        tokens = tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors="pt",
            return_length=True,
        )

        num_tokens = tokens.pop("length")

        return tokens, justificativa, num_tokens


class InputHandlerMapping(_BaseInputHandler):
    @classmethod
    def _val_to_tensor(
        cls, val: t.Sequence[int], *args, **kwargs
    ) -> t.Union[npt.NDArray[torch.Tensor], torch.Tensor]:
        if torch.is_tensor(val):
            return val

        if isinstance(val, np.ndarray):
            return torch.from_numpy(val)

        try:
            ret = torch.tensor(val)

        except ValueError:
            ret = torch.from_numpy(np.concatenate(val))

        return ret

    @classmethod
    def tokenize(
        cls,
        text: t.MutableMapping[str, t.Union[list[int], torch.Tensor]],
        *args,
        **kwargs,
    ) -> InputHandlerOutputType:
        tokens = transformers.tokenization_utils_base.BatchEncoding(
            {key: cls._val_to_tensor(val) for key, val in text.items()}
        )
        justificativa = None
        num_tokens = len(tokens["input_ids"])

        return tokens, justificativa, num_tokens


class InputHandlerDataset(_BaseInputHandler):
    @classmethod
    def tokenize(cls, text: datasets.Dataset, *args, **kwargs) -> InputHandlerOutputType:
        return InputHandlerMapping.tokenize(text.to_dict())


def tokenize_input(text: str, *args: t.Any, **kwargs: t.Any) -> InputHandlerOutputType:
    if isinstance(text, str):
        return InputHandlerString.tokenize(text, *args, **kwargs)

    if isinstance(text, datasets.arrow_dataset.Dataset):
        return InputHandlerDataset.tokenize(text, *args, **kwargs)

    if hasattr(text, "items"):
        return InputHandlerMapping.tokenize(text, *args, **kwargs)

    raise TypeError(
        f"Unrecognized 'text' type: {type(text)}. Please cast your text input to "
        "a string, datasets.Dataset, or a valid key-value mapping."
    )

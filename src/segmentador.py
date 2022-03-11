"""Legal text segmenter."""
import typing as t
import collections

import regex
import transformers
import torch
import torch.nn.functional as F
import torch.nn
import numpy as np


class Segmenter:
    r"""Brazilian Portuguese legal text segmenter class.

    Uses a pretrained Transformer Encoder to segment Brazilian Portuguese legal texts.
    The pretrained models support texts up to 1024 subwords. Texts larger than this
    value are pre-segmented into 1024 subword blocks, and each block is feed to the
    segmenter individually.

    Parameters
    ----------
    uri_model : str, default="neuralmind/bert-base-portuguese-cased"
        URI to load pretrained model from. May be a Hugginface HUB URL (if
        `local_files_only=False`) or a local file.

    uri_tokenizer : str or None, default=None
        URI to pretrained text Tokenizer. If None, will load the tokenizer from
        the `uri_model` path.

    local_files_only : bool, default=True
        If True, will search only for local pretrained model and tokenizers.
        If False, may download models from Huggingface HUB, if necessary.

    device : {'cpu', 'cuda'}, default="cpu"
        Device to segment document content.

    init_from_pretrained_weights : bool, default=True
        if True, load pretrained weights from the specified `uri_model` argument.
        If False, load only the model configuration from the same argument.

    config : t.Optional[transformers.BertConfig], default=None
        Custom model configuration. Used only if `init_from_pretrained_weights=False`.
        If `init_from_pretrained_weights=False` and `config=None`, will load the
        configuration file from `uri_model` with the following changes:
        - config.max_position_embeddings = 1024
        - config.num_hidden_layers = num_hidden_layers
        - config.num_labels = num_labels

    num_labels : int, default=4
        Number of labels in the configuration file.

    num_hidden_layers : int, default=6
        Number of maximum Transformer Encoder hidden layers. If the model has more hidden
        layers than the specified value in this parameter, later hidden layers will be
        removed.

    cache_dir_model : str, default="../cache/models"
        Cache directory for transformer encoder model.

    cache_dir_tokenizer : str, default="../cache/tokenizers"
        Cache directory for text tokenizer.

    regex_justificativa : str, regex.Pattern or None, default=None
        Regular expression specifying how the `justificativa` portion from legal
        documents should be detected. If None, will use the pattern predefined in
        `Segmenter.RE_JUSTIFICATIVA` class attribute.
    """

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
        local_files_only: bool = True,
        device: str = "cpu",
        init_from_pretrained_weights: bool = True,
        config: t.Optional[transformers.BertConfig] = None,
        num_labels: int = 4,
        num_hidden_layers: int = 6,
        cache_dir_model: str = "../cache/models",
        cache_dir_tokenizer: str = "../cache/tokenizers",
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ):
        labels = ("NO-OP", "SEG_START", "NOISE_START", "NOISE_END")

        self.local_files_only = bool(local_files_only)

        if config is None:
            config = transformers.BertConfig.from_pretrained(uri_model)
            config.max_position_embeddings = 1024
            config.num_hidden_layers = num_hidden_layers
            config.num_labels = num_labels
            config.label2id = dict(zip(labels, range(num_labels)))
            config.id2label = dict(zip(range(num_labels), labels))

        if uri_tokenizer is None:
            uri_tokenizer = uri_model

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        if init_from_pretrained_weights:
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                uri_model,
                local_files_only=self.local_files_only,
                cache_dir=cache_dir_model,
                label2id=config.label2id,
                id2label=config.id2label,
            )

        else:
            model = transformers.AutoModelForTokenClassification.from_config(config)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            uri_tokenizer,
            local_files_only=self.local_files_only,
            cache_dir=cache_dir_tokenizer,
        )

        model.resize_token_embeddings(tokenizer.vocab_size)

        self._tokenizer = tokenizer
        self._model = model.to(device)

        self.pipeline = transformers.pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
        )

        self.regex_justificativa = self.setup_regex_justificativa(regex_justificativa)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def save_pretrained(self, save_directory: str) -> None:
        """Save pipeline (model and tokenizer) in `save_directory` path."""
        self.pipeline.save_pretrained(save_directory)

    @classmethod
    def setup_regex_justificativa(
        cls,
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ) -> regex.Pattern:
        if regex_justificativa is None:
            regex_justificativa = cls.RE_JUSTIFICATIVA

        if isinstance(regex_justificativa, str):
            regex_justificativa = regex.compile(regex_justificativa)

        return regex_justificativa

    @classmethod
    def preprocess_legal_text(
        cls,
        text: str,
        return_justificativa: bool = False,
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ) -> t.Union[str, tuple[str, list[str]]]:
        """Apply minimal legal text preprocessing.

        The preprocessing steps are:
        1. Coalesce all blank spaces in text;
        2. Remove all trailing and leading blank spaces; and
        3. Pre-segment text into legal text content and `justificativa`.

        Parameters
        ----------
        text : str
            Text to be preprocessed.

        return_justificativa : bool, default=False
            If True, return a tuple in the format (content, justificativa).
            If False, return only `content`.

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
            Only returned if `return_justificativa=True`.
        """
        text = cls.RE_BLANK_SPACES.sub(" ", text)
        text = text.strip()

        regex_justificativa = cls.setup_regex_justificativa(regex_justificativa)
        text, *justificativa = regex_justificativa.split(text)

        if return_justificativa:
            return text, justificativa

        return text

    def segment_legal_text(
        self,
        text: str,
        return_justificativa: bool = False,
    ) -> t.Union[list[str], tuple[list[str], list[str]]]:
        """Segment legal `text`.

        The pretrained model support texts up to 1024 subwords. Texts larger than this
        value are pre-segmented into 1024 subword blocks, and each block is feed to the
        segmenter individually.

        Parameters
        ----------
        text : str
            Legal text to be segmented.

        return_justificativa : bool, default=False
            If True, return a tuple in the format (content, justificativa).
            If False, return only `content`.

        Returns
        -------
        preprocessed_text : list[str]
            Content from `text` after the preprocessing steps.

        justificativa_block : list[str]
            Detected legal text `justificativa` blocks.
            Only returned if `return_justificativa=True`.
        """
        self._model.eval()

        preproc_result = self.preprocess_legal_text(
            text,
            return_justificativa=return_justificativa,
            regex_justificativa=self.regex_justificativa,
        )

        if isinstance(preproc_result, tuple):
            text, justificativa = preproc_result

        else:
            text = preproc_result

        tokens = self._tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors="pt",
            return_length=True,
        )
        num_tokens = tokens.pop("length")

        subset = collections.defaultdict(list)

        for i in range(0, num_tokens, 1024):
            for key, vals in tokens.items():
                slice_ = vals[..., i : i + 1024]
                subset[key].append(slice_)

        for key, vals in subset.items():
            last_len = max(vals[-1].size())

            if last_len < 1024:
                vals[-1] = F.pad(
                    input=vals[-1],
                    pad=(0, 1024 - last_len),
                    mode="constant",
                    value=self._tokenizer.pad_token_id,
                )

            subset[key] = torch.vstack(vals).to(self._model.device)

        with torch.no_grad():
            model_out = self._model(**subset)

        model_out = model_out["logits"]
        model_out = model_out.cpu().numpy()
        model_out = model_out.argmax(axis=-1)

        seg_cls_id = self._model.config.label2id.get("SEG_START", 1)
        segment_start_inds = np.flatnonzero(model_out == seg_cls_id)
        segment_start_inds = np.hstack((0, segment_start_inds, num_tokens))

        segs: list[str] = []

        for i, i_next in zip(segment_start_inds[:-1], segment_start_inds[1:]):
            split_ = tokens["input_ids"].numpy().ravel()[i:i_next]
            seg = self._tokenizer.decode(split_, skip_special_tokens=True)
            if seg:
                segs.append(seg)

        if return_justificativa:
            return segs, justificativa

        return segs

    def __call__(
        self, *args, **kwargs
    ) -> t.Union[list[str], tuple[list[str], list[str]]]:
        return self.segment_legal_text(*args, **kwargs)

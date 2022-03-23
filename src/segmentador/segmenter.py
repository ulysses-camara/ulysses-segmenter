"""Legal text segmenter."""
import typing as t
import warnings
import collections

import regex
import transformers
import torch
import torch.nn.functional as F
import torch.nn
import numpy as np
import numpy.typing as npt

from . import poolers


class _BaseSegmenter:
    """Base class for Segmenter models."""

    NUM_CLASSES = 4
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

    _RE_REPR_TOKENIZER_ADJUST_01 = regex.compile(r"(?<=[\(,]\s*)(?=[a-z_]+\s*=)")
    _RE_REPR_TOKENIZER_ADJUST_02 = regex.compile(r"(?<=[{,]\s*)(?=[a-z_']+\s*:)")

    def __init__(
        self,
        uri_tokenizer: t.Optional[str] = None,
        inference_pooling_operation: t.Literal[
            "max", "sum", "gaussian", "assymetric-max"
        ] = "assymetric-max",
        local_files_only: bool = True,
        device: str = "cpu",
        cache_dir_tokenizer: str = "../cache/tokenizers",
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ):
        self.local_files_only = bool(local_files_only)

        self._tokenizer: transformers.models.bert.tokenization_bert_fast.BertTokenizerFast = (
            transformers.AutoTokenizer.from_pretrained(
                uri_tokenizer,
                local_files_only=self.local_files_only,
                cache_dir=cache_dir_tokenizer,
            )
        )

        self.regex_justificativa = self.setup_regex_justificativa(regex_justificativa)

        self._moving_window_pooler = poolers.AutoMovingWindowPooler(
            pooling_operation=inference_pooling_operation,
        )

        self._model: t.Union[
            torch.nn.Module,
            transformers.models.bert.modeling_bert.BertForTokenClassification,
        ]

        self.device = device

    def __call__(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Union[list[str], tuple[list[t.Any], ...]]:
        return self.segment_legal_text(*args, **kwargs)

    def __repr__(self) -> str:
        strs: list[str] = []

        strs.append(f"{self.__class__.__name__} pipeline")
        strs.append(" o Regex JUSTIFICATIVA pattern:")
        strs.append(" |   '''")
        strs.append(" |   " + self.regex_justificativa.pattern.replace("|", "|\n |   "))
        strs.append(" |   '''")
        strs.append(" | ")
        strs.append(f" o Device: {self.device}")

        strs.append(" | ")
        strs.append("(1) Tokenizer:")

        text_tokenizer = str(self._tokenizer)
        text_tokenizer = self._RE_REPR_TOKENIZER_ADJUST_01.sub("\n  ", text_tokenizer)
        text_tokenizer = self._RE_REPR_TOKENIZER_ADJUST_02.sub("\n    ", text_tokenizer)
        strs.append(" | " + text_tokenizer.replace("\n", "\n |  "))

        strs.append(" | ")
        strs.append("(2) Segmenter model:")
        strs.append(" | " + str(self._model).replace("\n", "\n |  "))

        strs.append(" | ")
        strs.append("(3) Inference pooler:")
        strs.append("   " + str(self._moving_window_pooler).replace("\n", "\n |  "))

        return "\n".join(strs)

    @property
    def model(
        self,
    ) -> t.Union[
        torch.nn.Module,
        transformers.models.bert.modeling_bert.BertForTokenClassification,
    ]:
        # pylint: disable='missing-function-docstring'
        return self._model

    @property
    def tokenizer(
        self,
    ) -> transformers.models.bert.tokenization_bert_fast.BertTokenizerFast:
        # pylint: disable='missing-function-docstring'
        return self._tokenizer

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

    def _build_minibatches(
        self,
        tokens: transformers.tokenization_utils_base.BatchEncoding,
        num_tokens: int,
        batch_size: int,
        moving_window_size: int,
        window_shift_size: int,
    ) -> list[transformers.tokenization_utils_base.BatchEncoding]:
        """Break BatchEncoding items into proper smaller minibatches."""
        minibatches: list[transformers.tokenization_utils_base.BatchEncoding] = []
        minibatch = transformers.tokenization_utils_base.BatchEncoding()

        total_minibatches = 1 + max(
            0, int(np.ceil((num_tokens - moving_window_size) / window_shift_size))
        )

        for i in range(total_minibatches):
            i_start = i * window_shift_size
            i_end = i_start + moving_window_size

            for key, vals in tokens.items():
                slice_ = vals[..., i_start:i_end]

                minibatch.setdefault(key, [])
                minibatch[key].append(slice_)

            if (i + 1) % batch_size == 0:
                minibatches.append(minibatch)
                minibatch = transformers.tokenization_utils_base.BatchEncoding()

        if minibatch:
            minibatches.append(minibatch)

        for minibatch in minibatches:
            for key, vals in minibatch.items():
                for i in reversed(range(len(vals))):
                    cur_len = int(max(vals[i].size()))

                    if cur_len >= moving_window_size:
                        break

                    vals[i] = F.pad(
                        input=vals[i],
                        pad=(0, moving_window_size - cur_len),
                        mode="constant",
                        value=int(self._tokenizer.pad_token_id or 0),
                    )

                minibatch[key] = torch.vstack(vals)

        return minibatches

    def _generate_segments_from_labels(
        self,
        tokens: transformers.tokenization_utils_base.BatchEncoding,
        num_tokens: int,
        label_ids: npt.NDArray[np.int32],
    ) -> list[str]:
        """Convert predicted labels and subword tokens to text segments."""
        seg_cls_id: int

        try:
            seg_cls_id = self._model.config.label2id.get("SEG_START", 1)  # type: ignore

        except AttributeError:
            seg_cls_id = 1

        segment_start_inds = np.flatnonzero(label_ids == seg_cls_id)
        segment_start_inds = np.hstack((0, segment_start_inds, num_tokens))

        segs: list[str] = []
        np_token_ids: npt.NDArray[np.int32] = tokens["input_ids"].numpy().ravel()

        for i, i_next in zip(segment_start_inds[:-1], segment_start_inds[1:]):
            split_ = np_token_ids[i:i_next]
            seg = self._tokenizer.decode(split_, skip_special_tokens=True)
            if seg:
                segs.append(seg)

        return segs

    @staticmethod
    def _pack_results(
        keys: list[str], vals: list[t.Any], inclusion: list[bool]
    ) -> t.Union[list[str], tuple[list[t.Any], ...]]:
        """Build result tuple (if more than one value) or return segment list."""
        ret_keys: list[str] = []
        ret_vals: list[t.Any] = []

        for key, val, inc in zip(keys, vals, inclusion):
            if not inc:
                continue

            ret_keys.append(key)
            ret_vals.append(val)

        if len(ret_vals) == 1:
            segs: list[str] = ret_vals[0]
            return segs

        ret_type = collections.namedtuple("SegmentationResults", ret_keys)  # type: ignore

        return ret_type(*ret_vals)

    def segment_legal_text(
        self,
        text: t.Union[str, dict[str, list[int]]],
        batch_size: int = 32,
        moving_window_size: int = 1024,
        window_shift_size: t.Union[float, int] = 0.5,
        return_justificativa: bool = False,
        return_labels: bool = False,
        return_logits: bool = False,
    ) -> t.Union[list[str], tuple[list[t.Any], ...]]:
        """Segment legal `text`.

        The pretrained model support texts up to 1024 subwords. Texts larger than this
        value are pre-segmented into 1024 subword blocks, and each block is feed to the
        segmenter individually.

        The block size can be configured to smaller (not larger) values using the
        `moving_window_size` from `BERTSegmenter.segment_legal_text` method during inference.

        Parameters
        ----------
        text : str or dict[str, list[int]]
            Legal text to be segmented.

        batch_size : int, default=32
            Maximum batch size feed document blocks in parallel to model. Higher values
            leads to faster inference with higher memory cost.

        moving_window_size : int, default=1024
            Moving window size, which corresponds to the maximum number of subwords feed in
            parallel to the segmenter model. Higher values leads to larger contexts for every
            tokens, at the expense of higher memory usage.

        window_shift_size : int or float, default=0.5
            Moving window shift size, to feed documents larger than 1024 subwords tokens into
            the segmenter model.
            - If integer, specify exactly the shift size per step, and it must be in [1, 1024]
            range.
            - If float, the shift size is calculated from the corresponding fraction of the window
            size (1024 subword tokens), and it must be in the (0.0, 1.0] range.
            Overlapping logits are combined using the strategy specified by the argument
            `inference_pooling_operation` in Segmenter model initialization, and the final
            prediction for each token is derived from the combined logits.

        return_justificativa : bool, default=False
            If True, return contents from the 'justificativa' block from document.

        return_labels : bool, default=False
            If True, return label list for each token.

        return_logits : bool, default=False
            If True, return logit array for each token.

        Returns
        -------
        preprocessed_text : list[str]
            Content from `text` after the preprocessing steps.

        justificativa_block : list[str]
            Detected legal text `justificativa` blocks.
            Only returned if `return_justificativa=True`.

        labels : npt.NDArray[np.int32] of shape (N,)
            Predicted labels for each token, where `N` is the length of tokenized
            document (in subword units). The `-100` labels is a special legal, and
            ignored while computing the loss function during training.
            Only returned if `return_labels=True`.

        logits : npt.NDArray[np.float64] of shape (N, C)
            Predicted logits for each token, where `N` is the length of tokenized
            document (in subword units), and `C` is equal to the `Segmenter.NUM_CLASSES`
            attribute.
            Only returned if `return_logits=True`.
        """
        if batch_size < 1:
            raise ValueError(f"'batch_size' parameter must be >= 1 (got {batch_size=}).")

        if moving_window_size < 1:
            raise ValueError(
                f"'moving_window_size' parameter must be >= 1 (got {moving_window_size=})."
            )

        try:
            max_moving_window_size_allowed = int(
                self._model.config.max_position_embeddings  # type: ignore
            )

            if moving_window_size > max_moving_window_size_allowed:
                warnings.warn(
                    message=(
                        "'moving_window_size' is larger than model's positional embeddings "
                        f"({moving_window_size=}, {max_moving_window_size_allowed=}). "
                        "Will set 'moving_window_size' to the maximum allowed value."
                    ),
                    category=UserWarning,
                )
                moving_window_size = max_moving_window_size_allowed

        except AttributeError:
            pass

        if isinstance(window_shift_size, float):
            if not 0.0 < window_shift_size <= 1.0:
                raise ValueError("If 'window_shift_size' is a float, it must be in (0, 1] range.")

            window_shift_size = int(np.ceil(moving_window_size * window_shift_size))

        if window_shift_size < 1:
            raise ValueError(
                f"'window_shift_size' parameter must be >= 1 (got '{window_shift_size=}')."
            )

        if window_shift_size > moving_window_size:
            warnings.warn(
                message=(
                    f"'window_shift_size' parameter must be <= {moving_window_size} "
                    f"(got '{window_shift_size=}'). "
                    f"Will set it to {moving_window_size} automatically."
                ),
                category=UserWarning,
            )
            window_shift_size = moving_window_size

        if isinstance(text, str):
            preproc_result = self.preprocess_legal_text(
                text,
                return_justificativa=return_justificativa,
                regex_justificativa=self.regex_justificativa,
            )

            if isinstance(preproc_result, tuple):
                text, justificativa = preproc_result

            else:
                text, justificativa = preproc_result, None

            tokens = self._tokenizer(
                text,
                padding=False,
                truncation=False,
                return_tensors="pt",
                return_length=True,
            )

            num_tokens = tokens.pop("length")
        else:

            tokens = transformers.tokenization_utils_base.BatchEncoding({
                key: val if torch.is_tensor(val) else torch.tensor(val)
                for key, val in text.items()
            })
            justificativa = None
            num_tokens = len(tokens["input_ids"])

        minibatches = self._build_minibatches(
            tokens=tokens,
            num_tokens=num_tokens,
            batch_size=batch_size,
            moving_window_size=moving_window_size,
            window_shift_size=int(window_shift_size),
        )

        self._model.eval()
        all_logits: list[npt.NDArray[np.float64]] = []

        with torch.no_grad():
            for minibatch in minibatches:
                minibatch = minibatch.to(self.device)
                model_out = self._model(**minibatch)
                model_out = model_out["logits"]
                model_out = model_out.cpu().numpy()
                all_logits.append(model_out)

        logits = np.vstack(all_logits)
        del all_logits

        logits = self._moving_window_pooler(
            logits=logits,
            window_shift_size=window_shift_size,
        )

        label_ids = logits.argmax(axis=-1)
        label_ids = label_ids.squeeze()

        segs = self._generate_segments_from_labels(
            tokens=tokens, num_tokens=num_tokens, label_ids=label_ids
        )

        label_ids = label_ids[:num_tokens]
        logits = logits.reshape(-1, self.NUM_CLASSES)
        logits = logits[:num_tokens, :]

        assert label_ids.size == logits.shape[0]

        ret = self._pack_results(
            keys=["segments", "justificativa", "labels", "logits"],
            vals=[segs, justificativa, label_ids, logits],
            inclusion=[True, return_justificativa, return_labels, return_logits],
        )

        return ret


class BERTSegmenter(_BaseSegmenter):
    """BERT segmenter for PT-br legal text data.

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

    inference_pooling_operation : {"max", "sum", "gaussian", "assymetric-max"},\
            default="assymetric-max"
        Specify the strategy used to combine logits during model inference for documents
        larger than 1024 subword tokens. Larger documents are sharded into possibly overlapping
        windows of 1024 subwords each. Thus, a single token may have multiple logits (and,
        therefore, predictions) associated with it. This argument defines how exactly the
        logits should be combined in order to derive the final verdict for that said token.
        The possible choices for this argument are:
        - `max`: take the maximum logit of each token;
        - `sum`: sum the logits associated with the same token;
        - `gaussian`: build a gaussian filter that weights higher logits based on how close
            to the window center they are, diminishing its weights closer to the window
            limits; and
        - `assymetric-max`: take the maximum logit of each token for all classes other than
            the `No-operation` class, which in turn receives the minimum among all corresponding
            logits instead.

    local_files_only : bool, default=True
        If True, will search only for local pretrained model and tokenizers.
        If False, may download models from Huggingface HUB, if necessary.

    device : {'cpu', 'cuda'}, default="cpu"
        Device to segment document content.

    init_from_pretrained_weights : bool, default=True
        if True, load pretrained weights from the specified `uri_model` argument.
        If False, load only the model configuration from the same argument.

    config : transformers.BertConfig or None, default=None
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

    def __init__(
        self,
        uri_model: str = "neuralmind/bert-base-portuguese-cased",
        uri_tokenizer: t.Optional[str] = None,
        inference_pooling_operation: t.Literal[
            "max", "sum", "gaussian", "assymetric-max"
        ] = "assymetric-max",
        local_files_only: bool = True,
        device: str = "cpu",
        init_from_pretrained_weights: bool = True,
        config: t.Optional[t.Union[transformers.BertConfig, transformers.PretrainedConfig]] = None,
        num_labels: int = 4,
        num_hidden_layers: int = 6,
        cache_dir_model: str = "../cache/models",
        cache_dir_tokenizer: str = "../cache/tokenizers",
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ):
        super().__init__(
            uri_tokenizer=uri_tokenizer if uri_tokenizer is not None else uri_model,
            local_files_only=local_files_only,
            inference_pooling_operation=inference_pooling_operation,
            device=device,
            cache_dir_tokenizer=cache_dir_tokenizer,
            regex_justificativa=regex_justificativa,
        )

        labels = ("NO-OP", "SEG_START", "NOISE_START", "NOISE_END")

        if config is None:
            config = transformers.BertConfig.from_pretrained(uri_model)
            config.max_position_embeddings = 1024
            config.num_hidden_layers = num_hidden_layers
            config.num_labels = num_labels
            config.label2id = dict(zip(labels, range(num_labels)))
            config.id2label = dict(zip(range(num_labels), labels))

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

        model.resize_token_embeddings(self._tokenizer.vocab_size)

        self._model: transformers.models.bert.modeling_bert.BertForTokenClassification = model.to(
            device
        )


# Alias for 'BERTSegmenter'.
Segmenter = BERTSegmenter


class _LSTMSegmenterTorchModule(torch.nn.Module):
    """Bidirecional LSTM Torch model for legal document segmentation."""

    def __init__(
        self,
        lstm_hidden_layer_size: int,
        lstm_num_layers: int,
        num_embeddings: int,
        pad_id: int,
        num_classes: int,
    ):
        super().__init__()

        self.embeddings = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=768,
            padding_idx=pad_id,
        )

        self.lstm = torch.nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden_layer_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            proj_size=0,
        )

        self.lin_out = torch.nn.Linear(
            2 * lstm_hidden_layer_size,
            num_classes,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> dict[str, torch.Tensor]:
        # pylint: disable='missing-function-docstring', 'unused-argument'
        out = input_ids

        out = self.embeddings(out)
        out, *_ = self.lstm(out)
        out = self.lin_out(out)

        return {"logits": out}


class LSTMSegmenter(_BaseSegmenter):
    """Bidirectional LSTM segmenter for PT-br legal text data.

    Uses a pretrained Bidirectional LSTM to segment Brazilian Portuguese legal texts.

    Parameters
    ----------
    uri_model : str
        URI to load pretrained model from.

    uri_tokenizer : str or None
        URI to pretrained text Tokenizer.

    lstm_hidden_layer_size : int
        Dimension of LSTM model hidden layer.

    lstm_num_layers : int
        Number of layers in LSTM model.

    inference_pooling_operation : {"max", "sum", "gaussian", "assymetric-max"},\
            default="assymetric-max"
        Specify the strategy used to combine logits during model inference for documents
        larger than `moving_window_size` subword tokens (see `LSTMSegmenter.segment_legal_text`
        documentation). Larger documents are sharded into possibly overlapping windows of
        `moving_window_size` subwords each. Thus, a single token may have multiple logits (and,
        therefore, predictions) associated with it. This argument defines how exactly the
        logits should be combined in order to derive the final verdict for that said token.
        The possible choices for this argument are:
        - `max`: take the maximum logit of each token;
        - `sum`: sum the logits associated with the same token;
        - `gaussian`: build a gaussian filter that weights higher logits based on how close
            to the window center they are, diminishing its weights closer to the window
            limits; and
        - `assymetric-max`: take the maximum logit of each token for all classes other than
            the `No-operation` class, which in turn receives the minimum among all corresponding
            logits instead.

    local_files_only : bool, default=True
        If True, will search only for local pretrained model and tokenizers.
        If False, may download models from Huggingface HUB, if necessary.

    device : {'cpu', 'cuda'}, default="cpu"
        Device to segment document content.

    cache_dir_tokenizer : str, default="../cache/tokenizers"
        Cache directory for text tokenizer.

    regex_justificativa : str, regex.Pattern or None, default=None
        Regular expression specifying how the `justificativa` portion from legal
        documents should be detected. If None, will use the pattern predefined in
        `Segmenter.RE_JUSTIFICATIVA` class attribute.
    """

    def __init__(
        self,
        uri_model: str,
        uri_tokenizer: t.Optional[str],
        lstm_hidden_layer_size: int,
        lstm_num_layers: int,
        inference_pooling_operation: t.Literal[
            "max", "sum", "gaussian", "assymetric-max"
        ] = "assymetric-max",
        local_files_only: bool = True,
        device: str = "cpu",
        cache_dir_tokenizer: str = "../cache/tokenizers",
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ):
        super().__init__(
            uri_tokenizer=uri_tokenizer,
            local_files_only=local_files_only,
            inference_pooling_operation=inference_pooling_operation,
            device=device,
            cache_dir_tokenizer=cache_dir_tokenizer,
            regex_justificativa=regex_justificativa,
        )

        self._model = _LSTMSegmenterTorchModule(
            lstm_hidden_layer_size=lstm_hidden_layer_size,
            lstm_num_layers=lstm_num_layers,
            num_embeddings=self._tokenizer.vocab_size,
            pad_id=int(self._tokenizer.pad_token_id or 0),
            num_classes=self.NUM_CLASSES,
        )

        state_dict = torch.load(uri_model)

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self._model.load_state_dict(state_dict)
        self._model = self._model.to(device)

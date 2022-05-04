"""Base classes for segmenter models."""
import typing as t
import warnings

import regex
import transformers
import torch
import torch.nn
import numpy as np
import numpy.typing as npt
import tqdm.auto

from . import output_handlers
from . import input_handlers


class BaseSegmenter:
    """Base class for Segmenter models."""

    NUM_CLASSES = 4

    _RE_REPR_TOKENIZER_ADJUST_01 = regex.compile(r"(?<=[\(,]\s*)(?=[a-z_]+\s*=)")
    _RE_REPR_TOKENIZER_ADJUST_02 = regex.compile(r"(?<=[{,]\s*)(?=[a-z_']+\s*:)")

    def __init__(
        self,
        uri_tokenizer: t.Optional[str] = None,
        inference_pooling_operation: str = "assymetric-max",
        local_files_only: bool = True,
        device: str = "cpu",
        cache_dir_tokenizer: str = "./cache/tokenizers",
    ):
        self.local_files_only = bool(local_files_only)

        self._model: t.Union[torch.nn.Module, transformers.BertForTokenClassification]
        self._tokenizer: transformers.BertTokenizerFast

        if uri_tokenizer:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                uri_tokenizer,
                local_files_only=self.local_files_only,
                cache_dir=cache_dir_tokenizer,
            )

        self._moving_window_pooler = output_handlers.AutoMovingWindowPooler(
            pooling_operation=inference_pooling_operation,
        )

        self.device = device

    def __call__(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Union[t.List[str], t.Tuple[t.List[t.Any], ...]]:
        return self.segment_legal_text(*args, **kwargs)

    def __repr__(self) -> str:
        strs: t.List[str] = []

        strs.append(f"{self.__class__.__name__} pipeline")
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

    def eval(self) -> "BaseSegmenter":
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def train(self) -> "BaseSegmenter":
        """Set model to train mode."""
        self.model.train()
        return self

    @property
    def model(self) -> t.Union[torch.nn.Module, transformers.BertForTokenClassification]:
        # pylint: disable='missing-function-docstring'
        return self._model

    @property
    def tokenizer(self) -> transformers.BertTokenizerFast:
        # pylint: disable='missing-function-docstring'
        return self._tokenizer

    @property
    def RE_JUSTIFICATIVA(self) -> regex.Pattern:
        """Regular expression used to detect 'justificativa' blocks."""
        # pylint: disable='invalid-name'
        return input_handlers.InputHandlerString.RE_JUSTIFICATIVA

    @classmethod
    def preprocess_legal_text(
        cls,
        text: str,
        return_justificativa: bool = False,
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ) -> t.Union[str, t.Tuple[str, t.List[str]]]:
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

        justificativa_block : t.List[str]
            Detected legal text `justificativa` blocks.
            Only returned if `return_justificativa=True`.
        """
        ret = input_handlers.InputHandlerString.preprocess_legal_text(
            text=text,
            regex_justificativa=regex_justificativa,
        )

        if return_justificativa:
            return ret

        preprocessed_text, _ = ret

        return preprocessed_text

    def _generate_segments_from_labels(
        self,
        tokens: transformers.BatchEncoding,
        num_tokens: int,
        label_ids: npt.NDArray[np.int32],
    ) -> t.List[str]:
        """Convert predicted labels and subword tokens to text segments."""
        seg_cls_id: int

        try:
            seg_cls_id = self._model.config.label2id.get("SEG_START", 1)  # type: ignore

        except AttributeError:
            seg_cls_id = 1

        segment_start_inds = np.flatnonzero(label_ids == seg_cls_id)
        segment_start_inds = np.hstack((0, segment_start_inds, num_tokens))

        segs: t.List[str] = []
        np_token_ids: npt.NDArray[np.int32] = tokens["input_ids"].ravel()

        for i, i_next in zip(segment_start_inds[:-1], segment_start_inds[1:]):
            split_ = np_token_ids[i:i_next]
            seg = self._tokenizer.decode(split_, skip_special_tokens=True)
            if seg:
                segs.append(seg)

        return segs

    def _preprocess_minibatch(
        self, minibatch: transformers.BatchEncoding
    ) -> transformers.BatchEncoding:
        """Perform necessary minibatch transformations before inference.

        Can be used by subclasses. In this base class, this method is No-op/identity operator.
        """
        # pylint: disable='no-self-use'
        return minibatch

    def _predict_minibatch(self, minibatch: transformers.BatchEncoding) -> npt.NDArray[np.float64]:
        """Predict a tokenized minibatch."""
        model_out = self._model(**minibatch)
        model_out = model_out["logits"]
        model_out = model_out.cpu().numpy()

        logits: npt.NDArray[np.float64] = model_out.astype(np.float64, copy=False)

        return logits

    def segment_legal_text(
        self,
        text: t.Union[str, t.Dict[str, t.List[int]]],
        batch_size: int = 32,
        moving_window_size: int = 1024,
        window_shift_size: t.Union[float, int] = 0.5,
        return_justificativa: bool = False,
        return_labels: bool = False,
        return_logits: bool = False,
        remove_noise_subsegments: bool = False,
        show_progress_bar: bool = False,
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ) -> t.Union[t.List[str], t.Tuple[t.List[t.Any], ...]]:
        """Segment legal `text`.

        The pretrained model support texts up to 1024 subwords. Texts larger than this
        value are pre-segmented into 1024 subword blocks, and each block is feed to the
        segmenter individually.

        The block size can be configured to smaller (not larger) values using the
        `moving_window_size` from `BERTSegmenter.segment_legal_text` method during inference.

        Parameters
        ----------
        text : str or t.Dict[str, t.List[int]]
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

        remove_noise_subsegments : bool, default=False
            If True, remove all tokens between tokens classified as `noise_start` (inclusive) and
            `noise_end` or `segment` (either exclusive), whichever occurs first.

            - Tokens classified as `noise_end` are kept. In other words, they are the first
              non-noise token past the previous noise subsegment.
            - Tokens between `noise_start` and the sentence end are also removed.
            - Tokens between the sentence end and `noise_end` are kept.
            - Only the closest `noise_start` for every `noise_end` (or the sentence end) are
              considered. In other words, redundant `noise_start` tokens are ignored.

        show_progress_bar : bool, default=False
            If True, show segmentation progress bar.

        regex_justificativa : str, regex.Pattern or None, default=None
            Regular expression specifying how the `justificativa` portion from legal
            documents should be detected. If None, will use the pattern predefined in
            `Segmenter.RE_JUSTIFICATIVA` class attribute.

        Returns
        -------
        segments : t.List[str]
            Segmented legal text.

        justificativa : t.List[str]
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
            raise ValueError(f"'batch_size' parameter must be >= 1 (got '{batch_size}').")

        if moving_window_size < 1:
            raise ValueError(
                f"'moving_window_size' parameter must be >= 1 (got '{moving_window_size}')."
            )

        try:
            max_moving_window_size_allowed = int(
                self._model.config.max_position_embeddings  # type: ignore
            )

            if moving_window_size > max_moving_window_size_allowed:
                warnings.warn(
                    message=(
                        "'moving_window_size' is larger than model's positional embeddings "
                        f"(moving_window_size={moving_window_size}, "
                        f"max_moving_window_size_allowed={max_moving_window_size_allowed}). "
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
                f"'window_shift_size' parameter must be >= 1 (got '{window_shift_size}')."
            )

        if window_shift_size > moving_window_size:
            warnings.warn(
                message=(
                    f"'window_shift_size' parameter must be <= {moving_window_size} "
                    f"(got '{window_shift_size}'). "
                    f"Will set it to {moving_window_size} automatically."
                ),
                category=UserWarning,
            )
            window_shift_size = moving_window_size

        tokens, justificativa, num_tokens = input_handlers.tokenize_input(
            text=text,
            tokenizer=self.tokenizer,
            regex_justificativa=regex_justificativa,
        )

        minibatches = input_handlers.build_minibatches(
            tokens=tokens,
            num_tokens=num_tokens,
            batch_size=batch_size,
            moving_window_size=moving_window_size,
            window_shift_size=int(window_shift_size),
            pad_id=int(self._tokenizer.pad_token_id or 0),
        )

        self.eval()
        all_logits: t.List[npt.NDArray[np.float64]] = []

        with torch.no_grad():
            for minibatch in tqdm.auto.tqdm(minibatches, disable=not show_progress_bar):
                minibatch = self._preprocess_minibatch(minibatch)
                minibatch = minibatch.to(self.device)
                model_out = self._predict_minibatch(minibatch)
                all_logits.append(model_out)

        logits = np.vstack(all_logits)
        del all_logits

        logits = self._moving_window_pooler(
            logits=logits,
            window_shift_size=window_shift_size,
        )

        label_ids = logits.argmax(axis=-1)
        label_ids = label_ids.squeeze()

        tokens = transformers.BatchEncoding(
            {key: val.cpu().detach().numpy() for key, val in tokens.items()}
        )

        label2id: t.Dict[str, int]

        if remove_noise_subsegments:
            try:
                label2id = self._model.config.label2id  # type: ignore

            except AttributeError:
                label2id = dict(
                    seg_cls_id=1,
                    noise_start_cls_id=2,
                    noise_end_cls_id=3,
                )

            label_ids, (logits, *tokens_vals) = output_handlers.remove_noise_subsegments(
                label_ids,
                logits,
                *tokens.values(),
                label2id=label2id,
            )

            for key, val in zip(tokens.keys(), tokens_vals):
                tokens[key] = val

        segs = self._generate_segments_from_labels(
            tokens=tokens,
            num_tokens=num_tokens,
            label_ids=label_ids,
        )

        label_ids = label_ids[:num_tokens]
        logits = logits.reshape(-1, self.NUM_CLASSES)
        logits = logits[:num_tokens, :]

        assert label_ids.size == logits.shape[0]

        ret = output_handlers.pack_results(
            keys=["segments", "justificativa", "labels", "logits"],
            vals=[segs, justificativa, label_ids, logits],
            inclusion=[True, return_justificativa, return_labels, return_logits],
        )

        return ret


class LSTMSegmenterTorchModule(torch.nn.Module):
    """Bidirecional LSTM Torch model for legal document segmentation."""

    def __init__(
        self,
        lstm_hidden_layer_size: int,
        lstm_num_layers: int,
        num_embeddings: int,
        pad_id: int,
        num_classes: int,
        quantize: bool = False,
    ):
        super().__init__()

        fn_factory_emb = torch.nn.quantized.Embedding if quantize else torch.nn.Embedding
        fn_factory_lstm = torch.nn.quantized.dynamic.LSTM if quantize else torch.nn.LSTM
        fn_factory_linear = torch.nn.quantized.dynamic.Linear if quantize else torch.nn.Linear

        self.embeddings = fn_factory_emb(
            num_embeddings=num_embeddings,
            embedding_dim=768,
            padding_idx=pad_id,
        )

        self.lstm = fn_factory_lstm(
            input_size=768,
            hidden_size=lstm_hidden_layer_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.lin_out = fn_factory_linear(
            2 * lstm_hidden_layer_size,
            num_classes,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Dict[str, torch.Tensor]:
        # pylint: disable='missing-function-docstring', 'unused-argument'
        out = input_ids

        out = self.embeddings(out)
        out, *_ = self.lstm(out)
        out = self.lin_out(out)

        return {"logits": out}

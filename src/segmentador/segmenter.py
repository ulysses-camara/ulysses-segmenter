"""Legal text segmenter."""
import typing as t
import warnings

import regex
import transformers
import torch

from . import _base


class BERTSegmenter(_base.BaseSegmenter):
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
    ):
        super().__init__(
            uri_tokenizer=uri_tokenizer if uri_tokenizer is not None else uri_model,
            local_files_only=local_files_only,
            inference_pooling_operation=inference_pooling_operation,
            device=device,
            cache_dir_tokenizer=cache_dir_tokenizer,
        )

        if config is None:
            labels = ("NO-OP", "SEG_START", "NOISE_START", "NOISE_END")
            config = transformers.BertConfig.from_pretrained(uri_model)
            config.max_position_embeddings = 1024
            config.num_hidden_layers = num_hidden_layers
            config.num_labels = num_labels
            config.label2id = dict(zip(labels, range(num_labels)))
            config.id2label = dict(zip(range(num_labels), labels))

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


class LSTMSegmenter(_base.BaseSegmenter):
    """Bidirectional LSTM segmenter for PT-br legal text data.

    Uses a pretrained Bidirectional LSTM to segment Brazilian Portuguese legal texts.

    Parameters
    ----------
    uri_model : str
        URI to load pretrained model from.

    uri_tokenizer : str
        URI to pretrained text Tokenizer.

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

    quantize_weights : bool, default=False
        TODO.

    lstm_hidden_layer_size : int
        Dimension of LSTM model hidden layer.

    lstm_num_layers : int
        Number of layers in LSTM model.

    cache_dir_tokenizer : str, default="../cache/tokenizers"
        Cache directory for text tokenizer.
    """

    def __init__(
        self,
        uri_model: str,
        uri_tokenizer: str,
        inference_pooling_operation: t.Literal[
            "max", "sum", "gaussian", "assymetric-max"
        ] = "assymetric-max",
        local_files_only: bool = True,
        device: str = "cpu",
        quantize_weights: bool = False,
        lstm_hidden_layer_size: t.Optional[int] = None,
        lstm_num_layers: t.Optional[int] = None,
        cache_dir_tokenizer: str = "../cache/tokenizers",
    ):
        super().__init__(
            uri_tokenizer=uri_tokenizer,
            local_files_only=local_files_only,
            inference_pooling_operation=inference_pooling_operation,
            device=device,
            cache_dir_tokenizer=cache_dir_tokenizer,
        )

        state_dict = torch.load(uri_model, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if lstm_hidden_layer_size is None:
            _, doubled_hidden_size = state_dict["lin_out.weight"].shape
            lstm_hidden_layer_size = doubled_hidden_size // 2

        if lstm_num_layers is None:
            re_find_layer_inds = regex.compile(r"(?<=lstm\.weight.*_l)([0-9]+)")

            lstm_ind_matches = [re_find_layer_inds.search(key) for key in state_dict.keys()]
            all_layer_inds = {int(match.group(1)) for match in lstm_ind_matches if match}

            if not all_layer_inds:
                warnings.warn(
                    "Could not infer lstm number of layers (parameter 'lstm_num_layers')"
                    "From checkpoint file. Will set it to '1'."
                )
                all_layer_inds = {0}

            lstm_num_layers = 1 + max(all_layer_inds)

        self._model = _base.LSTMSegmenterTorchModule(
            lstm_hidden_layer_size=lstm_hidden_layer_size,
            lstm_num_layers=lstm_num_layers,
            num_embeddings=self._tokenizer.vocab_size,
            pad_id=int(self._tokenizer.pad_token_id or 0),
            num_classes=self.NUM_CLASSES,
            quantize=quantize_weights,
        )

        self._model.load_state_dict(state_dict)
        self._model = self._model.to(device)

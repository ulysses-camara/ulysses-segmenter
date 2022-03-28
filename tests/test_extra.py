"""Test extra features from the package."""
import segmentador


def test_segmenter_model_to_string(fixture_model_2_layers: segmentador.Segmenter):
    assert isinstance(str(fixture_model_2_layers), str)


def test_lstm_instantiation_infer_model_parameters():
    segmentador.LSTMSegmenter(
        uri_model="pretrained_segmenter_model/128_6000_2_lstm/checkpoints/epoch=3-step=3591.ckpt",
        uri_tokenizer="tokenizers/6000_subwords",
        device="cpu",
        local_files_only=True,
    )
    assert True

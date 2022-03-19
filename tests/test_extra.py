"""Test extra features from the package."""
import segmentador


def test_segmenter_model_to_string(fixture_model_2_layers: segmentador.Segmenter):
    assert isinstance(str(fixture_model_2_layers), str)

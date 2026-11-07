import pytest

from api.model_loader import ModelLoader


def test_missing_model_artifact_fails_closed(tmp_path):
    loader = ModelLoader(tmp_path / "missing.joblib", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        loader.load()
    assert not loader.is_ready()


def test_missing_model_metadata_fails_closed(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"not-loaded")
    loader = ModelLoader(model_path, tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="Model metadata not found"):
        loader.load()
    assert not loader.is_ready()
import pytest
from inference import initialize_faiss
from langchain.vectorstores import FAISS
import os

class Test_InferenceInitializeFaiss:
    @pytest.mark.positive
    def test_initialize_faiss_successful_load(self, monkeypatch):
        def mock_load_local(*args, **kwargs):
            return FAISS({})
        
        monkeypatch.setattr(FAISS, 'load_local', mock_load_local)

        result = initialize_faiss()
        assert result is not None
        assert isinstance(result, FAISS)

    @pytest.mark.negative
    def test_initialize_faiss_load_failure_index_missing(self, monkeypatch):
        def mock_load_local(*args, **kwargs):
            raise FileNotFoundError()
        
        monkeypatch.setattr(FAISS, 'load_local', mock_load_local)

        with pytest.raises(FileNotFoundError):
            result = initialize_faiss()
            assert result is None

    @pytest.mark.negative
    def test_initialize_faiss_load_failure_invalid_index(self, monkeypatch):
        def mock_load_local(*args, **kwargs):
            raise ValueError()
        
        monkeypatch.setattr(FAISS, 'load_local', mock_load_local)

        with pytest.raises(ValueError):
            result = initialize_faiss()
            assert result is None

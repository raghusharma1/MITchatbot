import pytest
import os
from unittest.mock import patch, MagicMock
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from inference import initialize_faiss

class Test_InferenceInitializeFaiss:

    @pytest.mark.valid
    def test_load_faiss_index_success(self, monkeypatch):
        mock_faiss = MagicMock(return_value=FAISS)        
        monkeypatch.setattr(FAISS, 'load_local', mock_faiss)

         # TODO Mock the actual value provided
        mock_path = MagicMock(return_value='path_to_input')
        monkeypatch.setattr(os.path, 'exists', mock_path)

        result = initialize_faiss()
        assert isinstance(result, FAISS)
        assert mock_faiss.called
        mock_path.assert_called_once_with('./faiss_db')

    @pytest.mark.negative
    def test_load_faiss_index_failure_no_index(self, monkeypatch):
        mock_faiss = MagicMock(side_effect=FileNotFoundError)
        monkeypatch.setattr(FAISS, 'load_local', mock_faiss)

         # TODO Mock the actual value provided
        mock_path = MagicMock(return_value='path_to_input')
        monkeypatch.setattr(os.path, 'exists', mock_path)

        result = initialize_faiss()
        assert result is None
        mock_path.assert_called_once_with('./faiss_db')

    @pytest.mark.negative
    def test_load_faiss_index_failure_incompatible_embeddings(self, monkeypatch):
        mock_faiss = MagicMock(side_effect=Exception)
        monkeypatch.setattr(FAISS, 'load_local', mock_faiss)

         # TODO Mock the actual value provided
        mock_path = MagicMock(return_value='path_to_input')
        monkeypatch.setattr(os.path, 'exists', mock_path)

        result = initialize_faiss()
        assert result is None
        mock_path.assert_called_once_with('./faiss_db')

    @pytest.mark.negative
    def test_load_faiss_index_failure_incorrect_settings(self, monkeypatch):     
        mock_faiss = MagicMock(side_effect=RuntimeError)
        monkeypatch.setattr(FAISS, 'load_local', mock_faiss)

         # TODO Mock the actual value provided
        mock_path = MagicMock(return_value='path_to_input')
        monkeypatch.setattr(os.path, 'exists', mock_path)

        result = initialize_faiss()
        assert result is None
        mock_path.assert_called_once_with('./faiss_db')

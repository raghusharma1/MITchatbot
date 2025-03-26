import pytest
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from inference import initialize_faiss, search_with_score
import gradio as gr
from main import chatbot

class Test_MainChatbot:

    @pytest.mark.valid
    def test_valid_query(self):
        db = initialize_faiss()
        query = "valid query"
        expected_result = "expected answer"
        assert chatbot(query, 0.5) == expected_result
    
    @pytest.mark.negative
    def test_faiss_index_loading_failure(self):
        db = None
        query = "valid query"
        expected_result = "Faiss index loading failed"
        assert chatbot(query, 0.5) == expected_result

    @pytest.mark.valid
    def test_high_score_failure(self):
        db = initialize_faiss()
        query = "query that yields high score"
        expected_result = "Search not conducted accurately"
        assert chatbot(query, 100) == expected_result

    @pytest.mark.valid
    def test_handling_multiple_documents(self):
        db = initialize_faiss()
        query = "query that returns multiple documents"
        expected_result = "Output that handles multiple documents appropriately"
        assert chatbot(query, 0.5) == expected_result

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import chatbot
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_main(query:str, score:float):
    return chatbot(query, score)

client = TestClient(app)

class Test_MainChatbot:

    @pytest.mark.invalid
    def test_fastapi_db_initialization(self):
        response = client.get("/", params={"query": "Hello", "score": 0.5})

        assert response.status_code == 200
        assert response.json() == "Faiss index loading failed"

    @pytest.mark.valid
    @patch('main.search_with_score')
    def test_chatbot_response(self, mock_search):

        expected_output = {'answer':'This is a chatbot answer'}
        mock_search.return_value = expected_output

        response = client.get("/", params={"query": "Hello", "score": 0.5})
        assert response.status_code == 200
        assert response.json() == expected_output

    @pytest.mark.valid
    @patch('main.search_with_score')
    def test_inference_search_accuracy(self, mock_search):

        expected_doc = [{'doc':'doc_1', 'score':1}, {'doc':'doc_2', 'score':0.8}, {'doc':'doc_3', 'score':0.6}]

        mock_search.return_value = expected_doc
        response = client.get("/", params={"query": "Hello", "score": 1})

        assert response.status_code == 200
        assert response.json() == expected_doc[0]['doc']

    @pytest.mark.valid
    @patch('main.search_with_score')
    @pytest.mark.capture_print
    def test_result_print(self, mock_search):

        expected_doc = [{'doc':'doc_1', 'score':1}, {'doc':'doc_2', 'score':0.8}, {'doc':'doc_3', 'score':0.6}]

        mock_search.return_value = expected_doc   
        response = client.get("/", params={"query": "Hello", "score": 1})

        captured = capsys.readouterr()  # Capture print outputs
        assert expected_doc in captured.out  # Check print statement is as expected

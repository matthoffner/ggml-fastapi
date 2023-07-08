from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient
from main import app # assuming your code is in a file called main.py

client = TestClient(app)

mocked_model_response = 'Hello world'

@pytest.fixture
def mock_llm(monkeypatch):
    mock_llm = Mock()
    mock_llm.tokenize.return_value = "mocked tokens"
    mock_llm.generate.return_value = iter([mocked_model_response])
    mock_llm.detokenize.return_value = mocked_model_response
    monkeypatch.setattr("main.llm", mock_llm)
    return mock_llm

def test_v1_completions(mock_llm):
    response = client.post("/v1/completions", json={"prompt": "Hello world"})
    assert response.status_code == 200
    # Assert that the llm __call__ method was used correctly
    mock_llm.__call__.assert_called_once_with("Hello world")

def test_v1_chat_completions(mock_llm):
    data = {
        "messages": [{"role": "system", "content": "Hello world"}],
        "max_tokens": 50
    }
    response = client.post("/v1/chat/completions", json=data)
    assert response.status_code == 200
    # Assert that the mock_llm's methods were called correctly
    mock_llm.tokenize.assert_called_once_with("Hello world")
    mock_llm.generate.assert_called_once_with("mocked tokens")

def test_v2_chat_completions(mock_llm):
    data = {
        "messages": [{"role": "system", "content": "Hello world"}],
        "max_tokens": 50
    }
    response = client.post("/v2/chat/completions", json=data)
    assert response.status_code == 200
    # Assert that the mock_llm's methods were called correctly
    mock_llm.tokenize.assert_called_once_with("Hello world")
    mock_llm.generate.assert_called_once_with("mocked tokens")

def test_v0_chat_completions(mock_llm):
    response = client.post("/v0/chat/completions", json={"prompt": "Hello world"})
    assert response.status_code == 200
    # Assert that the llm tokenize and generate methods were called correctly
    mock_llm.tokenize.assert_called_once_with("Hello world")
    mock_llm.generate.assert_called_once_with("mocked tokens")


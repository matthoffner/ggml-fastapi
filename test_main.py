from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient
from main import app, llm_wrapper

client = TestClient(app)

mocked_model_response = 'Hello world'

@pytest.fixture
def mock_model(monkeypatch):
    mock_model = Mock()
    mock_model.tokenize.return_value = "mocked tokens"
    mock_model.generate.return_value = iter([mocked_model_response])
    mock_model.detokenize.return_value = mocked_model_response

    def mock_load_model(*args, **kwargs):
        return mock_model

    monkeypatch.setattr(llm_wrapper, 'load_model', mock_load_model)
    return mock_model

def test_v1_completions(mock_model):
    response = client.post("/v1/completions", json={"prompt": "Hello world"})
    assert response.status_code == 200
    mock_model.tokenize.assert_called_once_with("Hello world")
    mock_model.generate.assert_called_once()

def test_v1_chat_completions(mock_model):
    data = {
        "messages": [{"role": "system", "content": "Hello world"}],
        "max_tokens": 50
    }
    response = client.post("/v1/chat/completions", json=data)
    assert response.status_code == 200
    mock_model.tokenize.assert_called_once_with("Hello world")
    mock_model.generate.assert_called_once()
    mock_model.detokenize.assert_called_once()


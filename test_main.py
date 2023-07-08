from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

mocked_model_response = 'Hello world'

@pytest.fixture
def mock_llm(monkeypatch):
    mock_llm = Mock()
    mock_llm.tokenize.return_value = "mocked tokens"
    mock_llm.generate.return_value = iter([mocked_model_response])
    mock_llm.detokenize.return_value = mocked_model_response

    def mock_from_pretrained(*args, **kwargs):
        return mock_llm

    monkeypatch.setattr("ctranformers.AutoModelForCausalLM.from_pretrained", mock_from_pretrained)
    monkeypatch.setattr("main.llm", mock_llm)
    return mock_llm

def test_v1_completions(mock_llm):
    response = client.post("/v1/completions", json={"prompt": "Hello world"})
    assert response.status_code == 200
    mock_llm.tokenize.assert_called_once_with("Hello world")
    mock_llm.generate.assert_called_once_with("mocked tokens")

def test_v1_chat_completions(mock_llm):
    data = {
        "messages": [{"role": "system", "content": "Hello world"}],
        "max_tokens": 50
    }
    response = client.post("/v1/chat/completions", json=data)
    assert response.status_code == 200
    mock_llm.tokenize.assert_called_once_with("Hello world")
    mock_llm.generate.assert_called_once_with("mocked tokens")
    mock_llm.detokenize.assert_called_once()

def test_v2_chat_completions(mock_llm):
    data = {
        "messages": [{"role": "system", "content": "Hello world"}],
        "max_tokens": 50
    }
    response = client.post("/v2/chat/completions", json=data)
    assert response.status_code == 200
    mock_llm.tokenize.assert_called_once_with("Hello world")
    mock_llm.generate.assert_called_once_with("mocked tokens")
    mock_llm.detokenize.assert_called_once()

def test_v0_chat_completions(mock_llm):
    response = client.post("/v0/chat/completions", json={"prompt": "Hello world"})
    assert response.status_code == 200
    mock_llm.tokenize.assert_called_once_with("Hello world")
    mock_llm.generate.assert_called_once_with("mocked tokens")
    mock_llm.detokenize.assert_called_once()

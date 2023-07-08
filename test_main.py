# test_main.py
from unittest.mock import Mock
from fastapi.testclient import TestClient
import pytest
from main import app, llm_wrapper

client = TestClient(app)

@pytest.fixture
def mock_model_fixture(monkeypatch):
    # This is our mock model.
    # It will be used in place of the actual model object.
    mock_model = Mock()
    mock_model.return_value = "mocked response"

    # We need to mock the get_model function to return our mock model.
    def mock_get_model():
        return mock_model

    monkeypatch.setattr(llm_wrapper, 'get_model', mock_get_model)

    yield mock_model  # we're yielding the mock model, so we can assert on it later

def test_v1_completions(mock_model_fixture):
    response = client.post("/v1/completions", json={"prompt": "Hello world"})
    assert response.status_code == 200
    assert response.json() == "mocked response"

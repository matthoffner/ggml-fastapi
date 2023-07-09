# ggml-fastapi

### [fastapi](https://fastapi.tiangolo.com/)
### [ggml](https://github.com/ggerganov/ggml)
### [ctransformers](https://github.com/marella/ctransformers)

## Models

* [x] Falcon
* [x] WizardCoder
* [x] MPT

## Setup

The server requires Python 3.7 or later and FastAPI. The easiest way to install FastAPI is with pip:

```sh
pip install fastapi uvicorn
```

In addition, the server requires the `ctransformers` package, which can be installed with pip:

```sh
pip install ctransformers
```

## Usage

Run the server with:

```sh
python main.py
```

The server will be available at `http://localhost:8000`.

## API Endpoints

### POST /v1/completions

Generates a text completion for a given prompt.

**Request Body:**

```json
{
  "prompt": "Once upon a time"
}
```

**Response:**

The response is a text string that continues the prompt.

### POST /v1/chat/completions

Generates a text completion for a chat conversation.

**Request Body:**

```json
{
  "messages": [
    {"role": "system", "content": "Once upon a time"},
    {"role": "user", "content": "Tell me more."}
  ],
  "max_tokens": 250
}
```

**Response:**

The response is a streaming response, where each chunk is a JSON object that represents a continuation of the chat conversation. The server will continue to generate text until it generates a message with the content "[DONE]".

### POST /v2/chat/completions

This endpoint is similar to `v1/chat/completions`, but it's designed to handle multiple concurrent chat completions. It uses Python's `concurrent.futures` package to process the chat completions in parallel.

**Request Body:**

```json
{
  "messages": [
    {"role": "system", "content": "Once upon a time"},
    {"role": "user", "content": "Tell me more."}
  ],
  "max_tokens": 250
}
```

**Response:**

The response is a streaming response, where each chunk is a JSON object that represents a continuation of the chat conversation. The server will continue to generate text until it generates a message with the content "[DONE]".

## Customization

You can customize the model that the server uses by setting the `MODEL_NAME`, `MODEL_FILE`, and `MODEL_TYPE` environment variables before running the server.

For example, to use a custom model, you can run the server with:

```sh
MODEL_NAME=MyModel MODEL_FILE=my_model.bin MODEL_TYPE=my_model_type python main.py
```


## Examples

* Wizardcoder https://huggingface.co/spaces/matthoffner/wizardcoder-ggml
* Falcon https://huggingface.co/spaces/matthoffner/falcon-mini/blob/main/api.py
* Starchat https://huggingface.co/spaces/matthoffner/starchat-ggml

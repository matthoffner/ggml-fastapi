# ggml-fastapi

### [fastapi](https://fastapi.tiangolo.com/)
### [ggml](https://github.com/ggerganov/ggml)
### [ctransformers](https://github.com/marella/ctransformers)

## Models

* [x] Falcon
* [x] WizardCoder
* [x] MPT

## Setup

```sh
pip install -r requirements.txt
```

## Usage

You can start the FastAPI server with the following commands, depending on the model type you want to use. You need to replace the placeholders (`your_model_name`, `path_to_your_model_file`) with your actual values.

### Using `starcoder` model type

```shell
MODEL_NAME='TheBloke/WizardCoder-15B-1.0-GGML' MODEL_FILE='WizardCoder-15B-1.0.ggmlv3.q5_0.bin' MODEL_TYPE='starcoder' uvicorn main:app
```

### Using `falcon` model type

```shell
MODEL_NAME='TheBloke/falcon-40b-instruct-GGML' MODEL_FILE='falcon40b-instruct.ggmlv3.q2_K.bin' MODEL_TYPE='falcon' uvicorn main:app
```

### Using `mpt` model type

```shell
MODEL_NAME='TheBloke/mpt-30B-instruct-GGML' MODEL_FILE='mpt-30b-instruct.ggmlv0.q4_0.bin' MODEL_TYPE='mpt' uvicorn main:app
```

This will start the FastAPI application with the specified model. Note that these environment variables are only set for this specific command and won't be available in other shell sessions or scripts. If you need these environment variables for the whole system or across multiple sessions, consider setting them in your shell's configuration file, using Docker or a similar tool, or using a secrets manager.


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

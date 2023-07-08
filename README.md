# ggml-fastapi

FastAPI wrapper using [ctransformers](https://github.com/marella/ctransformers)

* [x] Falcon
* [x] WizardCoder
* [x] MPT

## Endpoints

2. **Completion Endpoint**
   - Path: `/v1/completions`
   - Method: `POST`
   - Summary: Completion
   - Parameters: `response_mode` (optional)
   - Request Body: `ChatCompletionRequestV0` schema
   - Responses: 200 (Successful Response), 422 (Validation Error)

3. **Chat Endpoint (v1)**
   - Path: `/v1/chat/completions`
   - Method: `POST`
   - Summary: Chat
   - Request Body: `ChatCompletionRequest` schema
   - Responses: 200 (Successful Response), 422 (Validation Error)

## Schema Definitions
1. **ChatCompletionRequest**: Requires `messages` (array of `Message` schema). Optional `max_tokens` (integer, default 250).
2. **ChatCompletionRequestV0**: Requires `prompt` (string).
3. **Message**: Requires `role` (string), `content` (string).
4. **ValidationError**: Requires `loc` (array of string or integer), `msg` (string), `type` (string).
5. **HTTPValidationError**: Contains `detail` (array of `ValidationError` schema).


## Examples

* Wizardcoder https://huggingface.co/spaces/matthoffner/wizardcoder-ggml
* Falcon https://huggingface.co/spaces/matthoffner/falcon-mini/blob/main/api.py
* Starchat https://huggingface.co/spaces/matthoffner/starchat-ggml

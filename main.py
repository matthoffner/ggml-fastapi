import os
import json
import fastapi
import uvicorn
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ctransformers import AutoModelForCausalLM
from pydantic import BaseModel

DEFAULT_MODEL_NAME = "TheBloke/WizardCoder-15B-1.0-GGML"
DEFAULT_MODEL_FILE = "WizardCoder-15B-1.0.ggmlv3.q4_0.bin"
DEFAULT_MODEL_TYPE = "starcoder"
MODEL_NAME = os.getenv('MODEL_NAME', DEFAULT_MODEL_NAME)
MODEL_FILE = os.getenv('MODEL_FILE', DEFAULT_MODEL_FILE)
MODEL_TYPE = os.getenv('MODEL_TYPE', DEFAULT_MODEL_TYPE)

class ModelWrapper:
    def __init__(self, model_name, model_file, model_type):
        self.model_name = model_name
        self.model_file = model_file
        self.model_type = model_type
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                model_file=self.model_file,
                model_type=self.model_type,
            )
        return self._model

llm_wrapper = ModelWrapper(MODEL_NAME, MODEL_FILE, MODEL_TYPE)

app = fastapi.FastAPI(title=f"{MODEL_NAME}-fastapi")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    html_content = """
    <html>
        <head>
          <title>ggml-fastapi</title>
        </head>
        <body style="background-color:black;font-family:system-ui">
            <h2>ggml-fastapi</h2>
            <h2><a href="/docs">FastAPI Docs</a></h2>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

class ChatCompletionRequestV0(BaseModel):
    prompt: str

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 250

@app.post("/v1/completions")
async def completion(request: ChatCompletionRequestV0, response_mode=None):
    llm = llm_wrapper.model
    response = llm(request.prompt)
    return response

@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    combined_messages = ' '.join([message.content for message in request.messages])
    llm = llm_wrapper.model
    tokens = llm.tokenize(combined_messages)

    try:
        chat_chunks = llm.generate(tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def format_response(chat_chunks: Generator, llm) -> Any:
        for chat_chunk in chat_chunks:
            response = {
                'choices': [
                    {
                        'message': {
                            'role': 'system',
                            'content': llm.detokenize(chat_chunk)
                        },
                        'finish_reason': 'stop' if llm.detokenize(chat_chunk) == "[DONE]" else 'unknown'
                    }
                ]
            }
            yield f"data: {json.dumps(response)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(format_response(chat_chunks, llm), media_type="text/event-stream")

@app.post("/v0/chat/completions")
async def chat(request: ChatCompletionRequestV0, response_mode=None):
    llm = llm_wrapper.model
    tokens = llm.tokenize(request.prompt)
    async def server_sent_events(chat_chunks, llm):
        for chat_chunk in llm.generate(chat_chunks):
            yield dict(data=json.dumps(llm.detokenize(chat_chunk)))
        yield dict(data="[DONE]")

    return EventSourceResponse(server_sent_events(tokens, llm))

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)

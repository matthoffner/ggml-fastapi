import json
import markdown
from typing import Callable, List, Dict, Any, Generator
from functools import partial

import fastapi
import uvicorn
from fastapi import HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from anyio import create_memory_object_stream
from anyio.to_thread import run_sync
from ctransformers import AutoModelForCausalLM
from pydantic import BaseModel

DEFAULT_MODEL_NAME = "TheBloke/WizardCoder-15B-1.0-GGML"
DEFAULT_MODEL_FILE = "WizardCoder-15B-1.0.ggmlv3.q4_0.bin"
DEFAULT_MODEL_TYPE = "starcoder"
MODEL_NAME = os.getenv('MODEL_NAME', DEFAULT_MODEL_NAME)
MODEL_FILE = os.getenv('MODEL_FILE', DEFAULT_MODEL_FILE)
MODEL_TYPE = os.getenv('MODEL_TYPE', DEFAULT_MODEL_TYPE)

llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                           model_file=MODEL_FILE,
                                           model_type=MODEL_TYPE)

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
        </head>
        <body style="background-color:black">
            <h2 style="font-family:system-ui"><a href="https://huggingface.co/TheBloke/WizardCoder-15B-1.0-GGML">ggml-fastapi</a></h2>
            <h2 style="font-family:system-ui"><a href="https://matthoffner-wizardcoder-ggml.hf.space/docs">FastAPI Docs</a></h2>
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
    response = llm(request.prompt)
    return response

@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    combined_messages = ' '.join([message.content for message in request.messages])
    tokens = llm.tokenize(combined_messages)
    
    try:
        chat_chunks = llm.generate(tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def format_response(chat_chunks: Generator) -> Any:
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

    return StreamingResponse(format_response(chat_chunks), media_type="text/event-stream")

async def stream_response(tokens: Any) -> None:
    try:
        iterator: Generator = llm.generate(tokens)
        for chat_chunk in iterator:
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
        yield b"event: done\ndata: {}\n\n"
    except Exception as e:
        print(f"Exception in event publisher: {str(e)}")


async def chatV2(request: Request, body: ChatCompletionRequest):
    combined_messages = ' '.join([message.content for message in body.messages])
    tokens = llm.tokenize(combined_messages)

    return StreamingResponse(stream_response(tokens))

@app.post("/v2/chat/completions")
async def chatV2_endpoint(request: Request, body: ChatCompletionRequest):
    return await chatV2(request, body)

@app.post("/v0/chat/completions")
async def chat(request: ChatCompletionRequestV0, response_mode=None):
    tokens = llm.tokenize(request.prompt)
    async def server_sent_events(chat_chunks, llm):
        for chat_chunk in llm.generate(chat_chunks):
            yield dict(data=json.dumps(llm.detokenize(chat_chunk)))
        yield dict(data="[DONE]")

    return EventSourceResponse(server_sent_events(tokens, llm))

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)

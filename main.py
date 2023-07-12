import os
import json
import fastapi
import uvicorn
import concurrent.futures
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ctransformers import AutoModelForCausalLM
from pydantic import BaseModel
from typing import List
import logging
import asyncio

DEFAULT_MODEL_NAME = ""
DEFAULT_MODEL_FILE = ""
DEFAULT_MODEL_TYPE = ""
MODEL_NAME = os.getenv('MODEL_NAME', DEFAULT_MODEL_NAME)
MODEL_FILE = os.getenv('MODEL_FILE', DEFAULT_MODEL_FILE)
MODEL_TYPE = os.getenv('MODEL_TYPE', DEFAULT_MODEL_TYPE)

class ModelWrapper:
    def __init__(self, model_name, model_file, model_type):
        self.model_name = model_name
        self.model_file = model_file
        self.model_type = model_type
        self._model = None

    def get_model(self):
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
    llm = llm_wrapper.get_model()
    response = llm(request.prompt)
    return response

async def generate_response(chat_chunks, llm):
    for chat_chunk in chat_chunks:
        response = {
            'choices': [
                {
                    'message': {
                        'role': 'system',
                        'content': llm.detokenize(chat_chunk)
                    },
                    'finish_reason': 'stop' if llm.is_eos_token(chat_chunk) else 'unknown'
                }
            ]
        }
        yield f"data: {json.dumps(response)}\n\n"
    yield "event: done\ndata: {}\n\n"

@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    combined_messages = ' '.join([message.content for message in request.messages])
    llm = llm_wrapper.get_model()
    tokens = llm.tokenize(combined_messages)
    
    try:
        chat_chunks = llm.generate(tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(generate_response(chat_chunks, llm), media_type="text/event-stream")

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.error(f"An error occurred: {exc}")
    return JSONResponse(status_code=500, content={"message": "An error occurred."})

def generate_chat_chunk(combined_messages):
    llm = llm_wrapper.get_model()
    tokens = llm.tokenize(combined_messages)
    try:
        chat_chunks = llm.generate(tokens)
        logging.info(f"Generated chat chunks: {chat_chunks}")
        return chat_chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v2/chat/completions")
async def chatV2(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    combined_messages = ' '.join([message.content for message in request.messages])
    background_tasks.add_task(generate_chat_chunk, combined_messages)
    return {"detail": "Chat generation started"}
        
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)

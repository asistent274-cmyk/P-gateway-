"""
main.py
-------
Minimaler Web-Service, der ai_router.py benutzt.

Startet einen FastAPI-Server mit einem einzigen Endpoint:
    POST /chat
    Body: {"message": "Deine Frage hier"}
    Antwort: {"provider": "gemini" | "groq" | "openrouter", "text": "..."}

Start lokal:
    uvicorn main:app --reload

Start auf Render (siehe SETUP.md):
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai_router import get_ai_response, AllProvidersFailedError

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"status": "ok", "info": "Sende POST /chat mit {'message': '...'} "}


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "user", "content": req.message}]
    try:
        result = await get_ai_response(messages)
    except AllProvidersFailedError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return result

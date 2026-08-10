"""
main.py
-------
Minimaler Web-Service, der ai_router.py benutzt.

Endpunkte:
    GET  /       -> zeigt die Chat-Oberfläche (gateway_interface.html)
    POST /chat   -> Body: {"messages": [{"role": "user", "content": "..."}]}
                    Antwort: {"provider": "gemini" | "groq" | "openrouter", "text": "..."}

Start lokal:
    uvicorn main:app --reload

Start auf Render (siehe SETUP.md):
    uvicorn main:app --host 0.0.0.0 --port $PORT

WICHTIG: Diese Datei muss im selben Ordner liegen wie "gateway_interface.html",
damit die Root-Route ("/") sie finden und anzeigen kann.
"""

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ai_router import get_ai_response, AllProvidersFailedError

app = FastAPI()

# CORS: erlaubt Zugriff aus dem Browser (z.B. von der gateway_interface.html aus).
# "*" erlaubt alle Ursprünge -- fuer ein rein privates Projekt ok, kann man
# spaeter auf eine bestimmte Domain einschraenken.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


@app.get("/")
async def root():
    return FileResponse("gateway_interface.html")


@app.get("/status")
async def status():
    return {"status": "ok", "info": "Sende POST /chat mit {'messages': [...]}"}


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        result = await get_ai_response(messages)
    except AllProvidersFailedError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return result

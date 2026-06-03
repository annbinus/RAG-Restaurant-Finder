from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from search import search
import requests, json

app = FastAPI(title="Yelp RAG Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

class SearchRequest(BaseModel):
    query: str
    city: str = ""
    min_stars: float = None
    top_k: int = 8

def generate_answer(query: str, restaurants: list[dict]) -> str:
    context = "\n\n".join([
        f"- {r['doc']}" for r in restaurants
    ])

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a restaurant recommendation assistant. "
                    "Answer the user's query using only the restaurants listed below. "
                    "Be concise, mention 2-3 standouts, and explain why they fit. "
                    "Do not make up restaurants or details not in the list."
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nRestaurants:\n{context}"
            }
        ],
        "stream": False,
        "options": {"temperature": 0.5}
    }

    res = requests.post(OLLAMA_URL, json=payload)
    res.raise_for_status()

    # Ollama sometimes returns newline-delimited JSON chunks — take the last valid one
    lines = [l for l in res.text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        try:
            data = json.loads(line)
            if "message" in data:
                return data["message"]["content"]
        except json.JSONDecodeError:
            continue

    return "Could not generate a response."

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/search")
async def search_endpoint(req: SearchRequest):
    results = search(req.query, city=req.city, min_stars=req.min_stars, top_k=req.top_k)
    answer = generate_answer(req.query, results)
    return {"query": req.query, "answer": answer, "results": results}
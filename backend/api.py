from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from search import search

app = FastAPI(title="Yelp Geo RAG Search API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    city: str = ""
    min_stars: float = None
    top_k: int = 8

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/search")
async def search_endpoint(req: SearchRequest):
    results = search(req.query, city=req.city, min_stars=req.min_stars, top_k=req.top_k)
    return {"query": req.query, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "yelp-geo-rag")
TOP_K = 8

def embed_query(text):
    return model.encode(text).tolist()


def enrich_query(query, city=""):
    if city:
        return f"{query} near {city}"
    return query


def search(query, city="", min_stars=None, top_k=TOP_K):
    enriched = enrich_query(query, city)
    vector = embed_query(enriched)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(PINECONE_INDEX)

    filter_dict = None
    if min_stars:
        filter_dict = {"stars": {"$gte": min_stars}}

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )

    return [
        {
            "name": m.metadata.get("name"),
            "city": m.metadata.get("city"),
            "state": m.metadata.get("state"),
            "stars": m.metadata.get("stars"),
            "review_count": m.metadata.get("review_count"),
            "categories": m.metadata.get("categories"),
            "score": round(m.score, 4),
        }
        for m in results.matches
    ]


if __name__ == "__main__":
    results = search("great Thai food", city="Philadelphia")
    for r in results:
        print(f"{r['name']} | {r['stars']}★ | {r['city']} | score: {r['score']}")
        print(f"  {r['categories']}")
        print()
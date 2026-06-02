import os
import time
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from enrich import stream_businesses, is_restaurant, build_enriched_doc

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"  # 90MB, 384-dim, fast and free
EMBED_DIM = 384
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "yelp-geo-rag")
EMBED_BATCH = 256  # local model can handle large batches
UPSERT_BATCH = 256

model = SentenceTransformer(EMBED_MODEL)

def get_embeddings(texts):
    return model.encode(texts, show_progress_bar=False).tolist()

def get_existing_ids(index, ids):
    existing = set()
    for i in range(0, len(ids), 100):
        batch = ids[i:i+100]
        resp = index.fetch(ids=batch)
        existing.update(resp.vectors.keys())
    return existing

def create_index(pc):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        print(f"Creating index '{PINECONE_INDEX}'...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(1)
        print("Index ready.")
    else:
        print(f"Index '{PINECONE_INDEX}' already exists.")

def run(yelp_path, limit=None):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    create_index(pc)
    index = pc.Index(PINECONE_INDEX)

    print("Loading restaurants...")
    records = []
    for biz in stream_businesses(yelp_path):
        if not is_restaurant(biz):
            continue
        doc = build_enriched_doc(biz)
        records.append({
            "id": biz["business_id"],
            "doc": doc,
            "metadata": {
                "name": biz.get("name", ""),
                "city": biz.get("city", ""),
                "state": biz.get("state", ""),
                "lat": float(biz.get("latitude") or 0),
                "lng": float(biz.get("longitude") or 0),
                "stars": float(biz.get("stars") or 0),
                "review_count": int(biz.get("review_count") or 0),
                "categories": biz.get("categories") or "",
                "is_open": int(biz.get("is_open") or 0),
                "doc": doc,
            }
        })
        if limit and len(records) >= limit:
            break

    print(f"{len(records):,} restaurants loaded.")

    all_ids = [r["id"] for r in records]
    print("Checking existing vectors...")
    existing_ids = get_existing_ids(index, all_ids)
    records = [r for r in records if r["id"] not in existing_ids]
    print(f"{len(records):,} new to embed, {len(existing_ids):,} already indexed.")

    vectors = []
    docs = [r["doc"] for r in records]

    print("Embedding...")
    for i in tqdm(range(0, len(docs), EMBED_BATCH)):
        batch_docs = docs[i:i+EMBED_BATCH]
        batch_records = records[i:i+EMBED_BATCH]
        embeddings = get_embeddings(batch_docs)

        for record, emb in zip(batch_records, embeddings):
            vectors.append({
                "id": record["id"],
                "values": emb,
                "metadata": record["metadata"],
            })

        if len(vectors) >= UPSERT_BATCH:
            index.upsert(vectors=vectors)
            vectors = []

    if vectors:
        index.upsert(vectors=vectors)

    stats = index.describe_index_stats()
    print(f"Done. {stats.total_vector_count:,} vectors in Pinecone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yelp-path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.yelp_path, args.limit)
"""
metrics.py

Retrieval evaluation: NDCG@K, MRR, Hit@K.

Compares three retrieval strategies:

1. naive     — raw query embedding
2. enriched  — query enriched with location + quality hints
3. hybrid    — enriched + Pinecone metadata filter (stars >= 4.0)

Used for benchmarking retrieval improvements.
"""

import os, sys
import math
import json
from dataclasses import dataclass, field
from typing import Callable
from dotenv import load_dotenv
from pinecone import Pinecone
from pathlib import Path
from sentence_transformers import SentenceTransformer
import math


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from enrich import enrich_query

load_dotenv()

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "yelp-geo-rag")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))
    return R * c

def geo_score(user_lat, user_lng, doc_lat, doc_lng):
    if None in (user_lat, user_lng, doc_lat, doc_lng):
        return 0.0

    distance = haversine_km(user_lat, user_lng, doc_lat, doc_lng)

    # decay function (smooth + stable)
    return 1 / (1 + distance)

def dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    gains = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in top_k]
    ideal = sorted(gains, reverse=True)

    actual_dcg = dcg(gains)
    ideal_dcg = dcg(ideal)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return float(any(doc_id in relevant_ids for doc_id in retrieved_ids[:k]))


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    query: str
    relevant_ids: set[str]
    user_lat: float | None = None
    user_lng: float | None = None
    city: str = ""
    description: str = ""


@dataclass
class EvalResult:
    strategy: str
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    hit_at_5: float = 0.0
    mrr_score: float = 0.0
    n_queries: int = 0
    failures: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.strategy:12s}] "
            f"NDCG@5={self.ndcg_at_5:.3f}  "
            f"NDCG@10={self.ndcg_at_10:.3f}  "
            f"Hit@5={self.hit_at_5:.3f}  "
            f"MRR={self.mrr_score:.3f}  "
            f"(n={self.n_queries})"
        )

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(PINECONE_INDEX)

    def embed(self, text: str) -> list[float]:
        return self.embedder.encode(text).tolist()

    def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filter_dict: dict | None = None
    ) -> list[str]:

        kwargs = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True
        }

        if filter_dict:
            kwargs["filter"] = filter_dict

        results = self.index.query(**kwargs)
        return [
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata
            }
            for m in results.matches
        ]

    # -------------------------
    # Strategy 1: naive
    # -------------------------
    def naive(self, test: TestCase, top_k: int = 10):
        vec = self.embed(test.query)
        results = self.search(vec, top_k)

        return [r["id"] for r in results]

    # -------------------------
    # Strategy 2: enriched
    # -------------------------
    def enriched(self, test: TestCase, top_k: int = 10):
        q = enrich_query(test.query, test.user_lat, test.user_lng, test.city)
        vec = self.embed(q)

        results = self.search(vec, top_k)

        return self.geo_rerank(results, test.user_lat, test.user_lng)

    # -------------------------
    # Strategy 3: hybrid
    # -------------------------
    def hybrid(
        self,
        test: TestCase,
        top_k: int = 10,
        min_stars: float = 4.0
    ) -> list[str]:

        enriched_q = enrich_query(test.query, test.user_lat, test.user_lng, test.city)
        vec = self.embed(enriched_q)

        results = self.search(vec, top_k, filter_dict={"stars": {"$gte": 4.0}})
        return self.geo_rerank(results, test.user_lat, test.user_lng)
    
    def geo_rerank(self, results, user_lat, user_lng):
        reranked = []

        for r in results:
            meta = r.get("metadata") or {}

            doc_lat = meta.get("lat")
            doc_lng = meta.get("lng")

            geo = geo_score(user_lat, user_lng, doc_lat, doc_lng)

            final_score = (0.8 * r["score"]) + (0.2 * geo)

            reranked.append({
                "id": r["id"],
                "score": final_score
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return [r["id"] for r in reranked]

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def evaluate(
        self,
        test_cases: list[TestCase],
        strategy_fn: Callable,
        strategy_name: str,
        k_values: tuple[int, int] = (5, 10)
    ) -> EvalResult:

        result = EvalResult(strategy=strategy_name, n_queries=len(test_cases))
        k1, k2 = k_values

        for tc in test_cases:
            retrieved = strategy_fn(tc, top_k=max(k_values))

            result.ndcg_at_5 += ndcg_at_k(retrieved, tc.relevant_ids, k1)
            result.ndcg_at_10 += ndcg_at_k(retrieved, tc.relevant_ids, k2)
            result.hit_at_5 += hit_at_k(retrieved, tc.relevant_ids, k1)
            result.mrr_score += mrr(retrieved, tc.relevant_ids)

            if not hit_at_k(retrieved, tc.relevant_ids, k1):
                result.failures.append({
                    "query": tc.query,
                    "description": tc.description,
                    "top_5_retrieved": retrieved[:5],
                    "expected_any_of": list(tc.relevant_ids)[:3],
                })

        n = result.n_queries
        result.ndcg_at_5 /= n
        result.ndcg_at_10 /= n
        result.hit_at_5 /= n
        result.mrr_score /= n

        return result

    def run_comparison(self, test_cases: list[TestCase]) -> list[EvalResult]:
        r = self.retriever

        results = [
            self.evaluate(test_cases, r.naive, "naive"),
            self.evaluate(test_cases, r.enriched, "enriched"),
            self.evaluate(test_cases, r.hybrid, "hybrid"),
        ]

        print("\n=== Retrieval Evaluation Results ===")
        for res in results:
            print(res.summary())

        naive, enriched = results[0], results[1]
        pct = (enriched.ndcg_at_10 - naive.ndcg_at_10) / max(naive.ndcg_at_10, 1e-9) * 100

        print(f"\nEnriched vs Naive NDCG@10 improvement: {pct:+.1f}%")

        print("\nTop failure patterns (enriched strategy):")
        for f in results[1].failures[:5]:
            print(f"  Query: '{f['query']}' | {f['description']}")

        return results

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: list[EvalResult], path: str = "eval_results.json") -> None:
    data = [
        {
            "strategy": r.strategy,
            "ndcg_at_5": r.ndcg_at_5,
            "ndcg_at_10": r.ndcg_at_10,
            "hit_at_5": r.hit_at_5,
            "mrr": r.mrr_score,
            "n_queries": r.n_queries,
            "n_failures": len(r.failures),
        }
        for r in results
    ]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {path}")


if __name__ == "__main__":
    import json

    # 1. load test set
    with open("eval_testset.json") as f:
        raw = json.load(f)

    # 2. convert into TestCase objects
    test_cases = [
        TestCase(
            query=item["query"],
            relevant_ids={item["relevant_id"]},
            city=item.get("city", ""),
            user_lat=item.get("lat"),
            user_lng=item.get("lng"),
            description=item.get("restaurant_name", "")
        )
        for item in raw
    ]

    # 3. run evaluation
    retriever = Retriever()
    evaluator = Evaluator(retriever)

    results = evaluator.run_comparison(test_cases)

    # 4. save results
    save_results(results)

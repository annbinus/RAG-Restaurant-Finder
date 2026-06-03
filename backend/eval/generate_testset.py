"""
generate_testset.py (LOCAL VERSION)

Generates a labeled evaluation set using a local Ollama model.
No API keys required.
"""

import os
import json
import random
import argparse
import requests
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from enrich import load_and_enrich

SYSTEM_PROMPT = """You are generating test queries for a restaurant search system.
Given a restaurant description, generate exactly 3 short, natural user queries
that should retrieve this restaurant. Vary them: one cuisine-focused, one
vibe/atmosphere-focused, one specific feature-focused.

Return ONLY a JSON array of 3 strings. No explanation.
Example:
["great cheap sushi near downtown", "casual Japanese spot for lunch", "sushi place with outdoor seating"]"""


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

def generate_queries_for_restaurant(doc: str) -> list[str]:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\nRestaurant: {doc}"
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    content = response.json()["message"]["content"].strip()

    # clean markdown fences if model adds them
    content = content.strip("```json").strip("```").strip()

    return json.loads(content)


def run(yelp_path: str, n_samples: int, output_path: str) -> None:

    print(f"Loading restaurants from {yelp_path}...")
    all_records = load_and_enrich(yelp_path, limit=10_000)

    candidates = [
        r for r in all_records
        if r["metadata"]["stars"] >= 4.0 and r["metadata"]["review_count"] >= 50
    ]

    sampled = random.sample(candidates, min(n_samples, len(candidates)))
    print(f"Sampled {len(sampled)} restaurants.")

    test_cases = []
    failed = 0

    for record in tqdm(sampled):
        try:
            queries = generate_queries_for_restaurant(record["doc"])

            for q in queries:
                test_cases.append({
                    "query": q,
                    "relevant_id": record["id"],
                    "restaurant_name": record["metadata"]["name"],
                    "city": record["metadata"]["city"],
                    "lat": record["metadata"]["lat"],
                    "lng": record["metadata"]["lng"],
                    "stars": record["metadata"]["stars"],
                    "categories": record["metadata"]["categories"],
                    "description": f"{record['metadata']['name']} ({record['metadata']['city']})",
                })

        except Exception as e:
            print(f"  Failed for {record['metadata']['name']}: {e}")
            failed += 1

    print(f"\nGenerated {len(test_cases)} test cases ({failed} failed).")

    with open(output_path, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"Saved to {output_path}")


def load_testset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yelp-path", required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", default="eval_testset.json")

    args = parser.parse_args()
    run(args.yelp_path, args.n_samples, args.output)
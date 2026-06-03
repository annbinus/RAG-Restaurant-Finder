"""
enrich.py

Transforms raw Yelp business records into enriched document strings
that bake geo, rating, price, and category signals into text so the
embedding model encodes them semantically — not just as metadata.

Key idea: the same embedding model sees both the enriched doc and the
enriched query at search time. Distance, quality, and vibe all become
part of the vector geometry.
"""

import json
import ijson
from pathlib import Path
from typing import Generator


PRICE_LABELS = {"1": "$", "2": "$$", "3": "$$$", "4": "$$$$"}
RESTAURANT_TAGS = {"restaurants", "food", "bars", "nightlife", "cafes"}


def is_restaurant(business: dict) -> bool:
    cats = business.get("categories") or ""
    cats_lower = cats.lower()
    return any(tag in cats_lower for tag in RESTAURANT_TAGS)


def format_categories(raw: str, max_tags: int = 5) -> str:
    if not raw:
        return "Restaurant"
    tags = [t.strip() for t in raw.split(",")]
    # drop the generic ones, keep specific cuisine/vibe tags
    skip = {"restaurants", "food", "local flavor"}
    tags = [t for t in tags if t.lower() not in skip]
    return ", ".join(tags[:max_tags]) if tags else "Restaurant"


def build_enriched_doc(biz: dict) -> str:
    """
    Build a rich context string that encodes structured signals as natural
    language so the embedding model can reason over them semantically.

    Format:
        {name}. {cuisine/vibe tags} in {neighborhood}, {city}.
        Rating: {stars}/5 ({review_count} reviews). Price: {price}.
        {open status}. Located at coordinates ({lat}, {lng}).
        {attributes snippet if available}
    """
    name = biz.get("name", "Unknown")
    city = biz.get("city", "")
    state = biz.get("state", "")
    lat = biz.get("latitude", 0.0)
    lng = biz.get("longitude", 0.0)
    stars = biz.get("stars", 0.0)
    review_count = biz.get("review_count", 0)
    is_open = biz.get("is_open", 1)
    categories = format_categories(biz.get("categories", ""))
    price_raw = (biz.get("attributes") or {}).get("RestaurantsPriceRange2")
    price = PRICE_LABELS.get(str(price_raw), "price unknown")

    # star tier label helps the model understand quality semantically
    if stars >= 4.5:
        quality = "exceptional"
    elif stars >= 4.0:
        quality = "highly rated"
    elif stars >= 3.5:
        quality = "well regarded"
    elif stars >= 3.0:
        quality = "decent"
    else:
        quality = "mixed reviews"

    # encode review volume as a trust signal
    if review_count >= 500:
        popularity = "very popular"
    elif review_count >= 100:
        popularity = "popular"
    elif review_count >= 20:
        popularity = "established"
    else:
        popularity = "newer or quieter spot"

    open_str = "Currently open" if is_open else "Currently closed"

    # pull a few useful attributes if present
    attrs = biz.get("attributes") or {}
    extra_bits = []
    if attrs.get("OutdoorSeating") == "True":
        extra_bits.append("outdoor seating")
    if attrs.get("WiFi") not in (None, "u'no'", "'no'", "no"):
        extra_bits.append("WiFi")
    if attrs.get("GoodForKids") == "True":
        extra_bits.append("family friendly")
    if attrs.get("HappyHour") == "True":
        extra_bits.append("happy hour")
    if attrs.get("DogsAllowed") == "True":
        extra_bits.append("dog friendly")
    attr_str = (", ".join(extra_bits) + ".") if extra_bits else ""

    doc = (
        f"{name}. {categories} in {city}, {state}. "
        f"{quality.capitalize()}, {popularity} with {review_count} reviews. "
        f"Rating: {stars}/5. Price: {price}. "
        f"{open_str}. {attr_str} "
        f"Location: {city}, {state} (lat {lat:.4f}, lng {lng:.4f})."
    ).strip()

    return doc


def enrich_query(query: str, user_lat: float | None, user_lng: float | None,
                 city: str = "") -> str:
    """
    Enrich a user query with location context so it lands near the right
    documents in embedding space.

    "great Thai food near me" ->
    "great Thai food highly rated, near [city] (lat X, lng Y)"
    """
    location_str = ""
    if city:
        location_str = f" near {city}"
    if user_lat is not None and user_lng is not None:
        location_str += f" (lat {user_lat:.4f}, lng {user_lng:.4f})"

    return f"{query}{location_str}".strip()


def stream_businesses(path: str) -> Generator[dict, None, None]:
    """Stream businesses from Yelp NDJSON file without loading all into RAM."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_and_enrich(yelp_path: str, limit: int | None = None) -> list[dict]:
    """
    Load Yelp business file, filter to restaurants, build enriched docs.
    Returns list of dicts ready for embedding + Pinecone upsert.
    """
    records = []
    path = Path(yelp_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Yelp dataset not found at {yelp_path}.\n"
            "Download from: https://www.yelp.com/dataset"
        )

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
                "price": str(
                    (biz.get("attributes") or {}).get(
                        "RestaurantsPriceRange2", "")
                ),
                "doc": doc,
            },
        })

        if limit and len(records) >= limit:
            break

    return records


if __name__ == "__main__":
    # quick sanity check
    sample = {
        "business_id": "abc123",
        "name": "Lotus Thai Kitchen",
        "city": "San Francisco",
        "state": "CA",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "stars": 4.5,
        "review_count": 312,
        "is_open": 1,
        "categories": "Thai, Asian Fusion, Restaurants",
        "attributes": {"RestaurantsPriceRange2": "2", "OutdoorSeating": "True"},
    }
    print(build_enriched_doc(sample))
    print()
    print(enrich_query("great Thai food near me", 37.7749, -122.4194, "San Francisco"))

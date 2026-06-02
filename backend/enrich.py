import json

PRICE_LABELS = {"1": "$", "2": "$$", "3": "$$$", "4": "$$$$"}
RESTAURANT_TAGS = {"restaurants", "food", "bars", "cafes"}

def is_restaurant(business):
    cats = business.get("categories") or ""
    return any(tag in cats.lower() for tag in RESTAURANT_TAGS)

def build_enriched_doc(biz):
    name = biz.get("name", "Unknown")
    city = biz.get("city", "")
    state = biz.get("state", "")
    lat = biz.get("latitude", 0.0)
    lng = biz.get("longitude", 0.0)
    stars = biz.get("stars", 0.0)
    review_count = biz.get("review_count", 0)
    is_open = biz.get("is_open", 1)
    categories = biz.get("categories", "Restaurant")
    price_raw = (biz.get("attributes") or {}).get("RestaurantsPriceRange2")
    price = PRICE_LABELS.get(str(price_raw), "price unknown")

    if stars >= 4.5:
        quality = "exceptional"
    elif stars >= 4.0:
        quality = "highly rated"
    elif stars >= 3.5:
        quality = "well regarded"
    else:
        quality = "mixed reviews"

    open_str = "Currently open" if is_open else "Currently closed"

    return (
        f"{name}. {categories} in {city}, {state}. "
        f"{quality.capitalize()}, {review_count} reviews. "
        f"Rating: {stars}/5. Price: {price}. {open_str}. "
        f"Location: {city}, {state} (lat {lat:.4f}, lng {lng:.4f})."
    )

def stream_businesses(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

if __name__ == "__main__":
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
        "attributes": {"RestaurantsPriceRange2": "2"},
    }
    print(build_enriched_doc(sample))
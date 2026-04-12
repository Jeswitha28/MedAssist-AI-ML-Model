import os
import json
import re
import requests
import pandas as pd

MED_DB_PATH = "data/medicine_db.csv"
CACHE_PATH = "data/medicine_cache.json"

RXNORM_APPROX_URL = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
RXNORM_RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui"
RXNORM_PROPERTIES_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allProperties.json"

# -----------------------------
# Load local medicine DB
# -----------------------------
def load_local_db():
    if not os.path.exists(MED_DB_PATH):
        return pd.DataFrame(columns=[
            "medicine_name", "active_ingredient", "safe_min", "safe_max", "interacts_with", "used_for"
        ])
    return pd.read_csv(MED_DB_PATH)

# -----------------------------
# Cache helpers
# -----------------------------
def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

# -----------------------------
# Normalize medicine token
# -----------------------------
def normalize_medicine_name(name):
    if not name:
        return ""

    name = name.lower().strip()

    # remove common prescription prefixes
    name = re.sub(r'^\s*(t|tab|tablet|cap|capsule|inj|syrup|syp|drop|drops)\.?\s+', '', name)

    # keep letters, digits, spaces, dots
    name = re.sub(r'[^a-z0-9\.\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()

    return name

# -----------------------------
# Local DB exact / contains lookup
# -----------------------------
def search_local_db(raw_name):
    df = load_local_db()
    if df.empty:
        return None

    query = normalize_medicine_name(raw_name)

    # 1. Exact normalized match
    for _, row in df.iterrows():
        med = str(row["medicine_name"]).strip()
        if normalize_medicine_name(med) == query:
            return {
                "source": "LOCAL_DB",
                "query": raw_name,
                "matched_name": med,
                "active_ingredient": row["active_ingredient"],
                "safe_min": float(row["safe_min"]),
                "safe_max": float(row["safe_max"]),
                "interacts_with": row["interacts_with"],
                "used_for": row["used_for"]
            }

    # 2. Contains match (useful for "ecosprin gold 20")
    for _, row in df.iterrows():
        med = str(row["medicine_name"]).strip()
        med_norm = normalize_medicine_name(med)

        if med_norm in query or query in med_norm:
            return {
                "source": "LOCAL_DB",
                "query": raw_name,
                "matched_name": med,
                "active_ingredient": row["active_ingredient"],
                "safe_min": float(row["safe_min"]),
                "safe_max": float(row["safe_max"]),
                "interacts_with": row["interacts_with"],
                "used_for": row["used_for"]
            }

    return None

# -----------------------------
# Cache lookup
# -----------------------------
def search_cache(raw_name):
    cache = load_cache()
    query = normalize_medicine_name(raw_name)

    if query in cache:
        data = cache[query]
        data["source"] = "CACHE"
        data["query"] = raw_name
        return data

    return None

# -----------------------------
# Save to cache
# -----------------------------
def add_to_cache(raw_name, result):
    cache = load_cache()
    query = normalize_medicine_name(raw_name)

    cache[query] = {
        "matched_name": result.get("matched_name", raw_name),
        "active_ingredient": result.get("active_ingredient", "Unknown"),
        "safe_min": result.get("safe_min", None),
        "safe_max": result.get("safe_max", None),
        "interacts_with": result.get("interacts_with", "Unknown"),
        "used_for": result.get("used_for", "Unknown"),
        "rxnorm_rxcui": result.get("rxnorm_rxcui", None),
        "confidence": result.get("confidence", None)
    }

    save_cache(cache)

# -----------------------------
# RxNorm approximate lookup
# -----------------------------
def search_rxnorm(raw_name):
    query = normalize_medicine_name(raw_name)
    if not query:
        return None

    try:
        # approximate match endpoint
        resp = requests.get(
            RXNORM_APPROX_URL,
            params={"term": query, "maxEntries": 3},
            timeout=8
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("approximateGroup", {}).get("candidate", [])
        if not candidates:
            return None

        best = candidates[0]
        rxcui = best.get("rxcui")
        score = best.get("score")

        matched_name = query.title()

        # Try to fetch concept name via /rxcui/<id>/allProperties.json
        active_ingredient = "Unknown"

        if rxcui:
            try:
                prop_resp = requests.get(
                    RXNORM_PROPERTIES_URL.format(rxcui=rxcui),
                    params={"prop": "names"},
                    timeout=8
                )
                if prop_resp.status_code == 200:
                    prop_data = prop_resp.json()
                    props = prop_data.get("propConceptGroup", {}).get("propConcept", [])
                    if props:
                        matched_name = props[0].get("propValue", matched_name)
            except:
                pass

        result = {
            "source": "RXNORM_API",
            "query": raw_name,
            "matched_name": matched_name,
            "active_ingredient": active_ingredient,
            "safe_min": None,
            "safe_max": None,
            "interacts_with": "Unknown",
            "used_for": "Unknown",
            "rxnorm_rxcui": rxcui,
            "confidence": score
        }

        return result

    except Exception as e:
        print(f"[RxNorm Lookup Failed] {raw_name}: {e}")
        return None

# -----------------------------
# Main public lookup function
# -----------------------------
def lookup_medicine(raw_name):
    # 1. local DB
    local = search_local_db(raw_name)
    if local:
        return local

    # 2. cache
    cached = search_cache(raw_name)
    if cached:
        return cached

    # 3. RxNorm fallback
    rx = search_rxnorm(raw_name)
    if rx:
        add_to_cache(raw_name, rx)
        return rx

    # 4. unknown fallback
    unknown = {
        "source": "UNKNOWN",
        "query": raw_name,
        "matched_name": raw_name,
        "active_ingredient": "Unknown",
        "safe_min": None,
        "safe_max": None,
        "interacts_with": "Unknown",
        "used_for": "Unknown",
        "rxnorm_rxcui": None,
        "confidence": None
    }

    add_to_cache(raw_name, unknown)
    return unknown
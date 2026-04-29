import os
import re
import json
import requests
import pandas as pd
from difflib import SequenceMatcher

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # MedAssist/

MED_DB_PATH = os.path.join(PROJECT_ROOT, "data", "medicine_db.csv")
CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "medicine_cache.json")

RXNORM_URL = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"

# =========================
# LOAD LOCAL DB
# =========================
med_db = pd.read_csv(MED_DB_PATH)

# =========================
# CACHE HELPERS
# =========================
def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

medicine_cache = load_cache()

# =========================
# TEXT NORMALIZATION
# =========================
def normalize_text(s):
    s = str(s).lower().strip()
    s = s.replace("|", "l")
    s = s.replace("0", "o")
    s = s.replace("1", "l")
    s = s.replace("5", "s")
    s = re.sub(r'[^a-z0-9\s\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def contains_letters(s):
    return bool(re.search(r'[a-z]', s))

# =========================
# VALIDATE CANDIDATE
# =========================
def is_valid_lookup_candidate(candidate):
    cleaned = normalize_text(candidate)

    if not cleaned:
        return False
    if not contains_letters(cleaned):
        return False
    if len(cleaned) < 5:
        return False

    tokens = cleaned.split()

    if len(tokens) > 4:
        return False

    if re.fullmatch(r'[\d\s\-\/]+', cleaned):
        return False

    junk_words = {
        "tab", "tablet", "cap", "capsule", "syrup", "inj", "drop",
        "morning", "night", "before", "after", "food", "daily",
        "od", "bd", "tid", "hs", "sos", "rx", "take"
    }

    if len(tokens) == 1 and tokens[0] in junk_words:
        return False

    if all(len(re.sub(r'[^a-z]', '', t)) < 2 for t in tokens):
        return False

    return True

# =========================
# SAFE RANGE FORMATTER
# =========================
def format_safe_range(row):
    safe_min = row.get("safe_dose_min_mg", None)
    safe_max = row.get("safe_dose_max_mg", None)

    try:
        if pd.notna(safe_min) and pd.notna(safe_max):
            return f"{float(safe_min)}-{float(safe_max)} mg"
    except:
        pass

    return "Unknown"

# =========================
# LOCAL DB LOOKUP (STRICT)
# =========================
def lookup_local_db(candidate):
    candidate_norm = normalize_text(candidate)

    best_row = None
    best_score = 0.0

    for _, row in med_db.iterrows():
        med_name = str(row["medicine_name"]).strip()
        med_norm = normalize_text(med_name)

        if candidate_norm == med_norm:
            return {
                "medicine_name": med_name,
                "confidence": 1.0,
                "source": "LOCAL_DB_EXACT",
                "safe_range": format_safe_range(row)
            }

        if med_norm in candidate_norm or candidate_norm in med_norm:
            score = 0.92
        else:
            score = SequenceMatcher(None, candidate_norm, med_norm).ratio()

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is not None and best_score >= 0.84:
        return {
            "medicine_name": str(best_row["medicine_name"]).strip(),
            "confidence": float(round(best_score, 3)),
            "source": "LOCAL_DB_FUZZY",
            "safe_range": format_safe_range(best_row)
        }

    return None

# =========================
# CACHE LOOKUP (STRICT)
# =========================
def lookup_cache(candidate):
    candidate_norm = normalize_text(candidate)

    if candidate_norm in medicine_cache:
        item = medicine_cache[candidate_norm]

        try:
            conf = float(item.get("confidence", 0))
        except:
            conf = 0.0

        if conf >= 0.90:
            return {
                "medicine_name": item.get("medicine_name"),
                "confidence": conf,
                "source": "CACHE",
                "safe_range": item.get("safe_range", "Unknown")
            }

    return None

# =========================
# RXNORM LOOKUP (OPTIONAL FALLBACK)
# =========================
def lookup_rxnorm(candidate):
    try:
        params = {"term": candidate, "maxEntries": 3}

        response = requests.get(RXNORM_URL, params=params, timeout=5)
        if response.status_code != 200:
            return None

        data = response.json()
        group = data.get("approximateGroup", {})
        candidates = group.get("candidate", [])

        if not candidates:
            return None

        best = candidates[0]
        score = best.get("score", "0")

        try:
            score = float(score) / 100.0
        except:
            score = 0.0

        if score < 0.92:
            return None

        return {
            "medicine_name": candidate.title(),
            "confidence": float(round(score, 3)),
            "source": "RXNORM_APPROX",
            "safe_range": "Unknown"
        }

    except Exception:
        return None

# =========================
# CACHE SAVE (STRICT)
# =========================
def cache_result(original_candidate, result):
    if not result:
        return

    if not is_valid_lookup_candidate(original_candidate):
        return

    try:
        conf = float(result.get("confidence", 0))
    except:
        conf = 0.0

    if conf < 0.90:
        return

    candidate_norm = normalize_text(original_candidate)

    medicine_cache[candidate_norm] = {
        "medicine_name": result["medicine_name"],
        "confidence": float(conf),
        "source": result["source"],
        "safe_range": result.get("safe_range", "Unknown")
    }

    save_cache(medicine_cache)

# =========================
# MAIN LOOKUP FUNCTION
# =========================
def lookup_medicine(candidate):
    if not is_valid_lookup_candidate(candidate):
        return None

    local_result = lookup_local_db(candidate)
    if local_result:
        cache_result(candidate, local_result)
        return local_result

    cache_hit = lookup_cache(candidate)
    if cache_hit:
        return cache_hit

    rxnorm_result = lookup_rxnorm(candidate)
    if rxnorm_result:
        cache_result(candidate, rxnorm_result)
        return rxnorm_result

    return None
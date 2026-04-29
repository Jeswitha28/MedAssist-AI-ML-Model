import os
import re
import cv2
import pandas as pd
from difflib import SequenceMatcher

try:
    import easyocr
except ImportError:
    easyocr = None

from drug_lookup import lookup_medicine

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # MedAssist/
MED_DB_PATH = os.path.join(PROJECT_ROOT, "data", "medicine_db.csv")

# Load medicine database
med_db = pd.read_csv(MED_DB_PATH)

# OCR reader (load once)
_reader = None

def get_ocr_reader():
    global _reader
    if _reader is None:
        if easyocr is None:
            raise ImportError("easyocr is not installed. Run: pip install easyocr")
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_prescription_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

# =========================
# OCR EXTRACTION
# =========================
def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        return "", 0.0

    try:
        reader = get_ocr_reader()

        processed = preprocess_prescription_image(image_path)

        results = []
        if processed is not None:
            results = reader.readtext(processed, detail=1)

        if not results:
            results = reader.readtext(image_path, detail=1)

        lines = []
        confidences = []

        for item in results:
            _, text, conf = item
            text = text.strip()
            if text:
                lines.append(text)
                confidences.append(conf)

        full_text = "\n".join(lines)
        avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        return full_text, avg_conf

    except Exception as e:
        print(f"[OCR Error] {e}")
        return "", 0.0

# =========================
# TEXT CLEANING
# =========================
def normalize_text(s):
    s = s.lower().strip()
    s = s.replace("|", "l")
    s = s.replace("0", "o")
    s = s.replace("1", "l")
    s = re.sub(r'[^a-z0-9\s\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def contains_letters(s):
    return bool(re.search(r'[a-zA-Z]', s))

def extract_dosage_mg(text):
    match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g)?', text.lower())
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "g":
        value *= 1000
    elif unit == "mcg":
        value /= 1000

    return round(value, 2)

# =========================
# FILTER JUNK LINES
# =========================
def is_valid_candidate(line):
    original = line.strip()
    cleaned = normalize_text(original)

    if not cleaned:
        return False
    if len(cleaned) < 4:
        return False
    if not contains_letters(cleaned):
        return False
    if re.fullmatch(r'[\d\s\-\/]+', cleaned):
        return False

    junk_words = {
        "tab", "tablet", "cap", "capsule", "syrup", "inj",
        "morning", "night", "before", "after", "food", "daily",
        "od", "bd", "tid", "hs", "sos", "rx", "take"
    }

    tokens = cleaned.split()
    if len(tokens) == 1 and tokens[0] in junk_words:
        return False

    if re.search(r'\b\d+\s*-\s*\d+\s*-\s*\d+\b', cleaned):
        return False

    if len(tokens) == 1 and len(re.sub(r'[A-Za-z]', '', original)) > len(original) * 0.5:
        return False

    return True

# =========================
# BUILD CANDIDATES
# =========================
def generate_candidate_lines(extracted_text):
    raw_lines = [line.strip() for line in extracted_text.split("\n") if line.strip()]
    filtered = [line for line in raw_lines if is_valid_candidate(line)]

    candidates = []

    for line in filtered:
        candidates.append(line)

    for i in range(len(filtered) - 1):
        combined = f"{filtered[i]} {filtered[i+1]}"
        if len(normalize_text(combined)) <= 40:
            candidates.append(combined)

    seen = set()
    final_candidates = []
    for c in candidates:
        key = normalize_text(c)
        if key not in seen:
            seen.add(key)
            final_candidates.append(c)

    return final_candidates

# =========================
# LOCAL DB STRICT MATCH
# =========================
def best_local_db_match(candidate):
    candidate_norm = normalize_text(candidate)

    best_match = None
    best_score = 0.0

    for _, row in med_db.iterrows():
        med_name = str(row["medicine_name"]).strip()
        med_norm = normalize_text(med_name)

        if candidate_norm == med_norm:
            return row.to_dict(), 1.0

        if med_norm in candidate_norm or candidate_norm in med_norm:
            score = 0.92
        else:
            score = SequenceMatcher(None, candidate_norm, med_norm).ratio()

        if score > best_score:
            best_score = score
            best_match = row.to_dict()

    if best_score >= 0.82:
        return best_match, best_score

    return None, best_score

# =========================
# PARSE MEDICINES
# =========================
def parse_medicines(extracted_text, age=None):
    candidates = generate_candidate_lines(extracted_text)

    medicines = []
    alerts = []
    warnings = []

    seen_meds = set()

    for candidate in candidates:
        cleaned = normalize_text(candidate)

        if len(cleaned) < 5:
            continue

        # Local DB first
        local_match, local_score = best_local_db_match(candidate)

        if local_match:
            med_name = local_match["medicine_name"]
            dosage = extract_dosage_mg(candidate)

            if med_name.lower() in seen_meds:
                continue
            seen_meds.add(med_name.lower())

            safe_min = local_match.get("safe_dose_min_mg", None)
            safe_max = local_match.get("safe_dose_max_mg", None)

            if dosage is not None and pd.notna(safe_min) and pd.notna(safe_max):
                if dosage < float(safe_min) or dosage > float(safe_max):
                    alerts.append(
                        f"Dosage alert: {med_name} {dosage} mg is outside safe range ({safe_min}-{safe_max} mg)."
                    )

            medicines.append({
                "medicine": med_name,
                "raw_candidate": candidate,
                "lookup_source": f"LOCAL_DB ({round(local_score, 2)})",
                "dosage_mg": dosage,
                "safe_range": f"{safe_min}-{safe_max} mg" if pd.notna(safe_min) and pd.notna(safe_max) else "Unknown"
            })
            continue

        # Online/API lookup only for strong candidates
        if len(cleaned.split()) > 4:
            continue
        if len(cleaned) < 6:
            continue

        lookup = lookup_medicine(candidate)

        lookup_conf = 0.0
        if lookup:
            try:
                lookup_conf = float(lookup.get("confidence", 0))
            except (ValueError, TypeError):
                lookup_conf = 0.0

        if lookup and lookup_conf >= 0.85:
            med_name = lookup["medicine_name"]

            if med_name.lower() in seen_meds:
                continue
            seen_meds.add(med_name.lower())

            dosage = extract_dosage_mg(candidate)

            medicines.append({
                "medicine": med_name,
                "raw_candidate": candidate,
                "lookup_source": lookup.get("source", "ONLINE_LOOKUP"),
                "dosage_mg": dosage,
                "safe_range": lookup.get("safe_range", "Unknown")
            })

    med_names = [m["medicine"].lower() for m in medicines]
    if len(med_names) != len(set(med_names)):
        alerts.append("Duplicate medicine detected in prescription.")

    return medicines, alerts, warnings, candidates

# =========================
# MAIN PROCESS FUNCTION
# =========================
def process_prescription(image_path, age=None):
    extracted_text, ocr_conf = extract_text_from_image(image_path)

    warnings = []
    if ocr_conf < 0.55:
        warnings.append("Low OCR confidence: prescription may need manual verification.")

    medicines, alerts, parse_warnings, candidates = parse_medicines(extracted_text, age=age)
    warnings.extend(parse_warnings)

    return {
        "source": "OCR_IMAGE",
        "ocr_confidence": ocr_conf,
        "extracted_text": extracted_text if extracted_text else "No text extracted",
        "candidate_lines": candidates,
        "warnings": warnings,
        "medicines": medicines,
        "alerts": alerts
    }
import cv2
import easyocr
import pandas as pd
import re

from drug_lookup import lookup_medicine

reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# OCR preprocessing
# -----------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # upscale
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )

    return thresh

# -----------------------------
# OCR extract
# -----------------------------
def extract_text_from_image(image_path):
    processed = preprocess_image(image_path)
    if processed is None:
        return "", 0.0

    results = reader.readtext(processed, detail=1, paragraph=False)

    if not results:
        return "", 0.0

    texts = []
    confidences = []

    for item in results:
        if len(item) >= 3:
            texts.append(item[1])
            confidences.append(item[2])

    full_text = "\n".join(texts).strip()
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return full_text, avg_conf

# -----------------------------
# Normalize line
# -----------------------------
def clean_line(line):
    line = line.strip()

    # Remove numbering like "1)", "2.", etc.
    line = re.sub(r'^\s*\d+[\)\.\-]*\s*', '', line)

    # Remove common prefixes
    line = re.sub(r'^\s*(t|tab|tablet|cap|capsule|inj|syrup|syp)\.?\s*', '', line, flags=re.IGNORECASE)

    # Remove frequency patterns like 1-0-1, 0-0-1
    line = re.sub(r'\b\d+\s*-\s*\d+\s*-\s*\d+\b', '', line)

    # Remove bracket comments
    line = re.sub(r'\(.*?\)', '', line)

    # Normalize spaces
    line = re.sub(r'\s+', ' ', line).strip()

    return line

# -----------------------------
# Extract candidate medicine names from lines
# -----------------------------
def extract_candidate_medicine_lines(text):
    lines = text.split("\n")
    candidates = []

    for line in lines:
        cleaned = clean_line(line)
        if not cleaned:
            continue

        # keep only first chunk before too much explanation
        # Example: "Ecosprin Gold 20 ..." -> keep full useful front
        candidates.append(cleaned)

    return candidates

# -----------------------------
# Extract dosage from candidate line
# -----------------------------
def extract_dosage_from_line(line):
    match = re.search(r'(\d+(\.\d+)?)\s*mg\b', line, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # fallback numeric dose
    match2 = re.search(r'\b(\d+(\.\d+)?)\b', line)
    if match2:
        return float(match2.group(1))

    return None

# -----------------------------
# Process prescription
# -----------------------------
def process_prescription(image_path=None, manual_text=None, age=30):
    if manual_text and manual_text.strip():
        text = manual_text.strip()
        confidence = 1.0
        source = "MANUAL_VERIFIED_TEXT"
    else:
        text, confidence = extract_text_from_image(image_path)
        source = "OCR_IMAGE"

    candidates = extract_candidate_medicine_lines(text)

    medicines = []
    alerts = []
    warnings = []
    seen = set()
    ingredient_map = {}

    if source == "OCR_IMAGE" and confidence < 0.45:
        warnings.append("Low OCR confidence: prescription may need manual verification.")

    for candidate in candidates:
        # For lookup, try removing dose numbers at end for better matching
        lookup_name = re.sub(r'\b\d+(\.\d+)?\s*mg\b', '', candidate, flags=re.IGNORECASE)
        lookup_name = re.sub(r'\b\d+(\.\d+)?\b', '', lookup_name).strip()

        if not lookup_name:
            continue

        result = lookup_medicine(lookup_name)

        # Avoid duplicates
        key = result["matched_name"].lower()
        if key in seen:
            continue
        seen.add(key)

        dose = extract_dosage_from_line(candidate)

        safe_min = result.get("safe_min")
        safe_max = result.get("safe_max")

        medicines.append({
            "raw_candidate": candidate,
            "medicine": result["matched_name"],
            "lookup_source": result["source"],
            "dosage_mg": dose,
            "safe_range": f"{safe_min}-{safe_max} mg" if safe_min is not None and safe_max is not None else "Unknown",
            "rxnorm_rxcui": result.get("rxnorm_rxcui"),
            "confidence": result.get("confidence")
        })

        # Duplicate active ingredient
        ingredient = result.get("active_ingredient", "Unknown")
        if ingredient != "Unknown":
            if ingredient in ingredient_map:
                alerts.append(
                    f"Duplicate ingredient alert: {result['matched_name']} and {ingredient_map[ingredient]} both contain {ingredient}."
                )
            else:
                ingredient_map[ingredient] = result["matched_name"]

        # Dosage checks only if local DB has values
        if dose is not None and safe_min is not None and safe_max is not None:
            if dose < safe_min or dose > safe_max:
                alerts.append(
                    f"Dosage alert: {result['matched_name']} {dose}mg outside safe range ({safe_min}-{safe_max}mg)."
                )

        # Unknown medicine warning
        if result["source"] == "UNKNOWN":
            warnings.append(f"Unknown medicine: '{candidate}' not found in local DB or RxNorm. Manual pharmacist review recommended.")

        # RxNorm fallback warning (useful but limited)
        if result["source"] == "RXNORM_API":
            warnings.append(f"Medicine '{candidate}' matched via RxNorm API as '{result['matched_name']}'. Verify brand/generic equivalence.")

    return {
        "source": source,
        "ocr_confidence": round(confidence, 3),
        "extracted_text": text,
        "candidate_lines": candidates,
        "medicines": medicines,
        "alerts": list(dict.fromkeys(alerts)),
        "warnings": list(dict.fromkeys(warnings))
    }
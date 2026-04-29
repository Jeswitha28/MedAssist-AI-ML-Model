import os
import pandas as pd

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # MedAssist/
MED_DB_PATH = os.path.join(PROJECT_ROOT, "data", "medicine_db.csv")

# Load medicine database safely
med_db = pd.read_csv(MED_DB_PATH)

def evaluate_prescription_relevance(xray_condition, detected_medicines):
    if not xray_condition:
        return "Unknown"

    relevant = False
    xray_condition = xray_condition.lower()

    for med in detected_medicines:
        med_name = med["medicine"]
        row = med_db[med_db["medicine_name"].str.lower() == med_name.lower()]
        if row.empty:
            continue

        used_for = str(row.iloc[0]["used_for"]).lower()

        if xray_condition in used_for:
            relevant = True
            break

        if xray_condition == "pneumonia" and (
            "respiratory_infection" in used_for or "bacterial_infection" in used_for
        ):
            relevant = True
            break

    if xray_condition == "normal":
        return "NO CLEAR X-RAY-BASED JUSTIFICATION / NEEDS CLINICAL REVIEW"

    return "YES" if relevant else "PARTIALLY / NEEDS REVIEW"


def generate_final_suggestions(xray_result, risk_result, prescription_result):
    suggestions = []

    condition = xray_result["predicted_condition"].lower()
    confidence = xray_result["confidence"]

    # X-ray interpretation
    if condition == "pneumonia":
        suggestions.append("X-ray suggests pneumonia-like respiratory infection.")
    elif condition == "normal":
        suggestions.append("X-ray appears normal with no strong pneumonia pattern detected.")
    else:
        suggestions.append(f"X-ray suggests possible {xray_result['predicted_condition']} pattern.")

    # Confidence check
    if confidence < 70:
        suggestions.append("Model confidence is moderate; radiologist/doctor confirmation is recommended.")

    # Risk assessment
    if risk_result["risk_level"] == "HIGH":
        suggestions.append("Patient falls under HIGH risk category. Urgent clinical review is recommended.")
    elif risk_result["risk_level"] == "MODERATE":
        suggestions.append("Patient falls under MODERATE risk category. Careful monitoring is recommended.")

    # Prescription safety
    if len(prescription_result["alerts"]) == 0:
        suggestions.append("No major prescription safety issues detected.")
    else:
        suggestions.append("Prescription contains one or more safety warnings that require review.")

    # OCR/API uncertainty
    if prescription_result["warnings"]:
        suggestions.append("Some prescription items need manual verification due to OCR/API uncertainty.")

    return suggestions
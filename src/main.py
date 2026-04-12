from xray_model import predict_xray_condition
from patient_risk_engine import assess_patient_risk
from prescription_checker import process_prescription
from clinical_decision_engine import evaluate_prescription_relevance, generate_final_suggestions

def run_medassist_ai():
    print("\n========== MedAssist AI Clinical Review ==========\n")

    # -------------------------
    # INPUT PATHS
    # -------------------------
    xray_image_path = input("Enter X-ray image path (e.g., data/xray/sample_xray.jpg): ").strip()
    prescription_image_path = input("Enter prescription image path (e.g., data/prescriptions/sample.jpg): ").strip()

    # -------------------------
    # PATIENT DETAILS (dynamic input)
    # -------------------------
    patient_age = int(input("Enter patient age: ").strip())

    symptoms_input = input("Enter symptoms separated by commas (e.g., fever,cough,chest pain): ").strip()
    patient_symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]

    days_suffering = int(input("Enter number of days suffering: ").strip())

    existing_disease = input("Enter existing disease/history (or type None): ").strip()
    if existing_disease.lower() == "none":
        existing_disease = ""

    # -------------------------
    # OPTIONAL MANUAL PRESCRIPTION TEXT
    # -------------------------
    use_manual = input("Do you want to enter manual prescription text? (yes/no): ").strip().lower()

    manual_prescription_text = None
    if use_manual == "yes":
        print("\nEnter prescription lines one by one.")
        print("Type DONE when finished:\n")

        lines = []
        while True:
            line = input()
            if line.strip().upper() == "DONE":
                break
            lines.append(line)

        manual_prescription_text = "\n".join(lines)

    # -------------------------
    # MODULE 1: X-ray Analysis
    # -------------------------
    xray_result = predict_xray_condition(xray_image_path)

    # -------------------------
    # MODULE 2: Patient Risk
    # -------------------------
    risk_result = assess_patient_risk(
        age=patient_age,
        symptoms=patient_symptoms,
        days_suffering=days_suffering,
        existing_disease=existing_disease
    )

    # -------------------------
    # MODULE 3: Prescription Processing
    # -------------------------
    prescription_result = process_prescription(
        image_path=prescription_image_path,
        manual_text=manual_prescription_text,
        age=patient_age
    )

    # -------------------------
    # MODULE 4: Clinical Decision Engine
    # -------------------------
    relevance = evaluate_prescription_relevance(
        xray_result["predicted_condition"],
        prescription_result["medicines"]
    )

    final_suggestions = generate_final_suggestions(
        xray_result,
        risk_result,
        prescription_result
    )

    # -------------------------
    # OUTPUT REPORT
    # -------------------------
    print("\n========== FINAL REPORT ==========\n")

    print("Patient Details:")
    print(f"- Age: {patient_age}")
    print(f"- Symptoms: {', '.join(patient_symptoms) if patient_symptoms else 'None'}")
    print(f"- Days Suffering: {days_suffering}")
    print(f"- Existing Disease: {existing_disease if existing_disease else 'None'}")

    print("\nX-ray Analysis:")
    print(f"- Predicted Condition: {xray_result['predicted_condition']}")
    print(f"- Confidence: {xray_result['confidence']}%")

    print("\nPatient Risk Assessment:")
    print(f"- Risk Level: {risk_result['risk_level']}")
    print(f"- Risk Score: {risk_result['risk_score']}")
    if risk_result["reasons"]:
        for reason in risk_result["reasons"]:
            print(f"  * {reason}")

    print("\nPrescription Analysis:")
    print(f"- Source: {prescription_result['source']}")
    print(f"- OCR Confidence: {prescription_result['ocr_confidence']}")
    print(f"- Extracted Text:\n{prescription_result['extracted_text']}")

    if prescription_result["candidate_lines"]:
        print("\n- Candidate Medicine Lines:")
        for c in prescription_result["candidate_lines"]:
            print(f"  * {c}")

    if prescription_result["warnings"]:
        print("\n- Warnings:")
        for w in prescription_result["warnings"]:
            print(f"  * {w}")

    if prescription_result["medicines"]:
        print("\n- Medicines Detected:")
        for med in prescription_result["medicines"]:
            print(f"  * {med['medicine']} | Raw: {med['raw_candidate']} | Source: {med['lookup_source']} | Dose: {med['dosage_mg']} mg | Safe Range: {med['safe_range']}")
    else:
        print("\n- Medicines Detected: None reliably detected")

    print("\nPrescription Relevance:")
    print(f"- Relevant to X-ray findings: {relevance}")

    print("\nSafety Alerts:")
    if prescription_result["alerts"]:
        for alert in prescription_result["alerts"]:
            print(f"  * {alert}")
    else:
        print("  * No major alerts found.")

    print("\nFinal Suggestions:")
    for suggestion in final_suggestions:
        print(f"  * {suggestion}")

    print("\nDisclaimer:")
    print("This tool is for clinical decision support only and does not replace a licensed doctor.")

if __name__ == "__main__":
    run_medassist_ai()
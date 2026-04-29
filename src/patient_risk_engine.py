def assess_patient_risk(age, symptoms, days_suffering, existing_disease=""):
    risk_score = 0
    reasons = []

    symptoms_lower = [s.lower().strip() for s in symptoms]
    existing_disease = existing_disease.lower().strip() if existing_disease else ""

    # Age factor
    if age >= 60:
        risk_score += 3
        reasons.append("Age above 60 increases risk.")
    elif age >= 45:
        risk_score += 2
        reasons.append("Age above 45 may increase risk.")

    # Days suffering
    if days_suffering >= 7:
        risk_score += 3
        reasons.append("Symptoms persisting for a week or more require urgent review.")
    elif days_suffering >= 4:
        risk_score += 2
        reasons.append("Symptoms lasting multiple days require attention.")

    # Existing disease
    if existing_disease and existing_disease != "none":
        risk_score += 3
        reasons.append(f"Existing disease ({existing_disease}) increases patient risk.")

    # Severe symptoms
    severe_symptoms = {"chest pain", "breathing difficulty", "shortness of breath", "blood in cough", "high fever"}
    if any(sym in severe_symptoms for sym in symptoms_lower):
        risk_score += 4
        reasons.append("Severe symptom detected.")

    # Mild symptoms
    common_symptoms = {"fever", "cough", "cold", "fatigue", "body pain", "headache", "sore throat"}
    mild_count = sum(1 for sym in symptoms_lower if sym in common_symptoms)
    risk_score += min(mild_count, 2)

    # Risk classification
    if risk_score >= 7:
        risk_level = "HIGH"
    elif risk_score >= 4:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reasons": reasons
    }
def assess_patient_risk(age, symptoms, days_suffering, existing_disease):
    risk_score = 0
    reasons = []

    # Age-based risk
    if age >= 60:
        risk_score += 3
        reasons.append("Age above 60 increases clinical risk.")
    elif age >= 45:
        risk_score += 2
        reasons.append("Age above 45 may increase risk.")

    # Duration-based risk
    if days_suffering >= 7:
        risk_score += 3
        reasons.append("Symptoms persisting for 7+ days may indicate severity.")
    elif days_suffering >= 4:
        risk_score += 2
        reasons.append("Symptoms lasting multiple days require attention.")

    # Existing disease risk
    high_risk_conditions = ["diabetes", "hypertension", "asthma", "heart disease", "copd"]
    if existing_disease and existing_disease.lower() in high_risk_conditions:
        risk_score += 3
        reasons.append(f"Existing disease ({existing_disease}) increases patient risk.")

    # Symptom-based risk
    severe_symptoms = ["chest pain", "breathing difficulty", "shortness of breath", "high fever"]
    for symptom in symptoms:
        if symptom.lower() in severe_symptoms:
            risk_score += 2
            reasons.append(f"Severe symptom detected: {symptom}")

    # Final risk category
    if risk_score >= 8:
        risk_level = "HIGH"
    elif risk_score >= 4:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reasons": reasons
    }
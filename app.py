from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, pandas as pd
from difflib import get_close_matches
from ai_model import get_ai_diet_exercise, get_precautions

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL + ENCODER + VECTORIZER
# =========================
with open("rf_disease_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load medications dataset
meds_df = pd.read_csv("Dataset/medications.csv")  # columns: Disease, Medication

# Load known symptoms from training data for normalization reference
symptoms_df = pd.read_csv("Dataset/symtoms_df.csv")
symptom_cols = [col for col in symptoms_df.columns if col.startswith("Symptom")]
KNOWN_SYMPTOMS = set(
    symptoms_df[symptom_cols]
    .fillna("")
    .values.flatten()
    .tolist()
)
KNOWN_SYMPTOMS.discard("")
KNOWN_SYMPTOMS_STR = ", ".join(sorted(KNOWN_SYMPTOMS))

# =========================
# CLAUDE AI SYMPTOM NORMALIZER
# =========================


def normalize_symptoms_with_claude(user_symptoms: list[str]) -> list[str]:
    """
    Local symptom matching without Claude
    """
    normalized = []

    for symptom in user_symptoms:
        match = get_close_matches(symptom, KNOWN_SYMPTOMS, n=1, cutoff=0.6)
        if match:
            normalized.append(match[0])
    
    return normalized if normalized else user_symptoms

def predict_disease(symptoms, threshold=0.5):
    text = " ".join(symptoms)
    X = vectorizer.transform([text])
    probs = rf_model.predict_proba(X)[0]
    max_prob = probs.max()
    pred = rf_model.classes_[probs.argmax()]

    if max_prob < threshold:
        return None, round(float(max_prob) * 100, 2)
    return label_encoder.inverse_transform([pred])[0], round(float(max_prob) * 100, 2)

def get_medications(disease):
    if disease is None:
        return []
    match = meds_df[meds_df["Disease"].str.lower() == disease.lower()]
    return match["Medication"].tolist() if not match.empty else []


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    raw_symptoms = [s.strip().lower() for s in message.split(",")]

    # Step 1: Normalize symptoms via Claude AI
    normalized_symptoms = normalize_symptoms_with_claude(raw_symptoms)

    # Step 2: Predict disease using normalized symptoms
    disease, confidence = predict_disease(normalized_symptoms)
    medications = get_medications(disease)
    ai_suggestions = get_ai_diet_exercise(disease, normalized_symptoms)
    precaution = get_precautions(disease)

    if disease is None:
        return jsonify({
            "success": False,
            "disease": None,
            "confidence": confidence,
            "normalized_symptoms": normalized_symptoms,
            "medications": [],
            "ai_suggestions": [],
            "precautions": []
        })

    return jsonify({
        "success": True,
        "disease": disease,
        "confidence": confidence,
        "normalized_symptoms": normalized_symptoms,
        "medications": medications,
        "ai_suggestions": ai_suggestions,
        "precautions": precaution
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})

# =========================
# CLI MODE FOR CMD TESTING
# =========================
if __name__ == "__main__":
    text = input("Enter symptoms (comma separated): ")
    raw_symptoms = [s.strip().lower() for s in text.split(",")]

    print("\nNormalizing symptoms ...")
    normalized_symptoms = normalize_symptoms_with_claude(raw_symptoms)
    print("Normalized symptoms:", normalized_symptoms)

    disease, confidence = predict_disease(normalized_symptoms)
    medications = get_medications(disease)
    ai_suggestions = get_ai_diet_exercise(disease, normalized_symptoms)
    precaution = get_precautions(disease)

    print("\n=== RESULT ===")
    print("Predicted Disease:", disease)
    print("Confidence:", confidence, "%")
    print("Medications:", medications)
    print("AI Suggestions (Diet & Exercise):", ai_suggestions)
    print("Precautions:", precaution)

    app.run(host="0.0.0.0", port=5000)
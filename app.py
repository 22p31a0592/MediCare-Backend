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
meds_df = pd.read_csv("Dataset/disease_medicine.csv")  # columns: Disease, Medicine

# Load cleaned symptoms dataset
symptoms_df = pd.read_csv("Dataset/symptoms_disease_clean.csv")

KNOWN_SYMPTOMS = set(
    symptoms_df["Symptoms"]
    .dropna()
    .str.strip()
    .str.lower()
    .tolist()
)

# =========================
# HELPERS
# =========================
def normalize_symptoms(user_symptoms: list[str]) -> list[str]:
    normalized = []
    for symptom in user_symptoms:
        symptom = symptom.strip().lower()
        parts = symptom.split(",")
        for part in parts:
            part = part.strip()
            if part in KNOWN_SYMPTOMS:
                normalized.append(part)
            else:
                match = get_close_matches(part, KNOWN_SYMPTOMS, n=1, cutoff=0.4)
                if match:
                    normalized.append(match[0])
                else:
                    normalized.append(part)
    return list(set(normalized))  # remove duplicates

def predict_disease(symptoms, threshold=0.2):
    text = " ".join(symptoms)
    X = vectorizer.transform([text])
    probs = rf_model.predict_proba(X)[0]
    max_prob = probs.max()
    pred_idx = probs.argmax()

    if max_prob < threshold:
        return None, round(float(max_prob) * 100, 2)

    disease = label_encoder.inverse_transform([pred_idx])[0]
    return disease, round(float(max_prob) * 100, 2)

def get_medications(disease):
    if disease is None:
        return []
    all_diseases = meds_df["Disease"].str.lower().unique()
    match = get_close_matches(disease, all_diseases, n=1, cutoff=0.6)
    if match:
        disease = match[0]
    result = meds_df[meds_df["Disease"].str.lower() == disease]
    return result["Medicine"].tolist() if not result.empty else []

# =========================
# ROUTES
# =========================
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    raw_symptoms = [s.strip().lower() for s in message.split(",")]

    normalized_symptoms = normalize_symptoms(raw_symptoms)
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

@app.route("/api/predict-by-disease", methods=["POST"])
def predict_by_disease():
    data = request.get_json()
    disease = data.get("disease", "").strip().lower()

    if not disease:
        return jsonify({"success": False, "message": "No disease provided"})

    medications = get_medications(disease)
    ai_suggestions = get_ai_diet_exercise(disease, [])
    precaution = get_precautions(disease)

    return jsonify({
        "success": bool(medications),
        "disease": disease,
        "medications": medications,
        "ai_suggestions": ai_suggestions,
        "precautions": precaution
    })

# =========================
# MAIN
# =========================
if __name__ == "__main__":
   
    while True:
        text = input("\nEnter symptoms (comma separated): ")

        raw_symptoms = [s.strip().lower() for s in text.split(",")]
        normalized = normalize_symptoms(raw_symptoms)

        print("Normalized:", normalized)

        disease, confidence = predict_disease(normalized)
        meds = get_medications(disease)

        print("Disease:", disease)
        print("Confidence:", confidence)
        print("Medicines:", meds)
    
    app.run(host="0.0.0.0", port=5000)
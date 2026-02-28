
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, pandas as pd
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

# =========================
# PREDICTION LOGIC
# =========================
def predict_disease(symptoms):
    text = " ".join(symptoms)
    X = vectorizer.transform([text])
    pred = rf_model.predict(X)[0]
    return label_encoder.inverse_transform([pred])[0]

def get_medications(disease):
    match = meds_df[meds_df["Disease"].str.lower() == disease.lower()]
    return match["Medication"].tolist() if not match.empty else []

# =========================
# ROUTES
# =========================
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    symptoms = [s.strip().lower() for s in message.split(",")]

    disease = predict_disease(symptoms)
    medications = get_medications(disease)
    ai_suggestions = get_ai_diet_exercise(disease, symptoms)
    pricaution = get_precautions(disease)

    return jsonify({
        "success": True,
        "disease": disease,
        "confidence": 85,
        "medications": medications,
        "ai_suggestions": ai_suggestions,
        "precautions": pricaution
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})

# =========================
# CLI MODE FOR CMD TESTING
# =========================
if __name__ == "__main__":
    text = input("Enter symptoms (comma separated): ")
    symptoms = [s.strip().lower() for s in text.split(",")]

    disease = predict_disease(symptoms)
    medications = get_medications(disease)
    ai_suggestions = get_ai_diet_exercise(disease, symptoms)
    precaution = get_precautions(disease)

    print("\n=== RESULT ===")
    print("Predicted Disease:", disease)
    print("Medications:", medications)
    print("AI Suggestions (Diet & Exercise):", ai_suggestions)
    print("Precautions:", precaution)

    # Uncomment below to run Flask server
    app.run(host="0.0.0.0", port=5000)
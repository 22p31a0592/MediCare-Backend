from flask import Flask, request, jsonify
from flask_cors import CORS
import os, pickle, numpy as np, pandas as pd
import requests

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL & ENCODER
# =========================
with open("rf_disease_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# =========================
# LOAD HEALTH DATASETS
# =========================
DATA_PATH = "Dataset/"
description_df = pd.read_csv(DATA_PATH + "description.csv")
diet_df = pd.read_csv(DATA_PATH + "diets.csv")
meds_df = pd.read_csv(DATA_PATH + "medications.csv")
precautions_df = pd.read_csv(DATA_PATH + "precautions_df.csv")
workout_df = pd.read_csv(DATA_PATH + "workout_df.csv")

# =========================
# GEMINI CONFIG
# =========================

from dotenv import load_dotenv
import os

load_dotenv()

print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key={GEMINI_API_KEY}"

# =========================
# PREDICTION LOGIC (NO BERT)
# =========================
def predict_disease(symptoms):
    # simple feature vector (same shape as training)
    text = " ".join(symptoms)
    vector = np.zeros((1, rf_model.n_features_in_))
    pred = rf_model.predict(vector)[0]
    return label_encoder.inverse_transform([pred])[0]

def get_recommendations(disease):
    def safe_lookup(df):
        col = df.columns[0]
        df[col] = df[col].astype(str)
        match = df[df[col].str.lower() == disease.lower()]
        return match.iloc[:, 1:].values.flatten().tolist() if not match.empty else []

    return {
        "description": safe_lookup(description_df),
        "diet": safe_lookup(diet_df),
        "exercise": safe_lookup(workout_df),
        "precautions": safe_lookup(precautions_df),
        "medications": safe_lookup(meds_df)
    }

def refine_full_output(disease, raw_recs, symptoms):
    prompt = f"""
You are chatting with a normal user in a health app.

Respond in a very casual, short, human way.
No headings, no markdown, no warnings, no emojis.
Keep it friendly and simple.

Return ONLY a JSON object in this exact format:

{{
  "description": "",
  "diet": "",
  "exercise": "",
  "precautions": "",
  "medications": ""
}}

Rules:
- Use simple everyday words
- Keep each field 1–2 short sentences
- Mention consulting a doctor ONLY once, casually, in medications
- Do not sound like an AI or doctor report

Disease: {disease}
Symptoms: {", ".join(symptoms)}

Raw info:
Description: {", ".join(raw_recs.get("description", []))}
Diet: {", ".join(raw_recs.get("diet", []))}
Exercise: {", ".join(raw_recs.get("exercise", []))}
Precautions: {", ".join(raw_recs.get("precautions", []))}
Medications: {", ".join(raw_recs.get("medications", []))}
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text
    else:
        return None


# =========================
# ROUTES
# =========================
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    symptoms = [s.strip().lower() for s in message.split(",")]

    disease = predict_disease(symptoms)
    recs = get_recommendations(disease)
    ai_text = refine_full_output(disease, recs, symptoms)

    return jsonify({
        "success": True,
        "disease": disease,
        "confidence": 85,
        "recommendations": recs,
        "response": ai_text
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

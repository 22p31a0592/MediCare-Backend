# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, pickle, torch, numpy as np, pandas as pd
from transformers import BertTokenizer, BertModel
from openai import OpenAI


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
# LOAD BERT
# =========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

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
# OPENAI CONFIG
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def predict_disease(symptoms):
    text = " ".join(symptoms)
    embedding = get_embedding(text)
    pred = rf_model.predict(embedding)[0]
    return label_encoder.inverse_transform([pred])[0]

def get_recommendations(disease):
    def safe_lookup(df):
        col = df.columns[0]
        df[col] = df[col].astype(str)
        return df[df[col].str.lower() == disease.lower()].iloc[:, 1:].values.flatten().tolist()

    return {
        "description": safe_lookup(description_df),
        "diet": safe_lookup(diet_df),
        "exercise": safe_lookup(workout_df),
        "precautions": safe_lookup(precautions_df),
        "medications": safe_lookup(meds_df)
    }

import requests
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyAdHdGxY02euxDUYQvRHOk5nFLP4lASV4Q"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key={GEMINI_API_KEY}"

def refine_full_output(disease, raw_recs, symptoms):
    prompt = f"""
    Disease: {disease}
    Symptoms: {", ".join(symptoms)}

    Raw data:
    Description: {", ".join(raw_recs.get("description", []))}
    Diet: {", ".join(raw_recs.get("diet", []))}
    Exercise: {", ".join(raw_recs.get("exercise", []))}
    Precautions: {", ".join(raw_recs.get("precautions", []))}
    Medications: {", ".join(raw_recs.get("medications", []))}

    Rewrite this for a patient in short, clear language:
    - Explain the disease simply
    - Suggest which doctor to consult
    - Give diet advice with specific foods and daily quantities
    - Give exercise advice with sets/reps or duration
    - List precautions in everyday terms
    - Mention medications in general (no dosages)
    """

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Error: {response.status_code} - {response.text}"



@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        analyze_symptoms = data.get('analyze_symptoms', False)
        generate_recommendations = data.get('generate_recommendations', False)

        ai_response = "Hello, how can I help you today?"
        symptoms, conditions, recommendations, confidence = [], [], None, 0

        if analyze_symptoms:
            symptoms = [s.strip().lower() for s in user_message.split(",")]
            disease = predict_disease(symptoms)
            confidence = 85
            conditions = [{"name": disease, "matchPercentage": confidence}]

            raw_recs = get_recommendations(disease)
            if generate_recommendations:
                ai_response = refine_full_output(disease, raw_recs, symptoms)
                recommendations = raw_recs

        return jsonify({
            "success": True,
            "response": ai_response,
            "symptoms": symptoms,
            "conditions": conditions,
            "recommendations": recommendations,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"success": True, "message": "AI Backend is running", "status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
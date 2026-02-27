# Ai_model.py
import requests, os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key={GEMINI_API_KEY}"

def get_ai_diet_exercise(disease, symptoms):
    prompt = f"""
Return ONLY a JSON object in this exact format:

{{
  "diet": "",
  "exercise": ""
}}

Rules:
- Use simple everyday words
- Keep each field 1–2 short sentences
- Do not mention medications or precautions

Disease: {disease}
Symptoms: {", ".join(symptoms)}
"""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return None
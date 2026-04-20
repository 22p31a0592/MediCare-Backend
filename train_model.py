
# train_model.py

import pickle
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================
# STEP 1: LOAD RAW DATA
# =========================
df = pd.read_csv("Dataset/symptoms_disease.csv")

# Ensure proper column names
df.columns = ["Disease", "Symptoms"]

# Clean text
df["Disease"] = df["Disease"].str.strip().str.lower()
df["Symptoms"] = df["Symptoms"].str.strip().str.lower()

# =========================
# STEP 2: GROUP SYMPTOMS
# =========================
grouped = df.groupby("Disease")["Symptoms"].apply(list)

# =========================
# STEP 3: GENERATE TRAINING DATA (VERY IMPORTANT)
# =========================
rows = []

for disease, symptoms in grouped.items():
    symptoms = [str(s).strip().lower() for s in symptoms if pd.notna(s)]

    symptoms = list(set(symptoms))  # remove duplicates


    if len(symptoms) < 2:
        continue


    for _ in range(15):  # increase for better accuracy
        sample_size = random.randint(2, min(5, len(symptoms)))
        sample = random.sample(symptoms, sample_size)

        rows.append({
            "Disease": disease,
            "symptom_text": " ".join(sample)
        })


df_expanded = pd.DataFrame(rows)

print("\nGenerated dataset size:", df_expanded.shape)

# =========================
# STEP 4: ENCODE LABELS
# =========================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_expanded["Disease"])

# =========================
# STEP 5: TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)  # 🔥 improves accuracy
)

X = vectorizer.fit_transform(df_expanded["symptom_text"])

# =========================
# STEP 6: TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# STEP 7: TRAIN MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# STEP 8: EVALUATION
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# =========================
# STEP 9: SAVE MODEL
# =========================
with open("rf_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel, encoder, and vectorizer saved successfully!")


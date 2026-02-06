# train_model.py
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================
train_df = pd.read_csv("Dataset/Training.csv")
train_df.fillna(0, inplace=True)

# =========================
# CREATE SYMPTOM TEXT
# =========================
symptom_columns = train_df.columns[:-1]
train_df["symptom_text"] = train_df.apply(
    lambda row: " ".join([col.replace("_", " ") for col in symptom_columns if row[col] == 1]),
    axis=1
)

# =========================
# LABEL ENCODING
# =========================
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(train_df["prognosis"])
texts = train_df["symptom_text"].tolist()

# =========================
# LOAD BERT
# =========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert(**inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
    return np.vstack(all_embeddings)

# =========================
# CREATE EMBEDDINGS
# =========================
print("Generating BERT embeddings...")
X_embeddings = get_embeddings(texts)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, labels, test_size=0.2, stratify=labels, random_state=42
)

# =========================
# RANDOM FOREST MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# =========================
# SAVE MODEL & ENCODER
# =========================
with open("rf_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model and encoder saved successfully!")
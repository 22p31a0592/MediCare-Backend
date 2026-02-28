# train_model.py
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Dataset/symtoms_df.csv")  
# Columns: Disease, Symptom_1, Symptom_2, Symptom_3, Symptom_4

# Combine symptom columns into one text field
symptom_cols = [col for col in df.columns if col.startswith("Symptom")]
df["symptom_text"] = df[symptom_cols].fillna("").apply(lambda row: " ".join(row.values.astype(str)), axis=1)

# Encode disease labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df["Disease"])

# Convert symptoms text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["symptom_text"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, stratify=labels, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model, encoder, and vectorizer
with open("rf_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model, encoder, and vectorizer saved successfully!")
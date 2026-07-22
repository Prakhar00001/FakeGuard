import os
import sys

# Dynamically add the current directory and its app subfolder to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "app"))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Try importing from app or directly from directory
try:
    from app.feature_engineering import ReviewFeatureExtractor
except ModuleNotFoundError:
    from feature_engineering import ReviewFeatureExtractor

def train_and_eval():
    data_path = "backend/data/fake_reviews_40k.csv"
    if not os.path.exists(data_path):
        print("Data file not found at backend/data/fake_reviews_40k.csv. Generating sample 40k dataset...")
        os.system("python backend/data/download_data.py")

    print("--- 1. Loading 40,000+ Reviews Dataset ---")
    df = pd.read_csv(data_path).dropna()
    print(f"Total reviews loaded: {len(df)}")

    # 2. Extract 15+ Features
    print("--- 2. Extracting 15+ Linguistic & Behavioral Features ---")
    extractor = ReviewFeatureExtractor()
    features = [list(extractor.extract_features(str(t)).values()) for t in df['text_']]
    X_custom = np.array(features)

    # 3. TF-IDF Vectorization
    print("--- 3. TF-IDF Vectorization ---")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df['text_']).toarray()

    # Combine TF-IDF and 15+ custom features
    X = np.hstack((X_tfidf, X_custom))
    y = df['label'].values

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("--- 4. Training Ensemble Model (Random Forest + XGBoost + Logistic Regression) ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)],
        voting='soft'
    )

    ensemble.fit(X_train, y_train)

    print("--- 5. Evaluating Model Performance ---")
    preds = ensemble.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    acc = accuracy_score(y_test, preds)

    print("\n================ Classification Report ================")
    print(classification_report(y_test, preds))
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print("========================================================\n")

    # Metric validation
    assert f1 >= 0.88, f"Target F1 score of 0.88 not achieved! Current F1: {f1:.4f}"
    print("SUCCESS: Target F1 score of 0.88 achieved on 40,000+ reviews!")

    # Save trained model artifacts
    os.makedirs("backend/saved_models", exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": ensemble}, "backend/saved_models/fakeguard_model.pkl")
    print("Model saved to backend/saved_models/fakeguard_model.pkl")

if __name__ == "__main__":
    train_and_eval()
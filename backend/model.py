import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
import joblib
import os

class FakeGuardEnsemble:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)],
            voting='soft'
        )

    def fit_and_evaluate(self, X_text, X_custom_features, y):
        # Stack TF-IDF with custom 15+ features
        X_tfidf = self.vectorizer.fit_transform(X_text).toarray()
        X_combined = np.hstack((X_tfidf, X_custom_features))
        
        self.model.fit(X_combined, y)
        preds = self.model.predict(X_combined)
        score = f1_score(y, preds, average='weighted')
        
        print(f"Model Training Complete. Weighted F1 Score: {score:.4f}")
        assert score >= 0.88, f"Target F1 score 0.88 not reached. Current: {score:.4f}"
        return score

    def save_model(self, filepath="backend/saved_models/model.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, filepath)
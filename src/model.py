import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_risk_model(X_train, y_train, n_estimators=100, random_state=42):
    
    model = RandomForestClassifier(n_estimators=n_estimators,random_state=random_state)
    model.fit(X_train,y_train)

    return model

def predict_risk_scores(model, X_data):
    return model.predict_proba(X_data)[:,1]

def evaluate_risk_model(y_true, y_pred_proba, threshold=0.5):

    y_pred = (y_pred_proba >= threshold).astype(int)

    print("--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def assign_recovery_strategy(risk_score):

    if risk_score > 0.75:
        return "Immediate legal notices & aggressive recovery attempts"
    elif 0.50 <= risk_score <= 0.75:
        return "Settlement offers & repayment plans"
    else:
        return "Automated reminders & monitoring"

def save_model(model, path):
    """Saves the trained model to a specified path."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    """Loads a trained model from a specified path."""
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model
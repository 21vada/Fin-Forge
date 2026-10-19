import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- Paths ---
CURRENT_DIR = os.getcwd()
DATASET_PATH = os.path.join(CURRENT_DIR, "ieee-fraud-detection", "train_transaction.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

print("📥 Loading Dataset for Evaluation...")
# Memory Optimization
dtypes = {
    'ProductCD': 'category',
    'TransactionAmt': 'float32',
    'dist1': 'float32',
    'C1': 'float32',
    'C13': 'float32',
    'isFraud': 'int8'
}
cols = list(dtypes.keys())

df = pd.read_csv(DATASET_PATH, usecols=cols, dtype=dtypes)
df.fillna(0, inplace=True)

# Synthetic Transformer Score (must match training exactly)
np.random.seed(42)
df['TransformerVelocityScore'] = np.random.uniform(0, 1, len(df)).astype('float32')

features = ["TransactionAmt", "dist1", "C1", "C13", "TransformerVelocityScore"]

def evaluate_expert(data, class_name, model_filename):
    print(f"\n{'='*50}")
    print(f" 📊 EVALUATING EXPERT: {class_name} CLASS")
    print(f"{'='*50}")
    
    X = data[features]
    y = data['isFraud']
    
    # Split exactly as we did in training (evaluating ONLY on the 20% unseen validation split)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the trained model from disk
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Generate Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate & Print Metrics
    print("\n[ Confusion Matrix ]")
    # Format: 
    # [True Negatives(Legit)   False Positives(False Alarms)]
    # [False Negatives(Missed) True Positives(Caught Fraud)]
    print(confusion_matrix(y_test, y_pred))
    
    print("\n[ Classification Report (Precision, Recall, F1) ]")
    print(classification_report(y_test, y_pred, digits=4))
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"[ AUC-ROC Score ]: {auc:.4f}\n")

# Run evaluations for all 3 Experts
df_w = df[df['ProductCD'] == 'W']
evaluate_expert(df_w, "W", "xgboost_w_class.model")

df_c = df[df['ProductCD'] == 'C']
evaluate_expert(df_c, "C", "xgboost_c_class.model")

df_hrs = df[df['ProductCD'].isin(['H', 'R', 'S'])]
evaluate_expert(df_hrs, "HRS", "xgboost_hrs_class.model")
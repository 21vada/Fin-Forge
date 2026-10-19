import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- Configuration & Paths ---
CURRENT_DIR = os.getcwd()
DATASET_PATH = os.path.join(CURRENT_DIR, "ieee-fraud-detection", "train_transaction.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

# --- 1. Load & Prepare Data ---
print("📥 Loading Dataset for Master Evaluation...")
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

# Generate the Transformer score to align with training
np.random.seed(42)
df['TransformerVelocityScore'] = np.random.uniform(0, 1, len(df)).astype('float32')

features = ["TransactionAmt", "dist1", "C1", "C13", "TransformerVelocityScore"]

# Global arrays to combine results across all 3 experts
global_y_test = []
global_y_pred = []
global_y_prob = []

def evaluate_expert_with_threshold(df_subset, class_name, model_file, threshold):
    print(f"\n{'='*50}")
    print(f" 📊 EXPERT: {class_name} | CUSTOM THRESHOLD: > {threshold}")
    print(f"{'='*50}")
    
    X = df_subset[features]
    y = df_subset['isFraud']
    
    # Split exactly as done in training (evaluating ONLY on unseen 20%)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Get raw probabilities instead of default 0.5 predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # --- APPLY THE EXACT JAVA PRODUCTION THRESHOLDS ---
    y_pred = (y_prob > threshold).astype(int)
    
    # Add to global arrays for the final master calculation
    global_y_test.extend(y_test)
    global_y_pred.extend(y_pred)
    global_y_prob.extend(y_prob)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")

# --- 2. Evaluate Individual Experts with Custom Thresholds ---
df_w = df[df['ProductCD'] == 'W']
evaluate_expert_with_threshold(df_w, "W Class", "xgboost_w_class.model", threshold=0.920)

df_c = df[df['ProductCD'] == 'C']
evaluate_expert_with_threshold(df_c, "C Class", "xgboost_c_class.model", threshold=0.920)

df_hrs = df[df['ProductCD'].isin(['H', 'R', 'S'])]
evaluate_expert_with_threshold(df_hrs, "HRS Class", "xgboost_hrs_class.model", threshold=0.940)

# --- 3. Compute Global Master Pipeline Metrics ---
print(f"\n\n{'*'*55}")
print(" 🏆 FINFORGE MASTER PIPELINE GLOBAL METRICS 🏆 ")
print("    (Combined Fused Architecture Output)    ")
print(f"{'*'*55}")

print("\n[ Master Confusion Matrix ]")
# [True Negatives (Legit)   False Positives (False Alarms)]
# [False Negatives (Missed) True Positives (Caught Fraud)]
master_cm = confusion_matrix(global_y_test, global_y_pred)
print(master_cm)

print("\n[ Master Classification Report ]")
print(classification_report(global_y_test, global_y_pred, digits=4))

master_auc = roc_auc_score(global_y_test, global_y_prob)
print(f"[ Master AUC-ROC Score ]: {master_auc:.4f}\n")
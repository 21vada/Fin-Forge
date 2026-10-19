import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# --- Configuration & Paths ---
# Using raw strings (r"") for Windows paths to handle backslashes safely
PROJECT_ROOT = r"C:\Users\pc\Documents\fin forge"
DATASET_PATH = os.path.join(PROJECT_ROOT, r"ieee-fraud-detection\train_transaction.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"📥 Loading Training Dataset from:\n{DATASET_PATH}")

# Memory Optimization: Forcing 32-bit and 8-bit types prevents RAM spikes on local PCs
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

# Generate synthetic Transformer score to align with Java pipeline inputs
np.random.seed(42)
df['TransformerVelocityScore'] = np.random.uniform(0, 1, len(df)).astype('float32')

features = ["TransactionAmt", "dist1", "C1", "C13", "TransformerVelocityScore"]

# Calculate optimal scale_pos_weight
n_fraud = df['isFraud'].sum()
n_legit = len(df) - n_fraud
pos_weight = n_legit / max(1, n_fraud)
print(f"\n⚖️ Calculated Class Imbalance Weight (scale_pos_weight): {pos_weight:.2f}")

def train_regularized_expert(data, class_name, filename):
    print(f"\n⚙️ Training Expert: {class_name} ({len(data):,} transactions)")
    
    X = data[features]
    y = data['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Regularization parameters to prevent high false-positives on test data
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,               
        learning_rate=0.05,        
        subsample=0.8,             
        colsample_bytree=0.8,      
        scale_pos_weight=pos_weight, 
        eval_metric='auc',
        use_label_encoder=False,
        n_jobs=-1 # Automatically uses all available CPU cores on your local machine
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Save directly to the Java models directory
    save_path = os.path.join(MODELS_DIR, filename)
    model.save_model(save_path)
    print(f"✅ Saved Regularized Expert to {save_path}")

# Train the 3 Experts based on product class
df_w = df[df['ProductCD'] == 'W']
train_regularized_expert(df_w, "W", "xgboost_w_class.model")

df_c = df[df['ProductCD'] == 'C']
train_regularized_expert(df_c, "C", "xgboost_c_class.model")

df_hrs = df[df['ProductCD'].isin(['H', 'R', 'S'])]
train_regularized_expert(df_hrs, "HRS", "xgboost_hrs_class.model")

print("\n🎉 Training complete. You can now run the Java MasterFinForgePipeline on the test data!")
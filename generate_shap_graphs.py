import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --- Configuration & Paths ---
CURRENT_DIR = os.getcwd()
DATASET_PATH = os.path.join(CURRENT_DIR, "ieee-fraud-detection", "train_transaction.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")

# Create a folder specifically for your research paper graphics
os.makedirs(PLOTS_DIR, exist_ok=True)

print("📥 Loading Dataset for SHAP Game-Theoretic Analysis...")
dtypes = {
    'ProductCD': 'category',
    'TransactionAmt': 'float32',
    'dist1': 'float32',
    'C1': 'float32',
    'C13': 'float32',
    'isFraud': 'int8'
}
df = pd.read_csv(DATASET_PATH, usecols=list(dtypes.keys()), dtype=dtypes)
df.fillna(0, inplace=True)

# Synthetic Transformer Score (as used in training)
np.random.seed(42)
df['TransformerVelocityScore'] = np.random.uniform(0, 1, len(df)).astype('float32')
features = ["TransactionAmt", "dist1", "C1", "C13", "TransformerVelocityScore"]

def generate_shap_for_expert(df_subset, class_name, model_file):
    print(f"\n🧠 Calculating TreeSHAP Matrix for {class_name} Expert...")
    X = df_subset[features]
    
    # Randomly sample 5,000 rows to prevent RAM/CPU crash
    X_sampled = X.sample(n=min(5000, len(X)), random_state=42)
    
    model_path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # --- CORE SHAP LOGIC ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sampled)
    
    # Generate high-resolution summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sampled, show=False)
    
    # Save to disk for the research paper
    plot_path = os.path.join(PLOTS_DIR, f"SHAP_Summary_{class_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved high-res research graph: {plot_path}")

# Generate visualizations for all 3 Experts
df_w = df[df['ProductCD'] == 'W']
generate_shap_for_expert(df_w, "W_Class", "xgboost_w_class.model")

df_c = df[df['ProductCD'] == 'C']
generate_shap_for_expert(df_c, "C_Class", "xgboost_c_class.model")

df_hrs = df[df['ProductCD'].isin(['H', 'R', 'S'])]
generate_shap_for_expert(df_hrs, "HRS_Class", "xgboost_hrs_class.model")

print(f"\n🎉 Done! All graphics exported to the {PLOTS_DIR} folder.")
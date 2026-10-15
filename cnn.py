import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import tf2onnx
import os

# 1. Configuration
DATA_PATH = "reports/lstm_sequential_tensors.csv"
MODEL_DIR = "models"
TIMESTEPS = 5
FEATURES_PER_STEP = 5

os.makedirs(MODEL_DIR, exist_ok=True)

print("=== FINFORGE 1D CNN TRAINING PIPELINE ===")

# 2. Data Loading & Preprocessing
print("Loading sequential data...")
df = pd.read_csv(DATA_PATH)

# Drop the CardID as neural networks cannot process raw ID strings
y = df['Target_isFraud'].values
X_flat = df.drop(columns=['CardID', 'Target_isFraud']).values

# Split data (Stratified to maintain fraud ratio)
X_train_flat, X_test_flat, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.15, stratify=y, random_state=42
)
X_train_flat, X_val_flat, y_train, y_val = train_test_split(
    X_train_flat, y_train, test_size=0.15, stratify=y_train, random_state=42
)

# 3. Scaling
# Neural networks require normalized inputs. We fit the scaler ONLY on training data to prevent leakage.
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_val_scaled = scaler.transform(X_val_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Save the scaler for Java Inference later
joblib.dump(scaler, os.path.join(MODEL_DIR, 'cnn_scaler.pkl'))

# 4. Tensor Reshaping [Samples, Timesteps, Features]
X_train = X_train_scaled.reshape(-1, TIMESTEPS, FEATURES_PER_STEP)
X_val = X_val_scaled.reshape(-1, TIMESTEPS, FEATURES_PER_STEP)
X_test = X_test_scaled.reshape(-1, TIMESTEPS, FEATURES_PER_STEP)

print(f"Train Tensor Shape: {X_train.shape}")
print(f"Validation Tensor Shape: {X_val.shape}")

# Calculate Class Weights to handle the massive normal-to-fraud imbalance
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = {classes[i]: weights[i] for i in range(len(classes))}
print(f"Applied Class Weights: {class_weights}")

# 5. Build the 1D Convolutional Neural Network
print("\nCompiling 1D CNN Architecture...")
model = models.Sequential([
    layers.InputLayer(input_shape=(TIMESTEPS, FEATURES_PER_STEP)),
    
    # First Convolutional Block: Detect low-level behavioral edges between adjacent transactions
    layers.Conv1D(filters=32, kernel_size=2, activation='relu', padding='valid'),
    layers.BatchNormalization(),
    
    # Second Convolutional Block: Detect higher-level patterns across the window
    layers.Conv1D(filters=64, kernel_size=2, activation='relu', padding='valid'),
    layers.BatchNormalization(),
    
    # Pool the most prominent signals
    layers.GlobalMaxPooling1D(),
    
    # Dense classification head
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Prevent overfitting
    layers.Dense(1, activation='sigmoid') # Final risk probability
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision')
    ]
)

# 6. Train the Model
print("\nTraining CNN...")
early_stop = EarlyStopping(monitor='val_pr_auc', mode='max', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# 7. Evaluate on Held-Out Test Set
print("\n=== FINAL TEST EVALUATION ===")
test_loss, test_pr_auc, test_recall, test_precision = model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Test PR-AUC:   {test_pr_auc:.4f}")
print(f"CNN Test Recall:   {test_recall*100:.2f}%")
print(f"CNN Test Precision:{test_precision*100:.2f}%")

# 8. Export to ONNX for Java Integration
print("\nExporting model to ONNX format...")
onnx_path = os.path.join(MODEL_DIR, "fraud_cnn.onnx")
spec = (tf.TensorSpec((None, TIMESTEPS, FEATURES_PER_STEP), tf.float32, name="input"),)
output_path, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_path)

print(f"Successfully saved ONNX model to: {output_path}")
print("The CNN is now ready to be loaded into the FinForge Java pipeline.")
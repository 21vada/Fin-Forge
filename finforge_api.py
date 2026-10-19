import os
import json
from contextlib import asynccontextmanager
import numpy as np
import xgboost as xgb
from google import genai
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Configuration & Paths ---
# --- 1. Configuration & Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

# Check main directory first; if not found, check the finforgejava subfolder
JSONL_LOG_PATH = os.path.join(CURRENT_DIR, "flagged_transactions.jsonl")
if not os.path.exists(JSONL_LOG_PATH):
    JSONL_LOG_PATH = os.path.join(CURRENT_DIR, "finforgejava", "flagged_transactions.jsonl")
# 🔑 MASTER CONFIGURATION 🔑
GEMINI_API_KEY = "paste your key here"  # <-- Paste your key here once
MODEL_ID = "gemini-3.8-flash"                # Valid, current GenAI model

experts = {}

# Initialize Google GenAI Client
ai_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_ACTUAL_API_KEY_HERE":
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("⚠️ WARNING: GEMINI_API_KEY not set in script. AI features disabled.")

# --- 2. Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Booting up FinForge API Server...")
    try:
        experts["W"] = xgb.XGBClassifier()
        experts["W"].load_model(os.path.join(MODELS_DIR, "xgboost_w_class.model"))
        
        experts["C"] = xgb.XGBClassifier()
        experts["C"].load_model(os.path.join(MODELS_DIR, "xgboost_c_class.model"))
        
        experts["HRS"] = xgb.XGBClassifier()
        experts["HRS"].load_model(os.path.join(MODELS_DIR, "xgboost_hrs_class.model"))
        print("✅ XGBoost Mixture of Experts successfully loaded.")
    except Exception as e:
        print(f"❌ Failed to load models: {str(e)}")
    yield
    experts.clear()

app = FastAPI(title="FinForge Fraud API & Chat", version="2.0", lifespan=lifespan)

# --- 3. Data Schemas ---
class TransactionRequest(BaseModel):
    TransactionID: str
    card1: str
    ProductCD: str
    TransactionAmt: float
    dist1: float
    C1: float
    C13: float
    TransformerVelocityScore: float

class ChatRequest(BaseModel):
    message: str

# --- 4. Helper Functions ---
def get_fraud_database_context():
    if not os.path.exists(JSONL_LOG_PATH):
        return "No transactions flagged yet."
    flagged_txs = []
    try:
        with open(JSONL_LOG_PATH, 'r') as f:
            for line in f:
                flagged_txs.append(json.loads(line.strip()))
        flagged_txs = sorted(flagged_txs, key=lambda x: x.get('XGBoostFinalScore', 0), reverse=True)[:20]
        return json.dumps(flagged_txs, indent=2)
    except Exception as e:
        return f"Error reading database: {str(e)}"

# --- 5. Transaction Analysis Endpoint ---
@app.post("/api/v1/analyze_transaction")
async def analyze_transaction(tx: TransactionRequest):
    if tx.ProductCD == "W":
        model, threshold = experts["W"], 0.979
    elif tx.ProductCD == "C":
        model, threshold = experts["C"], 0.983
    else:
        model, threshold = experts["HRS"], 0.983

    features = np.array([[tx.TransactionAmt, tx.dist1, tx.C1, tx.C13, tx.TransformerVelocityScore]])
    prob = float(model.predict_proba(features)[0][1])
    is_fraud = prob > threshold
    
    response = {
        "transaction_id": tx.TransactionID,
        "product_class": tx.ProductCD,
        "risk_score": round(prob, 4),
        "is_fraud": is_fraud,
        "action_taken": "BLOCKED" if is_fraud else "APPROVED",
        "xai_report": None
    }
    
    if is_fraud and ai_client:
        prompt = f"Write a 1-paragraph fraud report for TX {tx.TransactionID}. Class: {tx.ProductCD}, Amount: ${tx.TransactionAmt}, AI Risk Score: {prob}."
        try:
            report_res = ai_client.models.generate_content(model=MODEL_ID, contents=prompt)
            response["xai_report"] = report_res.text.strip()
        except Exception as e:
            response["xai_report"] = f"Report generation error: {str(e)}"
        
    return response

# --- 6. Copilot Chat Endpoint ---
@app.post("/api/v1/chat")
async def chat_with_finforge(chat: ChatRequest):
    if not ai_client:
        return {"response": "AI is disabled. Please set GEMINI_API_KEY in finforge_api.py."}
        
    db_context = get_fraud_database_context()
    prompt = f"""
    You are the FinForge Security Copilot. Here are the highest-risk transactions in the database:
    {db_context}
    The analyst asks: "{chat.message}"
    Provide a professional, concise response based ONLY on the transaction data provided above.
    """
    try:
        res = ai_client.models.generate_content(model=MODEL_ID, contents=prompt)
        return {"copilot_response": res.text.strip()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("finforge_api:app", host="127.0.0.1", port=8000, reload=True)
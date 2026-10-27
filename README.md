# 🛡️ FinForge: Hybrid Neural-Tree Fraud Detection & Generative XAI System

FinForge is an enterprise-grade real-time financial fraud detection framework. It combines a **Self-Attention Temporal Transformer** with an **XGBoost Mixture of Experts (MoE)** classifier to catch high-velocity anomaly patterns across millions of transactions with minimal false alarms.

The platform pairs a high-throughput **Java streaming ingestion engine** with an asynchronous **Python FastAPI microservice**, **TreeSHAP game-theoretic feature explanations**, and a **Gemini-powered Security Copilot** for natural language investigation reports and log querying.

---

## 🏗️ Architecture Overview

FinForge uses a multi-stage hybrid design to isolate noisy transaction categories while preserving high precision on high-risk transaction classes.


                        ┌─────────────────────────────────────────┐
                        │      IEEE-CIS Transaction Stream        │
                        └────────────────────┬────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
       ┌─────────────────────────────┐               ┌─────────────────────────────┐
       │ Temporal Transformer Model  │               │   XGBoost Mixture-of-       │
       │ (Sequence Velocity Scoring) │               │      Experts (MoE)          │
       └──────────────┬──────────────┘               └──────────────┬──────────────┘
                      │                                             │
                      │  TransformerVelocityScore                   │  ProductCD Routing
                      └──────────────────────┬──────────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ Java Streaming Core Pipeline│
                              │ (Custom Threshold Evaluator)│
                              └──────────────┬──────────────┘
                                             │ (Flags Anomalies)
                                             ▼
                              ┌─────────────────────────────┐
                              │ flagged_transactions.jsonl  │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │   FastAPI Backend + Gemini  │
                              │   (REST & XAI Copilot API)  │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ Streamlit Security Dashboard│
                              └─────────────────────────────┘

```

### Key Technical Pillars

1. **Temporal Velocity Extraction:** A Transformer layer evaluates sequence timing to generate a real-time `TransformerVelocityScore` for each incoming card profile.
2. **Mixture of Experts (MoE) Routing:** Transactions are dynamically routed to specialized sub-models based on `ProductCD`:
* **W Expert:** Dedicated to standard web purchases ($>0.979$ decision threshold).
* **C Expert:** Dedicated to commercial transactions ($>0.983$ decision threshold).
* **HRS Expert:** Dedicated to high-risk product categories ($>0.983$ decision threshold).


3. **High-Throughput Ingestion (Java Core):** Processes massive transaction CSV batches in real time, applies custom confidence thresholds, and streams flagged records to JSONL storage.
4. **Explainable AI (TreeSHAP & Gemini):** Combines game-theoretic SHAP feature contribution charts with Generative AI to automatically draft 3-paragraph incident investigation reports for blocked transactions.
5. **Interactive Copilot Chat:** An AI security assistant connected directly to the `flagged_transactions.jsonl` database for real-time natural language query resolution.

---

## 📊 Model Performance & Master Metrics

By applying high production thresholds ($0.979$ / $0.983$), FinForge optimizes for an **ultra-low friction environment**, eliminating false positives that freeze legitimate customer accounts.

### Fused Master Pipeline Performance

* **Master Accuracy:** `97.07%`
* **Master Precision (Fraud Class 1):** `89.27%`
* **Master AUC-ROC:** `0.8396`
* **False Alarms:** Only `97` false positives across `113,937` legitimate test transactions.

### Expert Sub-Model AUC-ROC

| Expert Sub-Model | Target Category | AUC-ROC Score |
| --- | --- | --- |
| **HRS Class Expert** | High-Risk Categories (`H`, `R`, `S`) | **`0.9101`** |
| **C Class Expert** | Commercial (`C`) | **`0.8918`** |
| **W Class Expert** | Standard Web (`W`) | **`0.7382`** |

---

## 📂 Project Directory Structure

```text
Fin-Forge/
├── ieee-fraud-detection/            # IEEE-CIS Fraud Detection dataset (CSVs)
├── models/                          # Trained XGBoost expert binaries (.model)
├── plots/                           # High-resolution TreeSHAP research charts
├── finforgejava/                    # Java high-throughput pipeline
│   ├── MasterFinForgePipeline.java  # Core Java CSV processing engine
│   └── flagged_transactions.jsonl   # Logged anomaly output database
├── async_analyst.py                 # Async Gemini XAI report generator
├── evaluate_master_pipeline.py      # Fused pipeline metrics & evaluation script
├── generate_shap_graphs.py          # TreeSHAP feature importance graph script
├── finforge_api.py                  # FastAPI REST microservice + Gemini Copilot
├── dashboard.py                     # Streamlit Security Center web dashboard
├── .gitignore                       # Git ignore configuration
└── README.md                        # Documentation

```

---

## 🛠️ Tech Stack & Dependencies

* **Primary Languages:** Python 3.10+, Java OpenJDK 17+
* **Machine Learning:** XGBoost, Scikit-Learn, PyTorch / Transformer
* **Explainability:** SHAP (TreeSHAP), Matplotlib
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Generative AI:** Google GenAI SDK (`google-genai` / `gemini-3.8-flash`)
* **Frontend UI:** Streamlit, Requests

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/21vada/Fin-Forge.git
cd Fin-Forge

```

### 2. Install Dependencies

```bash
pip install xgboost pandas numpy scikit-learn shap matplotlib fastapi uvicorn pydantic google-genai streamlit requests

```

### 3. Environment Setup

Configure your Gemini API key in `finforge_api.py` or set it as an environment variable:

```powershell
$env:GEMINI_API_KEY="your_google_gemini_api_key_here"

```

### 4. Evaluate Pipeline Metrics & Generate SHAP Graphs

Run the master evaluation script to evaluate pipeline metrics under production thresholds:

```bash
python evaluate_master_pipeline.py

```

To export high-resolution TreeSHAP feature importance graphs for your report:

```bash
python generate_shap_graphs.py

```

### 5. Start the FastAPI Microservice

Run the FastAPI backend server:

```bash
python finforge_api.py

```

* **Interactive API Documentation (Swagger UI):** `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`

### 6. Launch the Streamlit Dashboard

Open a second terminal window and run:

```bash
python -m streamlit run dashboard.py

```

* **Dashboard Interface:** `http://localhost:8501`

---

## 📡 API Endpoints Summary

### `POST /api/v1/analyze_transaction`

Routes an incoming transaction payload to the designated XGBoost Expert and returns risk probability, decision status, and an automated Gemini XAI incident report if blocked.

* **Sample Payload:**

```json
{
  "TransactionID": "TX-99841",
  "card1": "4512",
  "ProductCD": "C",
  "TransactionAmt": 4500.00,
  "dist1": 800.0,
  "C1": 150.0,
  "C13": 120.0,
  "TransformerVelocityScore": 0.99
}

```

### `POST /api/v1/chat`

Interacts with the FinForge Security Copilot. The endpoint reads the highest-risk anomaly logs directly from `flagged_transactions.jsonl` and feeds context to Gemini for natural language query resolution.

* **Sample Payload:**

```json
{
  "message": "What are the top 3 highest-risk fraud incidents currently logged in the system database?"
}

```

---

## 👨‍💻 Author

* **Balkrishna Sharma** — *Computer Applications & Machine Learning Systems*
* **Institution:** Manipal University

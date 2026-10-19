import streamlit as st
import requests

# --- Configuration ---
st.set_page_config(page_title="FinForge Dashboard", page_icon="🛡️", layout="wide")
API_URL = "http://127.0.0.1:8000/api/v1"

st.title("🛡️ FinForge Security Center")
st.markdown("Neural-Tree Fraud Detection & Generative AI Copilot")

tab1, tab2 = st.tabs(["⚡ Live Transaction Analysis", "🤖 Security Copilot"])

# --- TAB 1: Live API Testing ---
with tab1:
    st.header("Test a Transaction")
    with st.form("tx_form"):
        col1, col2, col3 = st.columns(3)
        tx_id = col1.text_input("Transaction ID", "TX-10045")
        prod = col2.selectbox("Product Class", ["W", "C", "H", "R", "S"], index=1)
        amt = col3.number_input("Amount ($)", value=4500.0)
        
        col4, col5, col6 = st.columns(3)
        dist = col4.number_input("Distance (dist1)", value=800.0)
        c1 = col5.number_input("C1 Feature", value=150.0)
        c13 = col6.number_input("C13 Feature", value=120.0)
        
        score = st.slider("Transformer Temporal Velocity Score", 0.0, 1.0, 0.99)
        submitted = st.form_submit_button("Run Analysis", type="primary")
        
        if submitted:
            payload = {
                "TransactionID": tx_id, 
                "card1": "0000", 
                "ProductCD": prod,
                "TransactionAmt": amt, 
                "dist1": dist, 
                "C1": c1, 
                "C13": c13,
                "TransformerVelocityScore": score
            }
            
            with st.spinner("Routing through XGBoost Experts..."):
                try:
                    res = requests.post(f"{API_URL}/analyze_transaction", json=payload).json()
                    
                    if res.get("is_fraud"):
                        st.error(f"🚨 FRAUD BLOCKED! Risk Score: {res.get('risk_score')}")
                        st.warning(f"**AI Investigator Report:**\n\n{res.get('xai_report')}")
                    else:
                        st.success(f"✅ APPROVED. Risk Score: {res.get('risk_score')}")
                except Exception:
                    st.error("❌ Failed to connect. Is your FastAPI backend running on port 8000?")

# --- TAB 2: Database Chatbot ---
with tab2:
    st.header("Ask the Copilot")
    st.info("The AI is actively monitoring the highest-risk anomalies in the `flagged_transactions.jsonl` database.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    # Chat Input Box
    if prompt := st.chat_input("Ask about top frauds, specific IDs, or current trends..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Investigating database..."):
            try:
                res = requests.post(f"{API_URL}/chat", json={"message": prompt}).json()
                reply = res.get("copilot_response", res.get("error", "AI Error."))
                
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.chat_message("assistant").write(reply)
            except Exception:
                st.error("❌ Failed to connect. Is your FastAPI backend running on port 8000?")
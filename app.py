import streamlit as st
import joblib
import pandas as pd

# --- 1. Load the AI Model ---
@st.cache_resource
def load_model():
    return joblib.load('aml_risk_model.pkl')

model = load_model()

# --- 2. Build the Web Interface ---
st.set_page_config(page_title="AML Risk Analyzer", page_icon="🏦")
st.title("🏦 Advanced AML Risk Analyzer")
st.write("Enter the transaction details below to run an instant AI risk assessment.")

st.markdown("---")

# --- 3. User Inputs ---
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount (INR)", min_value=1.0, value=10000.0, step=1000.0)
    hour = st.slider("Time of Transaction (24hr clock)", 0, 23, 14)

with col2:
    velocity = st.number_input("Account Velocity (Txns in last 24h)", min_value=1, value=2)

st.markdown("---")

# --- 4. The Analysis Engine ---
if st.button("Run Risk Analysis", use_container_width=True):
    with st.spinner("Analyzing behavioral patterns..."):
        
        # Apply the exact same Feature Engineering from Phase 2
        night_txn = 1 if hour < 5 else 0
        smurfing = 1 if 45000 <= amount < 50000 else 0
        
        # Package the data for the AI
        input_data = pd.DataFrame([[amount, night_txn, smurfing, velocity]], 
                                  columns=['Amount_INR', 'Night_Transaction', 'Smurfing_Flag', 'Account_Velocity'])
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # --- 5. Display Results ---
        if prediction[0] == 1:
            st.error("🚨 **HIGH RISK ALERT: Potential Money Laundering Detected!**")
            st.write("**Flagged Behaviors:**")
            if night_txn == 1: st.write("- 🌙 Suspicious late-night transaction hours.")
            if smurfing == 1: st.write("- ⚠️ Amount falls within known 'Smurfing' threshold.")
            if velocity > 5: st.write("- 🔄 Abnormally high transaction velocity for this account.")
        else:
            st.success("✅ **LOW RISK: Transaction appears normal.**")
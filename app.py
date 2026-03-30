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
        
        # Apply the exact same Feature Engineering
        night_txn = 1 if hour < 5 else 0
        smurfing = 1 if 45000 <= amount < 50000 else 0
        
        # Package the data
        input_data = pd.DataFrame([[amount, night_txn, smurfing, velocity]], 
                                  columns=['Amount_INR', 'Night_Transaction', 'Smurfing_Flag', 'Account_Velocity'])
        
        # UPGRADE: Get the exact probability score instead of a hard 0/1
        probabilities = model.predict_proba(input_data)
        
        # probabilities[0][1] gives us the percentage chance of class '1' (Laundering)
        risk_score = probabilities[0][1] * 100 
        
        # --- 5. Display Results with a Risk Meter ---
        st.markdown(f"### AI Risk Score: {risk_score:.2f}%")
        
        # Create a visual progress bar for the risk meter
        st.progress(int(risk_score))
        
        # We can set our own threshold now! If risk is over 40%, flag it for review.
        if risk_score > 40.0:
            st.error("🚨 **ALERT: Suspicious Behavior Detected. Route to Compliance Team.**")
            st.write("**Contributing Factors:**")
            if night_txn == 1: st.write("- 🌙 Occurred during high-risk hours (Midnight - 5 AM).")
            if smurfing == 1: st.write("- ⚠️ Amount sits just below reporting thresholds (Smurfing).")
            if velocity > 5: st.write("- 🔄 Unusually high transaction velocity.")
            if amount > 1000000: st.write("- 💰 Exceptionally high raw transfer volume.")
        else:
            st.success("✅ **LOW RISK: No significant behavioral anomalies detected.**")

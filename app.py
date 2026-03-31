import streamlit as st
import joblib
import pandas as pd

# --- 1. Load the AI Model ---
@st.cache_resource
def load_model():
    return joblib.load('aml_risk_model.pkl.gz')

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
        base_ai_score = probabilities[0][1] * 100 
        
        # --- THE HYBRID OVERRIDE ENGINE (TUNED FOR SMOTE AI) ---
        # Because our new SMOTE AI is much more sensitive, we reduce the manual penalties
        compliance_penalty = 0
        if smurfing == 1: compliance_penalty += 15.0  
        if night_txn == 1: compliance_penalty += 10.0 
        if velocity > 10: compliance_penalty += 15.0  
        
        # Calculate Final Score (Capped at 99%)
        final_risk_score = min(base_ai_score + compliance_penalty, 99.0)
        
        # --- 5. Display Results with a Risk Meter ---
        st.markdown(f"### Final AML Risk Score: {final_risk_score:.2f}%")
        st.caption(f"*(Base AI Score: {base_ai_score:.2f}% | Compliance Override Penalty: +{compliance_penalty:.2f}%)*")
        
        # Create a visual progress bar for the risk meter
        st.progress(int(final_risk_score) / 100.0)
        
        # Alert Threshold
        if final_risk_score >= 40.0:
            st.error("🚨 **ALERT: Suspicious Behavior Detected. Route to Compliance Team.**")
            st.write("**Contributing Factors:**")
            if night_txn == 1: st.write("- 🌙 Occurred during high-risk hours (Midnight - 5 AM).")
            if smurfing == 1: st.write("- ⚠️ Amount sits just below reporting thresholds (Smurfing).")
            if velocity > 5: st.write(f"- 🔄 Unusually high transaction velocity ({velocity} txns/24h).")
        else:
            st.success("✅ **LOW RISK: No significant behavioral anomalies detected.**")

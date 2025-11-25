import streamlit as st
import pandas as pd
from backend.risk_model import score_numeric_data, analyze_risk_with_llm

st.set_page_config(page_title="Xenber Track 2: AI Credit Risk", layout="wide")

st.title("🤖 AI Credit Risk Assessment (Multimodal)")
st.caption("Powered by LLM Fusion (Numeric + Text Analysis)")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Financial Profile")
    salary = st.number_input("Monthly Salary (RM)", value=5000)
    loan_amount = st.number_input("Loan Amount (RM)", value=20000)
    repayment_history = st.slider("Repayment History (10=Perfect)", 0, 10, 8)
    savings = st.number_input("Total Savings (RM)", value=10000)
    existing_debt = st.number_input("Existing Debt (RM)", value=500)

with col2:
    st.header("2. Loan Purpose")
    st.info("Try entering risky things like: 'I need to buy crypto fast' vs 'Renovating my house'")
    reason = st.text_area("Why do you need this loan?", height=150)
    
    assess_btn = st.button("🚀 Analyze Risk", type="primary")

if assess_btn:
    with st.spinner("Consulting AI Risk Engine..."):
        # 1. Prepare Data
        numeric_profile = {
            "salary": salary,
            "loan_amount": loan_amount,
            "repayment_history": repayment_history,
            "savings": savings,
            "existing_debt": existing_debt
        }
        
        # 2. Get Traditional Score (For comparison)
        base_score = score_numeric_data(salary, loan_amount, repayment_history, savings, existing_debt)
        
        # 3. Get LLM Fusion Score (The "Smart" Score)
        llm_result = analyze_risk_with_llm(numeric_profile, reason)
        
        final_score = llm_result['risk_score']
        decision = llm_result['decision']

    # --- RESULTS DISPLAY ---
    st.divider()
    
    # Score Cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Traditional Score", f"{base_score}/100", delta_color="inverse")
    c2.metric("AI-Adjusted Risk", f"{final_score}/100", 
              delta="-High Risk" if final_score > 60 else "Low Risk",
              delta_color="inverse")
    
    if decision == "ACCEPT":
        c3.success(f"✅ Recommendation: {decision}")
    else:
        c3.error(f"🛑 Recommendation: {decision}")

    # Explanation Section
    st.subheader("📝 AI Explanation")
    st.write(llm_result['explanation'])
    
    # Risk Factors Tags
    st.write("#### Detected Risk Factors:")
    if llm_result['risk_factors']:
        for factor in llm_result['risk_factors']:
            st.warning(f"⚠️ {factor}")
    else:
        st.success("No specific risk flags detected.")

    # Visualization (Judging Criteria: Prototype Quality)
    st.subheader("📊 Risk Distribution")
    chart_data = pd.DataFrame({
        "Method": ["Traditional Rule-Based", "AI Multimodal Fusion"],
        "Risk Score": [base_score, final_score]
    })
    st.bar_chart(chart_data, x="Method", y="Risk Score", color=["#FF4B4B"])
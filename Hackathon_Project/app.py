import streamlit as st
import pandas as pd
from backend.risk_model import analyze_risk_with_llm, score_numeric_data

st.set_page_config(page_title="Xenber Track 2: AI Credit Risk", layout="wide")

st.title("🤖 AI Credit Risk Assessment (Multimodal)")
st.caption("Upload a Bank Statement (CSV) to detect hidden behavioral risks.")

# --- SIDEBAR: API KEY ---
with st.sidebar:
    st.header("Configuration")
    if "HF_TOKEN" in st.secrets:
        st.success("API Key loaded from secrets")
    else:
        api_key = st.text_input("Enter Hugging Face Token", type="password")
        if api_key:
            st.session_state["HF_TOKEN"] = api_key

# --- MAIN INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Financial Documents")
    
    # NEW: File Uploader
    uploaded_file = st.file_uploader("Upload Bank Statement (CSV)", type=["csv"])
    
    # Default values (overwritten if CSV is uploaded)
    salary = st.number_input("Monthly Salary (RM)", value=5000)
    loan_amount = st.number_input("Loan Amount (RM)", value=20000)
    repayment_history = st.slider("Repayment History (10=Perfect)", 0, 10, 8)
    
    # Placeholders for calculated values
    savings = 0
    existing_debt = 0
    transaction_summary = "No CSV uploaded."

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(3), height=100) # Show preview
            
            # AUTOMATIC CALCULATION (Simple logic for demo)
            # Assuming CSV has columns: 'Description', 'Amount'
            # Positive amount = Income/Savings, Negative = Expense
            if 'Amount' in df.columns:
                savings = df[df['Amount'] > 0]['Amount'].sum()
                existing_debt = abs(df[df['Amount'] < 0]['Amount'].sum()) / 12 # Avg monthly spend
                
                st.info(f"💰 Auto-Calculated Savings: RM {savings:,.2f}")
                st.info(f"💸 Est. Monthly Spending: RM {existing_debt:,.2f}")
            
            # EXTRACT TRANSACTION NOTES for the LLM
            if 'Description' in df.columns:
                # We take the top 10 largest transactions or just list unique ones
                descriptions = df['Description'].unique()
                transaction_summary = ", ".join(descriptions[:20]) # Limit to 20 items
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

with col2:
    st.header("2. Loan Purpose")
    reason = st.text_area("Why do you need this loan?", height=150, 
                          placeholder="e.g., I need to renovate my house...")
    
    assess_btn = st.button("🚀 Analyze Risk", type="primary")

# --- EXECUTION ---
if assess_btn:
    # Check for API Key
    if "HF_TOKEN" not in st.session_state and "HF_TOKEN" not in st.secrets:
        st.error("Please enter an API Key in the sidebar first!")
    else:
        with st.spinner("Analyzing Financials + Behavioral Patterns..."):
            
            # Prepare Profile
            numeric_profile = {
                "salary": salary,
                "loan_amount": loan_amount,
                "repayment_history": repayment_history,
                "savings": savings if savings > 0 else 1000, # Fallback
                "existing_debt": existing_debt
            }
            
            # FUSION: Combine Loan Essay + Transaction Notes
            # This is the "Multimodal" part!
            full_text_context = f"Loan Essay: {reason}. \nRecent Transaction Notes: {transaction_summary}"
            
            # 1. Get Traditional Score
            base_score = score_numeric_data(salary, loan_amount, repayment_history, savings, existing_debt)
            
            # 2. Get LLM Score
            llm_result = analyze_risk_with_llm(numeric_profile, full_text_context)
            
            final_score = llm_result.get('risk_score', 50)
            decision = llm_result.get('decision', "Review")
            explanation = llm_result.get('explanation', "No details provided.")

        # --- DISPLAY RESULTS ---
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Traditional Score", f"{base_score}/100")
        m2.metric("AI-Adjusted Risk", f"{final_score}/100", 
                  delta="-High Risk" if final_score > 60 else "Low Risk", delta_color="inverse")
        if decision == "ACCEPT":
            m3.success(f"✅ {decision}")
        else:
            m3.error(f"🛑 {decision}")

        st.subheader("📝 AI Explanation")
        st.write(explanation)
        
        st.subheader("🔍 Hidden Behavioral Signals")
        st.write(f"**Analyzed Context:** {transaction_summary[:200]}...") 
        if llm_result.get('risk_factors'):
            for factor in llm_result['risk_factors']:
                st.warning(f"⚠️ {factor}")
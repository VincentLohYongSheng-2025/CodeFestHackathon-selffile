# backend/risk_model.py
import streamlit as st
from huggingface_hub import InferenceClient

# 1. Try to get the token from Streamlit's secret vault
#    If it fails (e.g., you forgot the file), it returns None
try:
    api_key = st.secrets["HF_TOKEN"]
except FileNotFoundError:
    api_key = None

# 2. Initialize the client securely
if api_key:
    client = InferenceClient(api_key=api_key)
else:
    # This prevents the app from crashing immediately if key is missing
    client = None

# ---------------- 1. TRADITIONAL NUMERIC SCORE (Baseline) ----------------
def score_numeric_data(salary, loan_amount, repayment_history, savings, existing_debt):
    """Compute a baseline numeric risk score (0-100) using traditional rules."""
    if salary == 0: salary = 1
    
    lti_ratio = loan_amount / (salary * 12)
    score = 0
    
    # Rule 1: Loan to Income
    if lti_ratio > 0.6: score += 50
    elif lti_ratio > 0.4: score += 30
    
    # Rule 2: Repayment History (0-10, where 10 is good)
    score += (10 - repayment_history) * 5
    
    # Rule 3: Savings Buffer
    if savings < (loan_amount * 0.1): score += 20
    
    return min(score, 100)

# ---------------- 2. LLM FUSION SCORE (Hugging Face Version) ----------------
def analyze_risk_with_llm(numeric_profile, loan_reason):
    """
    Sends BOTH numeric data and text reason to the Hugging Face LLM.
    """
    
    # We construct a prompt that forces the model to return JSON
    prompt = f"""
    [INST] You are a Credit Risk Officer. Analyze this loan application.
    
    APPLICANT DATA:
    - Salary: RM {numeric_profile['salary']}
    - Loan Request: RM {numeric_profile['loan_amount']}
    - Savings: RM {numeric_profile['savings']}
    - History Score: {numeric_profile['repayment_history']}/10
    
    REASON FOR LOAN:
    "{loan_reason}"
    
    INSTRUCTIONS:
    1. If the reason is gambling, crypto, or vague urgency -> HIGH RISK (Score > 80).
    2. If the financials are weak but reason is good -> MEDIUM RISK.
    3. Return valid JSON only. Format:
    {{
        "risk_score": <number 0-100>,
        "decision": "<ACCEPT or REJECT>",
        "risk_factors": ["<factor1>", "<factor2>"],
        "explanation": "<short explanation>"
    }}
    [/INST]
    """

    try:
        response = client.text_generation(
            prompt,
            model=repo_id,
            max_new_tokens=300,
            temperature=0.1, # Low temperature for consistent logic
            return_full_text=False
        )
        
        # Clean up response to ensure it's valid JSON
        # Sometimes models add text before/after the JSON, so we strip it
        clean_json = response.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0]
        elif "{" in clean_json:
             start_idx = clean_json.find("{")
             end_idx = clean_json.rfind("}") + 1
             clean_json = clean_json[start_idx:end_idx]

        return json.loads(clean_json)

    except Exception as e:
        print(f"Error: {e}")
        return {
            "risk_score": 50,
            "decision": "MANUAL CHECK",
            "risk_factors": ["AI Connection Error"],
            "explanation": "Could not reach Hugging Face API. Check internet/token."
        }
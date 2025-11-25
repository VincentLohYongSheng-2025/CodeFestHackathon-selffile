# backend/risk_model.py
import json
import streamlit as st
from huggingface_hub import InferenceClient

# ---------------- CONFIGURATION ----------------
# 1. Define the Model ID (This was the missing line!)
repo_id = "HuggingFaceH4/zephyr-7b-beta"

# 2. Setup the Client securely
# We try to get the token from secrets, or return None if missing
try:
    api_key = st.secrets["HF_TOKEN"]
except FileNotFoundError:
    try:
        api_key = st.session_state["HF_TOKEN"]
    except KeyError:
        api_key = None

if api_key:
    client = InferenceClient(token=api_key)
else:
    client = None

# ---------------- 1. TRADITIONAL NUMERIC SCORE ----------------
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

# ---------------- 2. LLM FUSION SCORE ----------------
def analyze_risk_with_llm(numeric_profile, loan_reason):
    """
    Sends BOTH numeric data and text reason to the Hugging Face LLM.
    """
    
    # Check if client is active
    if not client:
        return {
            "risk_score": 50,
            "decision": "ERROR",
            "risk_factors": ["API Key Missing"],
            "explanation": "Please enter your Hugging Face Token in the sidebar."
        }

    # Construct the prompt
    prompt = f"""
    <|system|>
    You are a Credit Risk Officer. Analyze this loan application strictly.
    output JSON format only.
    </s>
    <|user|>
    APPLICANT DATA:
    - Salary: RM {numeric_profile['salary']}
    - Loan Request: RM {numeric_profile['loan_amount']}
    - Savings: RM {numeric_profile['savings']}
    - History Score: {numeric_profile['repayment_history']}/10
    
    REASON / CONTEXT:
    "{loan_reason}"
    
    INSTRUCTIONS:
    1. If the reason mentions gambling, crypto, or vague urgency -> HIGH RISK (Score > 80).
    2. If the financials are weak but reason is good -> MEDIUM RISK.
    3. Return valid JSON only. Format:
    {{
        "risk_score": <number 0-100>,
        "decision": "<ACCEPT or REJECT>",
        "risk_factors": ["<factor1>", "<factor2>"],
        "explanation": "<short explanation>"
    }}
    </s>
    <|assistant|>
    """

    try:
        # Call the API using the global 'repo_id' variable
        response = client.text_generation(
            prompt,
            model=repo_id,  # <--- This is where your error was!
            max_new_tokens=300,
            temperature=0.1,
            return_full_text=False
        )
        
        # Clean JSON
        clean_json = response.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0]
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1]
        
        # Parse JSON
        return json.loads(clean_json)

    except Exception as e:
        # Return the specific error so you can debug
        return {
            "risk_score": 50,
            "decision": "MANUAL CHECK",
            "risk_factors": ["AI Connection Error"],
            "explanation": f"Detailed Error: {str(e)}"
        }

# backend/risk_model.py
import json
import streamlit as st
from huggingface_hub import InferenceClient

# ---------------- CONFIGURATION ----------------
# We stick with Zephyr because it is smart, but we will call it correctly now.
repo_id = "HuggingFaceH4/zephyr-7b-beta"

# Setup the Client securely
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
    
    if lti_ratio > 0.6: score += 50
    elif lti_ratio > 0.4: score += 30
    
    score += (10 - repayment_history) * 5
    if savings < (loan_amount * 0.1): score += 20
    
    return min(score, 100)

# ---------------- 2. LLM FUSION SCORE (Fixed for Chat Models) ----------------
def analyze_risk_with_llm(numeric_profile, loan_reason):
    """
    Sends data to Hugging Face using the correct 'Chat Completion' API.
    """
    
    if not client:
        return {
            "risk_score": 50,
            "decision": "ERROR",
            "risk_factors": ["API Key Missing"],
            "explanation": "Please enter your Hugging Face Token in the sidebar."
        }

    # New 'Chat' format (Messages list instead of one big string)
    messages = [
        {
            "role": "system", 
            "content": "You are a strict Credit Risk Officer. You analyze financial data and loan reasons to detect hidden risks. Output valid JSON only."
        },
        {
            "role": "user", 
            "content": f"""
            APPLICANT DATA:
            - Salary: RM {numeric_profile['salary']}
            - Loan Request: RM {numeric_profile['loan_amount']}
            - Savings: RM {numeric_profile['savings']}
            - History Score: {numeric_profile['repayment_history']}/10
            
            REASON & CONTEXT:
            "{loan_reason}"
            
            INSTRUCTIONS:
            1. Analyze the context. 'Casino', 'Crypto', 'Urgent Debt', 'Ah Long' are HIGH RISK.
            2. High savings + valid reason = LOW RISK.
            3. Return this JSON structure only:
            {{
                "risk_score": <int 0-100>,
                "decision": "<ACCEPT or REJECT>",
                "risk_factors": ["<factor1>", "<factor2>"],
                "explanation": "<short sentence>"
            }}
            """
        }
    ]

    try:
        # CHANGED: We now use chat_completion instead of text_generation
        response = client.chat_completion(
            messages=messages,
            model=repo_id,
            max_tokens=300,
            temperature=0.1
        )
        
        # Extract content from the chat response
        raw_content = response.choices[0].message.content
        
        # Clean JSON (remove ```json wrappers if the model adds them)
        clean_json = raw_content.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0]
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1]
            
        return json.loads(clean_json)

    except Exception as e:
        return {
            "risk_score": 50,
            "decision": "MANUAL CHECK",
            "risk_factors": ["AI Connection Error"],
            "explanation": f"Detailed Error: {str(e)}"
        }

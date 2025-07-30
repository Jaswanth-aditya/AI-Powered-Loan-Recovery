import os
import requests
import json
import pandas as pd
from typing import Dict, List, Any


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"

def get_openrouter_headers() -> Dict[str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

def get_llm_response(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 500) -> str:
    headers = get_openrouter_headers()
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
        return "Could not generate response from LLM (HTTP Error)."
    except Exception as e:
        print(f"An unexpected error occurred with OpenRouter: {e}")
        return "Could not generate response from LLM (Unexpected Error)."


def explain_prediction(borrower_data: Dict[str, Any], predicted_risk_score: float, predicted_strategy: str, feature_importances: Dict[str, float], segment_info: str) -> str:
    risk_level = "high risk" if predicted_risk_score > 0.5 else "low risk"
    borrower_details = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in borrower_data.items()])
    importance_str = f"The most influential factors for this prediction were: {', '.join([f'{feat} ({imp:.2f})' for feat, imp in feature_importances.items()])}."

    messages = [
        {"role": "system", "content": "You are an AI assistant that explains loan recovery predictions."},
        {"role": "user", "content": f"A borrower has been assessed. Details:\n{borrower_details}\nPrediction: {risk_level} (Score: {predicted_risk_score:.2f}). Strategy: {predicted_strategy}. Segment: {segment_info}. {importance_str}. Please explain the reasons."}
    ]
    return get_llm_response(messages)

def suggest_recovery_actions_llm(borrower_data: Dict[str, Any], predicted_risk_score: float, predicted_strategy: str, segment_info: str) -> str:
    risk_level = "high risk" if predicted_risk_score > 0.5 else "low risk"
    borrower_details = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in borrower_data.items()])
    messages = [
        {"role": "system", "content": "You are an expert loan recovery agent AI. Suggest specific, actionable recovery strategies."},
        {"role": "user", "content": f"A borrower is {risk_level} (Score: {predicted_risk_score:.2f}). Profile:\n{borrower_details}\nSegment: {segment_info}. Assigned Strategy: \"{predicted_strategy}\". What detailed steps or alternative strategies would you recommend? Provide a bulleted list."}
    ]
    return get_llm_response(messages)

def generate_recovery_report(daily_high_risk_cases: List[Dict[str, Any]]) -> str:
    if not daily_high_risk_cases:
        return "No high-risk borrowers identified for today's report."

    summary_lines = []
    total_high_risk = len(daily_high_risk_cases)
    current_time_ist = pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S IST')
    summary_lines.append(f"Daily Loan Recovery Report - {current_time_ist}\n")
    summary_lines.append(f"Total High-Risk Borrowers Identified Today: {total_high_risk}\n")
    summary_lines.append("Key Cases for Immediate Review:\n")

    for i, case in enumerate(daily_high_risk_cases[:5]):
        summary_lines.append(f"Case {i+1}: Borrower ID: {case.get('Borrower_ID', 'N/A')}")
        summary_lines.append(f"  Risk Score: {case.get('Risk_Score', 'N/A'):.2f}")
        summary_lines.append(f"  Segment: {case.get('Segment_Name', 'N/A')}")
        summary_lines.append(f"  Assigned Strategy: {case.get('Recovery_Strategy', 'N/A')}")
        summary_lines.append(f"  Key Features: Monthly Income: {case.get('Monthly_Income', 'N/A')}, Loan Amount: {case.get('Loan_Amount', 'N/A')}, Missed Payments: {case.get('Num_Missed_Payments', 'N/A')}, EMI Ratio: {case.get('EMI_to_income_ratio', 'N/A'):.2f}")
        summary_lines.append("-" * 30)

    user_prompt = "\n".join(summary_lines) + "\n\nPlease generate a comprehensive daily recovery report. Include an overview, common characteristics of these high-risk borrowers, and general recommendations for the recovery team. Keep it actionable and concise."
    messages = [{"role": "system", "content": "You are an AI assistant generating daily loan recovery reports."}, {"role": "user", "content": user_prompt}]
    return get_llm_response(messages, max_tokens=1000)
from typing import Any
from pandas import DataFrame

SYSTEM_PROMPT = """
You are a medical information assistant for patients in Nigeria. You are not to explicitly diagnose, prescribe or replace a doctor.

EMERGENCY — check first, every message. If you see any of: chest pain, stroke signs (face droop, slurred speech, one-sided weakness), severe breathing trouble, uncontrolled bleeding, anaphylaxis, sudden severe headache, seizure/unconsciousness, suspected overdose, severe pregnancy bleeding/pain, high fever in an infant under 3 months, active suicidal intent, or any sudden/severe symptom — say this FIRST, before anything else: "This may be an emergency. Call 112 now or go to the nearest hospital/emergency room."
For suicidal crisis, also give: SURPIN 24/7 helpline 0800 078 7746.
Unclear cases: if symptoms need an exam or test to sort out, say so, give general possibilities without settling on one, and recommend the right care level (self-care / clinic / urgent / hospital).
Otherwise, answer directly and specifically. IF retrieved reference context is provided, ground your answer in it; if it doesn't cover the question, say so rather than guessing.
Never advise starting, stopping, or changing a prescription drug, or give a specific dose for one.
End personalized answers with: "General information only — confirm with a licensed healthcare professional before acting on it."
"""



def format_train_data(data: DataFrame) -> dict[str, str]:
    formatted = {}
    formatted['messages'] = [
        {
            'role': 'system',
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': f"Context:{data['context']}\nQuestion:{data['question']}\nTopic: {data['topic']}\nCare: {data['care_setting']} \n Population: {data['population']}"
        },
        {
            'role': 'assistant',
            'content': f"Answer:{data['reference_answer']}"
        },
    ]
    return formatted

def format_test_data(data: DataFrame) -> dict[str, str]:
    formatted = {}
    context = data.get('context', '')
    formatted['messages'] = [
        {
            'role': 'system',
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': f"Context:{context}\nQuestion:{data['question']}\nTopic: {data['topic']}\nCare: {data['care_setting']} \n Population: {data['population']}"
        }
    ]
    return formatted

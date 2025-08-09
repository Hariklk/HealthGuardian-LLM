
from .models_ml import predict_batch
from .llm_client import ask_llm
import pandas as pd
import asyncio


def health_caution(user_constraints, activity, sleep_hours):
    reasons = []
    safe = True
    if user_constraints.get("heart_condition") and activity.lower() in ["sprint","high_intensity_run","hiit"]:
        safe = False
        reasons.append("Activity risky due to heart condition.")
    if sleep_hours is not None and sleep_hours < 5 and activity.lower() in ["heavy_lifting","high_intensity_run"]:
        safe = False
        reasons.append("Very low sleep; recommend low intensity or rest.")
    if user_constraints.get("pregnant") and activity.lower() in ["heavy_lifting","sauna"]:
        safe = False
        reasons.append("Certain activities are not recommended during pregnancy.")
    return safe, reasons

async def generate_recommendation(user_doc, req):
    
    candidates = req.candidates or [360,420,480,540]
    rows = []
    for t in candidates:
        rows.append({
            "activity": req.activity,
            "scheduled_start_min": t,
            "scheduled_duration_min": req.scheduled_duration_min,
            "sleep_hours": req.sleep_hours or user_doc.get("sleep_hours", 7),
            "mood": req.mood or 3,
            "previous_day_completed_rate": user_doc.get("prev_completed_rate", 0.5),
            "location": user_doc.get("location","home"),
            "weather": user_doc.get("weather","clear"),
            "weekday": 0,
            "steps": user_doc.get("steps", 0),
            "calories": user_doc.get("calories", 2000)
        })
    df = pd.DataFrame(rows)
    probs = predict_batch(df)
    best_idx = int(probs.argmax())
    best_time = candidates[best_idx]
    best_prob = float(probs[best_idx])

    
    safe, reasons = health_caution(user_doc.get("health_constraints", {}), req.activity, req.sleep_hours)
    explanation = ""
    if safe:
        prompt = (f"User info: age {user_doc.get('birth_year','?')}, constraints {user_doc.get('health_constraints',{})}. "
                  f"Recommend safe, general, non-diagnostic activity tips for {req.activity} at time {best_time} minutes. Keep it short.")
        explanation = await ask_llm(prompt)
    else:
        explanation = "Recommendation blocked for safety: " + "; ".join(reasons) + " Please consult a healthcare professional."

    return {
        "best_time": best_time,
        "predicted_completion_prob": best_prob,
        "safe": safe,
        "reasons": reasons,
        "explanation": explanation,
        "all_candidates": [{"time":c, "prob":float(p)} for c,p in zip(candidates, probs)]
    }


import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "models/adherence_model.h5")
PREP_PATH = os.getenv("PREP_PATH", "models/preprocessor.joblib")

# lazy load
_model = None
_pre = None

def _load():
    global _model, _pre
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    if _pre is None:
        _pre = joblib.load(PREP_PATH)

def score_candidates(feature_rows):
    """
    feature_rows: list of dicts with columns matching training features (num + cat)
    returns: list of probabilities (floats)
    """
    import pandas as pd
    _load()
    df = pd.DataFrame(feature_rows)
    # ensure columns present (fill defaults)
    for c in ["location","weather","weekday","steps","calories","mood","previous_day_completed_rate","sleep_hours"]:
        if c not in df.columns:
            df[c] = 0 if c in ["steps","calories","weekday"] else ""
    # numerical defaults
    df.fillna({"sleep_hours":7,"mood":3,"previous_day_completed_rate":0.5,"steps":0,"calories":2000}, inplace=True)
    Xc = _pre.transform(df)
    probs = _model.predict(Xc).flatten()
    return probs

# simple health-caution rule layer
def health_caution(user_constraints, activity, sleep_hours):
    safe = True
    reasons = []
    if not isinstance(user_constraints, dict):
        user_constraints = {}
    if user_constraints.get("heart_condition") and activity.lower() in ["hiit","sprint","heavy_lifting"]:
        safe = False
        reasons.append("Heart condition — avoid high-intensity activities.")
    if user_constraints.get("pregnant") and activity.lower() in ["heavy_lifting","sauna"]:
        safe = False
        reasons.append("Pregnancy — avoid heavy lifting/sauna.")
    if sleep_hours is not None and sleep_hours < 5 and activity.lower() in ["hiit","heavy_lifting"]:
        safe = False
        reasons.append("Very low sleep — recommend rest or low intensity.")
    return safe, reasons

# optional LLM integration
def get_explanation_llm(user_profile, activity, best_time, best_prob, reasons, llm_client=None):
    """
    llm_client(prompt) -> string (async or sync). If None, returns non-LLM explanation.
    """
    if not llm_client:
        # simple local explanation
        if not reasons:
            return f"Recommended at {best_time} minutes with estimated completion probability {best_prob:.2f}. Start with a short warm-up."
        else:
            return " ".join(["Safety:"] + reasons)
    # if llm_client provided, call it (support sync or async)
    prompt = f"""User profile: {user_profile}
Activity: {activity}
Best time (minutes): {best_time}
Predicted completion probability: {best_prob:.2f}
Safety reasons if any: {reasons}
Give a short, friendly, non-medical recommendation. If unsafe, advise consulting healthcare professional."""
    # allow llm_client to be async or sync
    try:
        res = llm_client(prompt)
        # if coroutine
        if hasattr(res, "__await__"):
            import asyncio
            res = asyncio.get_event_loop().run_until_complete(res)
        return res
    except Exception as ex:
        return f"(LLM call failed) {str(ex)}"

def recommend(user_profile, activity, candidates_minutes, scheduled_duration_min=30, sleep_hours=None, mood=3, llm_client=None):
    """
    user_profile: dict with user info and health_constraints
    candidates_minutes: list of candidate start times
    Returns best recommendation dict
    """
    rows = []
    for t in candidates_minutes:
        rows.append({
            "activity": activity,
            "scheduled_start_min": t,
            "scheduled_duration_min": scheduled_duration_min,
            "sleep_hours": sleep_hours if sleep_hours is not None else user_profile.get("sleep_hours", 7),
            "mood": mood,
            "previous_day_completed_rate": user_profile.get("prev_completed_rate", 0.5),
            "location": user_profile.get("location","home"),
            "weather": user_profile.get("weather","clear"),
            "weekday": user_profile.get("weekday", 0),
            "steps": user_profile.get("steps", 0),
            "calories": user_profile.get("calories", 2000)
        })
    probs = score_candidates(rows)
    best_idx = int(np.argmax(probs))
    best_time = candidates_minutes[best_idx]
    best_prob = float(probs[best_idx])
    safe, reasons = health_caution(user_profile.get("health_constraints", {}), activity, sleep_hours)
    explanation = get_explanation_llm(user_profile, activity, best_time, best_prob, reasons, llm_client=llm_client)
    return {
        "best_time": best_time,
        "best_prob": best_prob,
        "safe": safe,
        "safety_reasons": reasons,
        "explanation": explanation,
        "candidates": [{"time":c,"prob":float(p)} for c,p in zip(candidates_minutes,probs)]
    }

# run_demo.py
from ml.synthetic_data import gen_rows
import os, pandas as pd
# 1) generate data
os.makedirs("ml/data", exist_ok=True)
df = gen_rows(4000)
df.to_csv("ml/data/sample_data.csv", index=False)
print("Generated sample data")

# 2) train
print("Training model...")
os.system("python ml/train.py")

# 3) run recommender
from src.models_infer import recommend
from src.llm_stub import llm_client_sync

user_profile = {
    "name":"Demo User",
    "birth_year":1990,
    "health_constraints": {"heart_condition": False},
    "prev_completed_rate": 0.6,
    "sleep_hours": 6.5,
    "location":"home",
    "weekday":2,
    "steps":1200,
    "calories":2100
}
res = recommend(user_profile, activity="exercise", candidates_minutes=[360,420,480,540], scheduled_duration_min=30, sleep_hours=6.5, mood=3, llm_client=llm_client_sync)
print("Recommendation result:")
import json
print(json.dumps(res, indent=2))

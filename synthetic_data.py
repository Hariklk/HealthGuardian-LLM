import pandas as pd
import numpy as np
import os

def gen_rows(n=5000, seed=42):
    np.random.seed(seed)
    acts = ["exercise","meditation","walking","yoga","heavy_lifting","hiit"]
    rows = []
    for i in range(n):
        activity = np.random.choice(acts, p=[0.25,0.15,0.2,0.15,0.15,0.1])
        start = np.random.choice([360,420,480,540,600])  # minutes since midnight
        duration = np.random.choice([10,20,30,45,60])
        sleep = float(np.clip(np.random.normal(7,1.4), 3, 11))
        mood = int(np.clip(np.round(np.random.normal(3.5,1)),1,5))
        prev = float(np.random.beta(2,2))
        # simple rule for completed: more likely if prev high and sleep ok and not very demanding
        difficulty = 1 if activity in ["hiit","heavy_lifting"] else 0
        prob = 0.2 + 0.5*prev + 0.15*(sleep>6) - 0.2*difficulty + np.random.normal(0,0.05)
        completed = 1 if np.random.rand() < np.clip(prob, 0, 0.98) else 0
        # user constraints: randomly attach some constraints to "users" later; here keep fields
        rows.append({
            "activity": activity,
            "scheduled_start_min": start,
            "scheduled_duration_min": duration,
            "sleep_hours": sleep,
            "mood": mood,
            "previous_day_completed_rate": prev,
            "completed": completed,
            "location": "home",
            "weather": "clear",
            "weekday": int(np.random.randint(0,7)),
            "steps": int(np.clip(np.random.normal(3000,1500),0,30000)),
            "calories": int(np.clip(np.random.normal(2200,300),1200,4000))
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs("ml/data", exist_ok=True)
    df = gen_rows(8000)
    df.to_csv("ml/data/sample_data.csv", index=False)
    print("Saved ml/data/sample_data.csv (rows=%d)"%len(df))

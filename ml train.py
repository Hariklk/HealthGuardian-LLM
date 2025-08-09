
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from tensorflow.keras import layers, models, callbacks


def gen_row(n=5000):
    acts = ["exercise","meditation","walking","yoga","heavy_lifting"]
    rows=[]
    for i in range(n):
        activity=np.random.choice(acts)
        start=np.random.choice([360,420,480,540,600])
        duration=np.random.choice([10,20,30,45])
        sleep=np.clip(np.random.normal(7,1.5),3,10)
        mood=np.random.randint(1,6)
        prev=np.random.beta(2,2)
        completed = 1 if (prev>0.5 and sleep>5 and np.random.rand() < 0.8) or np.random.rand()<0.2 else 0
        rows.append([activity,start,duration,sleep,mood,prev,completed,"home","clear",0,2000])
    cols = ["activity","scheduled_start_min","scheduled_duration_min","sleep_hours","mood","previous_day_completed_rate","completed","location","weather","weekday","calories"]
    return pd.DataFrame(rows, columns=cols)

df = gen_row()
num_features = ['scheduled_start_min','scheduled_duration_min','sleep_hours','mood','previous_day_completed_rate','calories']
cat_features = ['activity','location','weather']

X = df[num_features+cat_features]
y = df['completed']

num_pipe = Pipeline([("scaler", StandardScaler())])
cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
pre = ColumnTransformer([("num", num_pipe, num_features), ("cat", cat_pipe, cat_features)])
Xp = pre.fit_transform(X)

X_train,X_val,y_train,y_val = train_test_split(Xp,y,test_size=0.2,random_state=42)
input_dim = X_train.shape[1]
model = models.Sequential([layers.Input(shape=(input_dim,)), layers.Dense(128,activation="relu"), layers.Dropout(0.3),
                          layers.Dense(64,activation="relu"), layers.Dense(1,activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC","accuracy"])
es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=30, batch_size=64, callbacks=[es])
import os
os.makedirs("backend/models", exist_ok=True)
model.save("backend/models/adherence_model.h5")
joblib.dump(pre, "backend/models/preprocessor.joblib")
print("Saved model & preprocessor")

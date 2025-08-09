# ml/train.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from tensorflow.keras import layers, models, callbacks

DATA = "ml/data/sample_data.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(path=DATA):
    df = pd.read_csv(path)
    # minimal cleaning / fill
    df.fillna({"sleep_hours": df["sleep_hours"].median(), "mood":3, "previous_day_completed_rate":0.5}, inplace=True)
    return df

def build_preprocessor(num_features, cat_features):
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    pre = ColumnTransformer([("num", num_pipe, num_features), ("cat", cat_pipe, cat_features)])
    return pre

def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC","accuracy"])
    return model

if __name__ == "__main__":
    df = load_data()
    num_features = ['scheduled_start_min','scheduled_duration_min','sleep_hours','mood','previous_day_completed_rate','steps','calories']
    cat_features = ['activity','location','weather','weekday']
    X = df[num_features + cat_features]
    y = df['completed'].astype(int)

    pre = build_preprocessor(num_features, cat_features)
    Xp = pre.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(Xp, y, test_size=0.15, random_state=42, stratify=y)
    model = build_model(X_train.shape[1])

    es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=50, batch_size=64, callbacks=[es])

    # save
    model.save(os.path.join(OUT_DIR, "adherence_model.h5"))
    joblib.dump(pre, os.path.join(OUT_DIR, "preprocessor.joblib"))
    print("Saved model and preprocessor to", OUT_DIR)

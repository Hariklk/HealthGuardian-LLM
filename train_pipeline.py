# train_pipeline.py
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os

# Keras imports
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------- Load data ----------
df = pd.read_csv("ml/data/sample_data.csv")  # your CSV
df.fillna({"sleep_hours": df["sleep_hours"].median(), "mood":3, "previous_day_completed_rate":0.5}, inplace=True)

# ---------- Feature lists ----------
num_features = ['scheduled_start_min','scheduled_duration_min','sleep_hours','mood','previous_day_completed_rate','steps','calories']
cat_features = ['activity','location','weather','weekday']
target = 'completed'

X = df[num_features + cat_features]
y = df[target].astype(int)

# ---------- Preprocessor ----------
num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
cat_pipe = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='unknown')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
preprocessor = ColumnTransformer([('num', num_pipe, num_features), ('cat', cat_pipe, cat_features)], remainder='drop')

# Fit preprocessor and transform
Xp = preprocessor.fit_transform(X)

# Save preprocessor
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, "models/preprocessor.joblib")

# ---------- Baseline (scikit-learn) ----------
X_train, X_val, y_train, y_val = train_test_split(Xp, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
yval_prob = rf.predict_proba(X_val)[:,1]
print("RandomForest AUC:", roc_auc_score(y_val, yval_prob))
joblib.dump(rf, "models/rf_baseline.joblib")

# ---------- Keras model using same preprocessed data ----------
def build_keras(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

input_dim = Xp.shape[1]
model = build_keras(input_dim)
es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(Xp, y, validation_split=0.15, epochs=50, batch_size=64, callbacks=[es])
# Save Keras model
model.save("models/adherence_model.h5")

print("Saved preprocessor, rf baseline, and Keras model to models/")

# Minimal wrapper to predict using a saved keras model + preprocessor.
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = os.getenv("MODEL_PATH", "models/adherence_model.h5")
PREPROC_PATH = os.getenv("PREPROC_PATH", "models/preprocessor.joblib")

_model = None
_preproc = None

def load_artifacts():
    global _model, _preproc
    if _model is None:
        _model = load_model(MODEL_PATH)
    if _preproc is None:
        _preproc = joblib.load(PREPROC_PATH)

def predict_batch(feature_df):
    """
    feature_df: pandas.DataFrame with the same columns used in training
    returns: numpy array of probabilities
    """
    import pandas as pd
    load_artifacts()
    Xc = _preproc.transform(feature_df)
    return _model.predict(Xc).flatten()

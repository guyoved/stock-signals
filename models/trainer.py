"""
Model training & inference – LightGBM classifier (free & strong).
"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from config.settings import MODELS_DIR, MODEL_NAME
from features.engineering import add_technical_features, prepare_ml_dataset, get_feature_columns

logger = logging.getLogger(__name__)


def train_lightgbm(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[lgb.LGBMClassifier, dict[str, Any]]:
    """
    Train a LightGBM binary classifier on the given OHLCV+features DataFrame.
    Returns (model, metrics_dict)
    """
    df_feat = add_technical_features(df)
    X, y = prepare_ml_dataset(df_feat)

    if len(X) < 100:
        raise ValueError(f"Not enough samples after feature engineering: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": dict(zip(X.columns, model.feature_importances_.tolist())),
        "trained_at": datetime.utcnow().isoformat(),
    }

    logger.info(f"Model trained – Accuracy: {metrics['accuracy']:.3f}, AUC: {metrics['roc_auc']:.3f}")
    return model, metrics


def save_model(model: lgb.LGBMClassifier, metrics: dict, name: str = MODEL_NAME) -> Path:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump({"model": model, "metrics": metrics, "features": list(model.feature_name_)}, path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(name: str = MODEL_NAME) -> Optional[dict]:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        logger.warning(f"Model file not found: {path}")
        return None
    return joblib.load(path)


def predict_proba(model_data: dict, df: pd.DataFrame) -> Optional[float]:
    """
    Return probability of upward move for the *latest* row.
    """
    model: lgb.LGBMClassifier = model_data["model"]
    feature_names = model_data.get("features") or model.feature_name_

    df_feat = add_technical_features(df)
    if df_feat.empty:
        return None

    latest = df_feat.iloc[[-1]]
    # Align columns
    missing = set(feature_names) - set(latest.columns)
    for m in missing:
        latest[m] = 0.0
    X = latest[feature_names].fillna(0)

    try:
        proba = model.predict_proba(X)[0, 1]
        return float(proba)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None


def get_signal_from_proba(proba: float, min_confidence: float = 0.58) -> str:
    """Convert probability to BUY / SELL / HOLD."""
    if proba is None:
        return "HOLD"
    if proba >= min_confidence:
        return "BUY"
    if proba <= (1 - min_confidence):
        return "SELL"
    return "HOLD"

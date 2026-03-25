"""
models.py
---------
Defines, trains, and returns the three classifiers used in this project:

  1. Baseline  – always predicts the majority class ("always up")
  2. Logistic Regression – linear baseline with interpretable coefficients
  3. Random Forest – ensemble that captures nonlinear patterns

All models expose a sklearn-compatible interface (fit / predict / predict_proba).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin


# ---------------------------------------------------------------------------
# Baseline classifier
# ---------------------------------------------------------------------------

class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the majority class seen during training (simplest possible baseline)."""

    def fit(self, X, y):
        counts = np.bincount(y)
        self.majority_class_ = int(np.argmax(counts))
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, 2))
        proba[:, self.majority_class_] = 1.0
        return proba


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_logistic_regression(random_state: int = 42) -> Pipeline:
    """
    Logistic Regression wrapped in a StandardScaler pipeline.
    max_iter=1000 ensures convergence; C=0.1 provides mild regularisation.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """
    Random Forest classifier.
    n_estimators=200, max_depth=6 balances complexity vs. overfitting.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Chronological train / validation / test split
# ---------------------------------------------------------------------------

def chronological_split(
    feat_df: pd.DataFrame,
    feature_cols: list,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    """
    Split a feature DataFrame chronologically.

    Returns a dict with keys:
      X_train, y_train,
      X_val,   y_val,
      X_test,  y_test,
      dates_train, dates_val, dates_test
    """
    n = len(feat_df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train = feat_df.iloc[:n_train]
    val   = feat_df.iloc[n_train : n_train + n_val]
    test  = feat_df.iloc[n_train + n_val :]

    return {
        "X_train": train[feature_cols].values,
        "y_train": train["Target"].values,
        "X_val":   val[feature_cols].values,
        "y_val":   val["Target"].values,
        "X_test":  test[feature_cols].values,
        "y_test":  test["Target"].values,
        "dates_train": train.index,
        "dates_val":   val.index,
        "dates_test":  test.index,
        "test_df":     test,      # full test slice (needed for backtesting)
    }


# ---------------------------------------------------------------------------
# Train all models on training data
# ---------------------------------------------------------------------------

def train_all_models(X_train, y_train) -> dict:
    """
    Fit all three models and return a dict of trained estimators.
    """
    models = {
        "Baseline (Majority Class)": MajorityClassBaseline(),
        "Logistic Regression":       build_logistic_regression(),
        "Random Forest":             build_random_forest(),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  Trained: {name}")
    return models

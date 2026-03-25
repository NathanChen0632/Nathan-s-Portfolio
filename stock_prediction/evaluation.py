"""
evaluation.py
-------------
Evaluates trained models and produces:
  - Classification report (accuracy, precision, recall, F1)
  - Confusion matrix (printed + saved as PNG)
  - Feature importance plot for Random Forest
  - Logistic Regression coefficient plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

OUTPUT_DIR = "results"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(name: str, model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute and print classification metrics for one model on one dataset.
    Returns a metrics dict.
    """
    y_pred = model.predict(X)

    metrics = {
        "accuracy":  accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall":    recall_score(y, y_pred, zero_division=0),
        "f1":        f1_score(y, y_pred, zero_division=0),
    }

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print()
    print(classification_report(y, y_pred, target_names=["Down (0)", "Up (1)"]))

    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(name: str, model, X: np.ndarray, y: np.ndarray):
    _ensure_output_dir()
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()

    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    path = os.path.join(OUTPUT_DIR, f"cm_{safe_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved confusion matrix → {path}")


# ---------------------------------------------------------------------------
# Feature importance (Random Forest)
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_cols: list, top_n: int = 15):
    _ensure_output_dir()

    # Unwrap Pipeline if needed
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "feature_importances_"):
        return

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_cols[i] for i in idx][::-1],
        importances[idx][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Top Feature Importances")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "rf_feature_importance.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved feature importance plot → {path}")


# ---------------------------------------------------------------------------
# Logistic Regression coefficients
# ---------------------------------------------------------------------------

def plot_lr_coefficients(model, feature_cols: list):
    _ensure_output_dir()

    # Unwrap Pipeline
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "coef_"):
        return

    coefs = clf.coef_[0]
    idx = np.argsort(np.abs(coefs))[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in coefs[idx][::-1]]
    ax.barh(
        [feature_cols[i] for i in idx][::-1],
        coefs[idx][::-1],
        color=colors,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient value")
    ax.set_title("Logistic Regression — Feature Coefficients")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "lr_coefficients.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved LR coefficients plot → {path}")


# ---------------------------------------------------------------------------
# Summary comparison table
# ---------------------------------------------------------------------------

def print_summary_table(results: dict):
    """
    results: {model_name: {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}}
    """
    print("\n" + "=" * 65)
    print(f"  {'Model':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("=" * 65)
    for name, m in results.items():
        print(
            f"  {name:<35} {m['accuracy']:>6.3f} {m['precision']:>6.3f}"
            f" {m['recall']:>6.3f} {m['f1']:>6.3f}"
        )
    print("=" * 65)

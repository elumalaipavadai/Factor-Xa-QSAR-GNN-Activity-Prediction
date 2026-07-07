"""
metrics.py — evaluation metrics for regression (Factor Xa potency) and
multi-task classification (Tox21).

VALIDATED with scikit-learn + scipy.
"""
import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, roc_auc_score)
from scipy.stats import spearmanr


def regression_metrics(y_true, y_pred):
    """Potency regression: RMSE, MAE, R2, and Spearman rho (rank corr matters
    most for triage / ranking compounds)."""
    return {
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def multitask_auroc(y_true, y_score, mask):
    """Tox21-style multi-task AUROC.
    y_true, y_score, mask are (N, T) arrays; mask=1 where a label is present.
    Skips tasks with no positives/negatives in the present subset.
    """
    per_task = []
    for t in range(y_true.shape[1]):
        m = mask[:, t].astype(bool)
        if m.sum() > 0 and len(np.unique(y_true[m, t])) > 1:
            per_task.append(roc_auc_score(y_true[m, t], y_score[m, t]))
    return {"mean_AUROC": float(np.mean(per_task)) if per_task else float("nan"),
            "per_task_AUROC": per_task}

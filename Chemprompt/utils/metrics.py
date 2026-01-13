from sklearn.metrics import (
    f1_score, accuracy_score, mean_squared_error, r2_score, precision_score, recall_score
)
from scipy.stats import pearsonr, spearmanr
import numpy as np

def get_metric_function(metric_name):
    """
    Returns the metric function based on user-specified metric name.

    Args:
        metric_name (str): Name of the metric (e.g., 'f1_micro', 'accuracy', 'rmse', 'r2', 'pearson')

    Returns:
        function: Metric function
    """
    metric_mapping = {
        # Classification Metrics
        "f1_micro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
        "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
        "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
        "precision_micro": lambda y_true, y_pred: precision_score(y_true, y_pred, average="micro"),
        "precision_macro": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro"),
        "recall_micro": lambda y_true, y_pred: recall_score(y_true, y_pred, average="micro"),
        "recall_macro": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro"),

        # Regression Metrics
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": lambda y_true, y_pred: r2_score(y_true, y_pred),
        "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        "mse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),

        # Correlation Metrics
        "pearson": lambda y_true, y_pred: pearsonr(y_true.flatten(), y_pred.flatten())[0],
        "spearman": lambda y_true, y_pred: spearmanr(y_true.flatten(), y_pred.flatten())[0],
    }

    if metric_name in metric_mapping:
        return metric_mapping[metric_name]
    else:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Available metrics: {list(metric_mapping.keys())}"
        )

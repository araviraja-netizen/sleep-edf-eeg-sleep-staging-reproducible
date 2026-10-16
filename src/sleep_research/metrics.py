"""Five-class metrics and calibration/ROC/PR inputs using installed NumPy only."""

import numpy as np

CLASS_NAMES = ("Wake", "N1", "N2", "N3", "REM")


def validate_predictions(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if (labels.ndim != 1 or not len(labels) or probabilities.shape != (len(labels), 5)
            or not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 4))
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("Expected integer labels 0..4 and finite normalized five-class probabilities")
    return labels.astype(np.int64), probabilities


def confusion_matrix(labels, predicted):
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    if labels.shape != predicted.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Aligned nonempty true/predicted label vectors required")
    if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(predicted.dtype, np.integer):
        raise ValueError("Labels must be integers")
    if np.any((labels < 0) | (labels > 4) | (predicted < 0) | (predicted > 4)):
        raise ValueError("Invalid class label")
    matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    return matrix


def classification_metrics(labels, probabilities, bins=10):
    labels, probabilities = validate_predictions(labels, probabilities)
    if bins < 1:
        raise ValueError("Calibration bin count must be positive")
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predicted)
    support, predicted_support = matrix.sum(axis=1), matrix.sum(axis=0)
    tp = np.diag(matrix)
    precision = np.divide(tp, predicted_support, out=np.zeros(5, dtype=float), where=predicted_support > 0)
    recall = np.divide(tp, support, out=np.zeros(5, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5, dtype=float), where=precision + recall > 0)
    accuracy = float(tp.sum() / len(labels))
    expected = float(np.dot(support.astype(float), predicted_support.astype(float)) / len(labels) ** 2)
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else None
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    calibration = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selection = (confidence >= lower) & ((confidence <= upper) if index == bins - 1 else (confidence < upper))
        count = int(selection.sum())
        conf = float(confidence[selection].mean()) if count else None
        acc = float(correct[selection].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(acc - conf)
        calibration.append({"lower": lower, "upper": upper, "count": count, "confidence": conf, "accuracy": acc})
    one_hot = np.eye(5)[labels]
    return {"sample_count": len(labels), "accuracy": accuracy,
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "weighted_f1": float(np.dot(support, f1) / len(labels)),
            "balanced_accuracy": float(recall[support > 0].mean()), "cohen_kappa": kappa,
            "log_loss": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
            "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "expected_calibration_error": ece, "calibration_bins": calibration,
            "confusion_matrix": matrix.tolist(),
            "classes": {name: {"label": label, "support": int(support[label]), "precision": float(precision[label]),
                               "recall": float(recall[label]), "f1": float(f1[label])} for label, name in enumerate(CLASS_NAMES)},
            "conventions": {"macro_classes": "all five; absent-class values are zero",
                            "balanced_accuracy": "mean recall over classes present in ground truth",
                            "brier": "mean sum of five squared probability errors",
                            "undefined_kappa": "null when chance agreement is one"}}


def binary_curves(positive, scores):
    positive, scores = np.asarray(positive, dtype=bool), np.asarray(scores, dtype=float)
    if positive.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Aligned finite ROC/PR inputs required")
    positives, negatives = int(positive.sum()), int((~positive).sum())
    if positives == 0 or negatives == 0:
        return {"defined": False, "reason": "Requires both positive and negative ground truth", "roc_auc": None, "average_precision": None}
    order = np.argsort(-scores, kind="stable")
    sorted_scores, truth = scores[order], positive[order]
    stops = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.cumsum(truth)[stops]
    fp = 1 + stops - tp
    recall = np.r_[0, tp / positives]
    fpr = np.r_[0, fp / negatives]
    precision = np.r_[1, tp / (tp + fp)]
    return {"defined": True, "thresholds": [None] + sorted_scores[stops].tolist(),
            "false_positive_rate": fpr.tolist(), "true_positive_rate": recall.tolist(),
            "precision": precision.tolist(), "recall": recall.tolist(),
            "roc_auc": float(np.trapezoid(recall, fpr)),
            "average_precision": float(np.sum(np.diff(recall) * precision[1:]))}

"""Task-specific metrics with strict validation and explicit denominators."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def probabilities(values, *, categorical=True):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or not len(p) or p.shape[1] < 1:
        raise ValueError("Expected nonempty [examples, classes] probabilities")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if categorical and (p.shape[1] < 2 or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Categorical probabilities must sum to one")
    return p


def categorical_metrics(targets, predictions, labels=None, bins=10):
    p = probabilities(predictions)
    y = np.asarray(targets)
    n, k = p.shape
    if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= k).any():
        raise ValueError("Targets must be integer class indices matching predictions")
    if bins < 1:
        raise ValueError("bins must be positive")
    labels = list(labels) if labels is not None else [str(i) for i in range(k)]
    if len(labels) != k or len(set(labels)) != k:
        raise ValueError("Labels must be unique and cover every output column")
    pred = p.argmax(1)
    correct = pred == y
    confidence = p.max(1)
    matrix = np.zeros((k, k), dtype=int)
    np.add.at(matrix, (y, pred), 1)
    per_class = {}
    for i, label in enumerate(labels):
        tp = int(matrix[i, i]); support = int(matrix[i].sum()); guessed = int(matrix[:, i].sum())
        per_class[label] = {"support": support, "precision": tp / guessed if guessed else 0.,
                            "recall": tp / support if support else 0.,
                            "f1": 2 * tp / (support + guessed) if support + guessed else 0.}
    reliability = []
    assignment = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.
    for b in range(bins):
        mask = assignment == b
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else None
        conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / n * abs(acc - conf)
        reliability.append({"lower": b / bins, "upper": (b + 1) / bins,
                            "count": count, "accuracy": acc, "mean_confidence": conf})
    risk_coverage = []
    for threshold in [0., .25, .5, .6, .7, .8, .9, .95, .99, 1.]:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        risk_coverage.append({"threshold": threshold, "accepted": count,
                              "coverage": count / n,
                              "risk": float(1 - correct[accepted].mean()) if count else None})
    nll = float(-np.log(np.clip(p[np.arange(n), y], EPS, 1)).mean())
    return {"examples": n, "accuracy": float(correct.mean()),
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "nll": nll, "normalized_nll": nll / float(np.log(k)),
            "brier_sum_over_classes": float(np.square(p - np.eye(k)[y]).sum(1).mean()),
            "ece": float(ece), "probability_log_floor": EPS,
            "labels": labels, "per_class": per_class, "confusion_matrix": matrix.tolist(),
            "reliability": reliability, "risk_coverage": risk_coverage}


def average_precision(targets, scores):
    """Non-interpolated AP, grouping equal scores before integrating recall."""
    y = np.asarray(targets); s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not len(y) or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("AP expects aligned binary targets and finite scores")
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]
    recall = tp / positives
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def multilabel_metrics(targets, predictions):
    p = probabilities(predictions, categorical=False)
    y = np.asarray(targets)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Multi-label targets must be an aligned binary matrix")
    aps = [average_precision(y[:, i], p[:, i]) for i in range(p.shape[1])]
    present = [v for v in aps if v is not None]
    clipped = np.clip(p, EPS, 1 - EPS)
    return {"examples": len(p), "macro_ap": float(np.mean(present)) if present else None,
            "micro_ap": average_precision(y.ravel(), p.ravel()), "per_label_ap": aps,
            "positive_support": y.sum(0).tolist(), "labels_without_positives": sum(v is None for v in aps),
            "bce": float(-(y * np.log(clipped) + (1-y) * np.log(1-clipped)).mean()),
            "brier_mean_over_labels": float(np.square(p-y).mean()),
            "exact_match_at_half": float(np.all((p >= .5) == y, axis=1).mean())}


def point_hit(point, bbox):
    xy = np.asarray(point, dtype=float); box = np.asarray(bbox, dtype=float)
    if xy.shape != (2,) or box.shape != (4,) or not np.isfinite(xy).all() or not np.isfinite(box).all():
        raise ValueError("Expected a finite point and box")
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Target box must have positive area")
    return bool(x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)


def grounding_metrics(target_boxes, points):
    if len(target_boxes) != len(points) or not len(points):
        raise ValueError("Grounding predictions must cover every example")
    hits = [point_hit(p, b) for p, b in zip(points, target_boxes)]
    return {"examples": len(hits), "hits": sum(hits), "hit_rate": float(np.mean(hits))}


def aligned_predictions(records, predictions):
    """Reject duplicate, missing, and extra outputs before any metric is computed."""
    expected = [r["id"] for r in records]
    actual = [r["id"] for r in predictions]
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual):
        raise ValueError("Duplicate IDs")
    if set(expected) != set(actual):
        raise ValueError("Prediction IDs must exactly cover manifest IDs")
    by_id = {r["id"]: r for r in predictions}
    return [by_id[i] for i in expected]


def clustered_accuracy_interval(correct, groups, seed=17, repeats=1000):
    """Percentile cluster bootstrap for this fixed checkpoint, not training-seed variance."""
    values = np.asarray(correct, dtype=float)
    groups = np.asarray(groups)
    if values.shape != groups.shape or values.ndim != 1 or not len(values):
        raise ValueError("Expected aligned nonempty correctness and cluster arrays")
    unique = np.unique(groups)
    if len(unique) < 2:
        return {"low": None, "high": None, "clusters": len(unique)}
    totals = np.array([values[groups == g].sum() for g in unique])
    counts = np.array([(groups == g).sum() for g in unique])
    draws = np.random.default_rng(seed).integers(0, len(unique), (repeats, len(unique)))
    scores = totals[draws].sum(1) / counts[draws].sum(1)
    low, high = np.quantile(scores, [.025, .975])
    return {"low": float(low), "high": float(high), "clusters": len(unique),
            "method": "95% percentile cluster bootstrap; fixed trained checkpoint", "repeats": repeats, "seed": seed}

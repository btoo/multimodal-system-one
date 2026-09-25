"""Public benchmark adapters and scoring, separate from training data."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re

import numpy as np


def stable_rank(value):
    return hashlib.sha256(('miso-v4-frontier-v1:' + str(value)).encode()).hexdigest()


def mmstar_question(text):
    """Preserve publisher choices, including literal 'nan' distractors."""
    question, sep, tail = text.rpartition('Options:')
    if not sep:
        question, sep, tail = text.rpartition('Choices:')
        if not sep:
            first = re.search(r'\n\(A\)\s*', text)
            if first is None: raise ValueError('MMStar options delimiter missing')
            question, tail = text[:first.start()], text[first.start():]
        pattern = r'(?:^|\n)\s*\(([A-J])\)\s*'
    else:
        pattern = r'(?:^|,\s*)([A-J]):\s*'
    matches = list(re.finditer(pattern, tail.strip()))
    tail = tail.strip()
    if [m.group(1) for m in matches] != list('ABCDEFGHIJ'[:len(matches)]) or len(matches) < 2:
        raise ValueError('Ambiguous MMStar option labels')
    choices = [tail[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(tail)].strip()
               for i, m in enumerate(matches)]
    if any(not c for c in choices):
        raise ValueError('Empty MMStar choice')
    return question.strip(), choices


def model_input(row):
    """The only fields allowed across the inference boundary."""
    return {key: row[key] for key in ('question', 'choices', 'media')}


def point_inside(point, box, image_size):
    """ScreenSpot convention: normalized predicted point inside source box."""
    if point is None or len(point) != 2 or not all(math.isfinite(x) and 0 <= x <= 1 for x in point):
        return False
    x, y = point[0] * image_size[0], point[1] * image_size[1]
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def summarize(rows, predictions):
    by_id = {p['id']: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('Exactly one prediction, including explicit failures, is required per case')
    results = {}
    groups = defaultdict(list)
    for row in rows:
        groups[row['benchmark']].append(row)
    for benchmark, cases in sorted(groups.items()):
        correct, latencies, nll, brier, confidences, correctness = [], [], [], [], [], []
        strata, clusters, failures = defaultdict(list), defaultdict(list), defaultdict(int)
        for row in cases:
            prediction = by_id[row['id']]
            ok = prediction.get('status') == 'ok'
            p = np.asarray(prediction.get('probabilities', []), dtype=float)
            if ok and (p.shape != (len(row['choices']),) or not np.isfinite(p).all() or
                       (p < 0).any() or (p > 1).any() or abs(p.sum() - 1) > 1e-5):
                raise ValueError('Invalid probability vector')
            hit = int(ok and int(p.argmax()) == row['target'])
            correct.append(hit); strata[row['stratum']].append(hit); clusters[row['group_id']].append(hit)
            if ok:
                nll.append(-math.log(max(p[row['target']], 1e-12)))
                one_hot = np.eye(len(p))[row['target']]
                brier.append(float(np.square(p - one_hot).sum()))
                confidences.append(float(p.max())); correctness.append(hit)
                latencies.append(prediction['timings']['request_ms'])
            else:
                failures[prediction.get('status', 'missing')] += 1
        c, y = np.asarray(confidences), np.asarray(correctness)
        ece = 0.0
        for i in range(10):
            mask = (c >= i / 10) & ((c < (i + 1) / 10) if i < 9 else (c <= 1))
            if mask.any(): ece += mask.mean() * abs(c[mask].mean() - y[mask].mean())
        rng = np.random.default_rng(20260925)
        totals = np.array([(sum(v), len(v)) for v in clusters.values()])
        sample = rng.integers(0, len(totals), (2000, len(totals)))
        samples = totals[sample].sum(1)
        ci = np.quantile(samples[:, 0] / samples[:, 1], [.025, .975]).tolist()
        selective = {}
        for threshold in (.5, .7, .8, .9, .95):
            keep = c >= threshold
            selective[str(threshold)] = {'coverage': int(keep.sum()) / len(cases),
                'accuracy': float(y[keep].mean()) if keep.any() else None, 'accepted': int(keep.sum())}
        results[benchmark] = {'cases': len(cases), 'correct': sum(correct), 'accuracy': float(np.mean(correct)),
            'accuracy_group_bootstrap_95': ci, 'groups': len(clusters), 'coverage': len(c) / len(cases),
            'failures': dict(failures), 'nll_successes_only': float(np.mean(nll)) if nll else None,
            'brier_successes_only': float(np.mean(brier)) if brier else None,
            'ece_successes_only': float(ece) if len(c) else None, 'selective': selective,
            'strata': {k: {'cases': len(v), 'accuracy': float(np.mean(v))} for k, v in sorted(strata.items())},
            'latency_ms': {'median': float(np.median(latencies)), 'p95': float(np.quantile(latencies, .95))} if latencies else None}
    return results

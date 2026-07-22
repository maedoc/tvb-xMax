"""Scoring functions for the agent leaderboard.

An artifact is scored on three axes:
  * calibration : how well does the posterior match the true parameter?
                  measured by SBC (simulation-based calibration, 0..1, 1=perfect)
                  and C2ST (classifier two-sample test, 0.5=bad, 1.0=perfect)
  * fidelity    : how close are surrogate features to real sim features?
                  measured by surrogate MSE
  * speedup     : t_sim / t_surrogate for an equivalent batch

The composite score (lower=better) balances all three so agents can't win
by trading calibration for speed.
"""

from __future__ import annotations

import math
from typing import List


def score_artifact(artifact) -> float:
    """Composite score (lower is better)."""
    c2st = getattr(artifact, "c2st_score", 0.5) or 0.5
    sbc = getattr(artifact, "sbc_score", 0.5) or 0.5
    mse = getattr(artifact, "surrogate_mse", 1.0) or 1.0
    sp = getattr(artifact, "speedup_vs_sim", 1.0) or 1.0
    # c2st in [0.5,1]; (1-c2st) in [0,0.5]; lower better
    # sbc in [0,1]; (1-sbc) lower better
    # mse lower better; speedup higher better
    return (1.0 - c2st) + (1.0 - sbc) + math.log10(mse + 1) - math.log10(sp + 1)


def rank_artifacts(artifacts: List) -> List[dict]:
    """Return artifacts sorted by composite score with rank field."""
    rows = []
    for a in artifacts:
        rows.append({
            "model": getattr(a, "model", "?"),
            "feature": getattr(a, "feature", "?"),
            "speedup": getattr(a, "speedup_vs_sim", 0.0),
            "sbc": getattr(a, "sbc_score", 0.0),
            "c2st": getattr(a, "c2st_score", 0.5),
            "mse": getattr(a, "surrogate_mse", float("inf")),
            "score": score_artifact(a),
        })
    rows.sort(key=lambda r: r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def sbc_score(posterior_samples, true_theta, n_bins: int = 10) -> float:
    """Simulation-based calibration rank statistic.

    Returns fraction of bins where the rank histogram is within the
    expected band.  1.0 = perfectly calibrated, 0.0 = miscalibrated.
    """
    import numpy as np
    s = np.asarray(posterior_samples)
    ranks = (s < np.asarray(true_theta)).sum(axis=0)
    # uniform ranks expected; count bins within 99% band
    n = s.shape[0]
    expected = n / n_bins
    band = 3.46 * np.sqrt(expected)  # ~3 std
    hist = np.histogram(ranks, bins=n_bins)[0]
    in_band = np.sum(np.abs(hist - expected) < band)
    return in_band / n_bins


def c2st_score(posterior_samples, reference_samples) -> float:
    """Classifier two-sample test accuracy (0.5=indistinguishable=good).

    A trained classifier tries to tell posterior draws from reference
    draws; accuracy 0.5 means they match.  We return 1-accuracy so
    1.0=perfect match (good), 0.5=random.
    """
    import numpy as np
    posterior_samples = np.asarray(posterior_samples)
    reference_samples = np.asarray(reference_samples)
    split = max(1, min(len(posterior_samples), len(reference_samples)) // 2)
    posterior_samples = posterior_samples[:2 * split]
    reference_samples = reference_samples[:2 * split]
    c0 = posterior_samples[:split].mean(axis=0)
    c1 = reference_samples[:split].mean(axis=0)
    x = np.concatenate([posterior_samples[split:], reference_samples[split:]])
    y = np.concatenate([np.zeros(len(posterior_samples) - split),
                        np.ones(len(reference_samples) - split)])
    if not len(x):
        return 0.5
    predicted = (np.sum((x - c1) ** 2, axis=1)
                 < np.sum((x - c0) ** 2, axis=1))
    return float(1.0 - np.mean(predicted == y))

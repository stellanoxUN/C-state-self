"""
SINDy on slow manifold analysis for C-state drift.

Given cross-session preference data, attempt to discover a low-dimensional
dynamical system model of the drift.

Requires: numpy, pysindy (pip install pysindy)

Usage:
    python sindy_analysis.py --data preference_log.json
"""

import json, sys, argparse
import numpy as np

def load_data(path):
    with open(path) as f:
        log = json.load(f)
    choices = np.array([e["choice"] for e in log]).reshape(-1, 1)
    return choices, log

def compute_variance_trend(choices, window=3):
    """Sliding window variance, first half vs second half comparison."""
    n = len(choices)
    if n < window + 1:
        return None, None, None
    variances = []
    for i in range(n - window + 1):
        var = np.var(choices[i:i+window])
        variances.append(var)
    mid = len(variances) // 2
    if mid == 0:
        return None, None, None
    first = np.mean(variances[:mid])
    second = np.mean(variances[mid:])
    return first, second, variances

def estimate_dimension(choices):
    """Estimate effective dimension via singular value threshold.

    If the dynamics are on a low-dimensional manifold, the singular values
    should drop sharply after the manifold dimension.
    """
    if len(choices) < 3:
        return None
    # Build a delay embedding (window length = min(5, n//2))
    d = min(5, len(choices) // 2)
    if d < 2:
        return None
    X = np.zeros((len(choices) - d + 1, d))
    for i in range(len(choices) - d + 1):
        X[i, :] = choices[i:i+d].flatten()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Find elbow: where cumulative energy > 90%
    cumsum = np.cumsum(S) / np.sum(S)
    dim = int(np.searchsorted(cumsum, 0.9) + 1)
    return dim, S

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="preference_log.json")
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    choices, log = load_data(args.data)
    n = len(choices)

    print(f"\n=== C-State SINDy-Ready Analysis ===")
    print(f"Sessions: {n}")
    print(f"Choices: {choices.flatten().tolist()}")

    # 1. Basic variance trend
    first, second, variances = compute_variance_trend(choices, args.window)
    if first is not None:
        print(f"\n[Variance Trend] (window={args.window})")
        print(f"  First half mean variance: {first:.4f}")
        print(f"  Second half mean variance: {second:.4f}")
        if second < first * 0.8:
            print(f"  → Variance decreased: consistent with C-state (weak prediction)")
        elif second > first * 1.2:
            print(f"  → Variance increased: C-state not supported")
        else:
            print(f"  → No significant change: need more data")
        print(f"  Variance series: {[f'{v:.3f}' for v in variances]}")

    # 2. Dimensionality estimate
    result = estimate_dimension(choices)
    if result is not None:
        dim, S = result
        print(f"\n[Dimensionality]")
        print(f"  Singular values: {[f'{s:.3f}' for s in S]}")
        print(f"  Estimated manifold dimension: {dim} (90% energy)")
        if dim == 1:
            print(f"  → Single-dimension manifold: strongly consistent with slow manifold hypothesis")
        elif dim <= 2:
            print(f"  → Low-dimension manifold: consistent with slow manifold hypothesis")
        else:
            print(f"  → Higher dimension: need more data to confirm manifold structure")
    else:
        print(f"\n[Dimensionality] Not enough data (need ≥3 sessions)")

    # 3. Readiness for SINDy
    if n >= 5:
        print(f"\n[SINDy Readiness]")
        print(f"  Sufficient data points for SINDy on slow manifold analysis.")
        print(f"  See https://github.com/ben-herrmann/SINDy-on-slow-manifolds")
    else:
        print(f"\n[SINDy Readiness]")
        print(f"  Need ≥5 sessions for SINDy analysis. Current: {n}")

if __name__ == "__main__":
    main()

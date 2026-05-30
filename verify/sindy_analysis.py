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
    choices = np.array([e["choice"] for e in log])
    has_confidence = any("confidence" in e for e in log)
    if has_confidence:
        confs = np.array([e.get("confidence", np.nan) for e in log])
        return choices, confs, log
    return choices, None, log

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

    choices, confs, log = load_data(args.data)
    n = len(choices)

    print(f"\n=== C-State SINDy Analysis ===")
    print(f"Sessions: {n}")
    print(f"Choices: {choices.tolist()}")
    if confs is not None:
        # use confidence where available, fall back to binary (0/1) elsewhere
        signal = np.where(np.isnan(confs), choices.astype(float), confs)
        conf_only = ~np.isnan(confs)
        has_conf = np.sum(conf_only)
        print(f"Confidence: {confs.tolist()}")
        print(f"(confidence available for {has_conf}/{n} sessions)")
    else:
        print(f"(binary only, no confidence data)")
        signal = choices.astype(float)
    print()

    # 1. Variance trend on binary choices
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
    # 3. SINDy analysis on continuous signal
    from sindy_runner import run_sindy
    result = run_sindy(signal, t_range=n, label="C-state signal")
    if result:
        terms, total_weight, details = result
        print(f"\n[SINDy] {terms} nonzero terms, total weight={total_weight:.3f}")
        if terms > 0 and terms <= 3:
            print(f"  → Sparse model found: consistent with slow manifold hypothesis")
        elif terms == 0:
            print(f"  → No model found: need more data or lower threshold")
        else:
            print(f"  → Dense model ({terms} terms): manifold structure unclear")

if __name__ == "__main__":
    main()

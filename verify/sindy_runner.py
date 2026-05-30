"""
SINDy runner for C-state drift analysis.
Run SINDy on a 1D signal and return model complexity metrics.
"""
import numpy as np
import pysindy as ps

def run_sindy(signal, t_range=None, label="signal", max_degree=2):
    """
    Attempt to discover sparse dynamical model from a 1D time series.
    
    Returns: (nonzero_terms, total_weight, details_dict) or None
    """
    n = len(signal)
    if n < 5:
        return None
    
    x = signal.reshape(-1, 1).astype(float)
    t = np.arange(n) if t_range is None else np.arange(min(t_range, n))
    
    # Test multiple thresholds to find sparsest valid model
    results = []
    for deg in [1, 2]:
        for thresh in [0.01, 0.05, 0.1, 0.2, 0.5]:
            try:
                lib = ps.PolynomialLibrary(degree=deg)
                opt = ps.STLSQ(threshold=thresh)
                sd = ps.SINDy(optimizer=opt, feature_library=lib)
                sd.fit(x, t=t)
                coefs = sd.coefficients()
                nonzero = np.count_nonzero(coefs)
                total = np.sum(np.abs(coefs))
                results.append({
                    "deg": deg, "threshold": thresh,
                    "nonzero": nonzero, "total_weight": total,
                    "model": sd
                })
            except Exception:
                continue
    
    if not results:
        return None
    
    # Pick the sparsest model that has nonzero terms
    nonzero_models = [r for r in results if r["nonzero"] > 0]
    
    if nonzero_models:
        # prefer sparsest; if tied, prefer lower threshold
        best = min(nonzero_models, key=lambda r: (r["nonzero"], r["threshold"]))
    else:
        best = min(results, key=lambda r: r["threshold"])
    
    return (best["nonzero"], best["total_weight"], {
        "degree": best["deg"],
        "threshold": best["threshold"],
        "n_points": n,
    })

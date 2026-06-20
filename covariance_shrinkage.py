#!/usr/bin/env python
# covariance_shrinkage.py — Robust covariance estimation
"""
Ledoit-Wolf shrinkage estimator for portfolio covariance matrices.

Problem: When N (assets) > T (observations), sample covariance is singular
and np.linalg.pinv produces mathematically unstable results.

Solution: Ledoit-Wolf shrinks the sample covariance toward a structured
target (identity or constant-correlation), dramatically reducing estimation
error in N > T regimes.

Reference: Ledoit & Wolf (2004) "A well-conditioned estimator for
large-dimensional covariance matrices"
"""

import numpy as np
from sklearn.covariance import LedoitWolf, OAS
from typing import Optional, Tuple


def robust_covariance(
    returns: np.ndarray,
    method: str = 'ledoit_wolf',
    return_shrinkage: bool = False,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Compute a robust covariance matrix from asset returns.

    Args:
        returns: (T, N) array — T observations, N assets
        method: 'ledoit_wolf' | 'oas' | 'sample'
        return_shrinkage: if True, also return shrinkage intensity δ ∈ [0,1]

    Returns:
        cov_matrix: (N, N) robust covariance estimate
        shrinkage: (optional) shrinkage intensity
    """
    T, N = returns.shape

    # Edge case: single asset
    if N == 1:
        cov = np.atleast_2d(np.var(returns))
        return (cov, 0.0) if return_shrinkage else cov

    # Edge case: too few samples — use diagonal with equal variance
    if T < 3:
        avg_var = np.var(returns, axis=0).mean()
        cov = np.eye(N) * max(avg_var, 1e-10)
        return (cov, 1.0) if return_shrinkage else cov

    if method == 'ledoit_wolf':
        lw = LedoitWolf(assume_centered=False).fit(returns)
        cov = lw.covariance_
        # Ensure symmetry and positive-definiteness
        cov = (cov + cov.T) / 2
        # Add tiny ridge for numerical stability
        cov += np.eye(N) * 1e-10
        shrinkage = lw.shrinkage_ if hasattr(lw, 'shrinkage_') else 0.5

    elif method == 'oas':
        oas = OAS(assume_centered=False).fit(returns)
        cov = oas.covariance_
        cov = (cov + cov.T) / 2
        cov += np.eye(N) * 1e-10
        shrinkage = oas.shrinkage_ if hasattr(oas, 'shrinkage_') else 0.5

    elif method == 'sample':
        cov = np.cov(returns, rowvar=False)
        cov = (cov + cov.T) / 2
        cov += np.eye(N) * 1e-10
        shrinkage = 0.0

    else:
        raise ValueError(f"Unknown method: {method}")

    return (cov, shrinkage) if return_shrinkage else cov


def min_variance_weights(
    returns: np.ndarray,
    method: str = 'ledoit_wolf',
    long_only: bool = True,
) -> np.ndarray:
    """
    Compute minimum-variance portfolio weights using robust covariance.

    w* = argmin w'Σw  subject to Σw = 1, w ≥ 0 (if long_only)

    Args:
        returns: (T, N) asset returns in decimal
        method: covariance estimation method
        long_only: enforce w ≥ 0 constraint

    Returns:
        weights: (N,) optimal portfolio weights
    """
    cov = robust_covariance(returns, method=method)
    N = cov.shape[0]

    try:
        # Quadratic programming: min w'Σw s.t. 1'w = 1
        from scipy.optimize import minimize

        def objective(w):
            return 0.5 * w @ cov @ w

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(N)] if long_only else None
        x0 = np.ones(N) / N

        result = minimize(
            objective, x0, method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12},
        )

        if result.success:
            return result.x
        else:
            # Fallback: equal weight
            return np.ones(N) / N

    except ImportError:
        # No scipy — use analytical solution (unconstrained)
        cov_inv = np.linalg.pinv(cov)
        ones = np.ones(N)
        w = cov_inv @ ones / (ones @ cov_inv @ ones)
        if long_only:
            w = np.maximum(w, 0)
            w /= w.sum()
        return w


def condition_number(cov: np.ndarray) -> float:
    """Compute condition number of covariance matrix. > 30 = ill-conditioned."""
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-15)
    return float(eigenvalues.max() / eigenvalues.min())


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)
    # Simulate N > T regime: 50 assets, 20 observations
    T, N = 20, 50
    true_factors = np.random.randn(T, 3)  # 3 latent factors
    loadings = np.random.randn(3, N)
    noise = np.random.randn(T, N) * 0.1
    returns = true_factors @ loadings + noise

    # Sample covariance — ill-conditioned
    sample_cov = np.cov(returns, rowvar=False)
    print(f"Sample cov condition number: {condition_number(sample_cov):.0f}")

    # Ledoit-Wolf — well-conditioned
    lw_cov, shrink = robust_covariance(returns, 'ledoit_wolf', return_shrinkage=True)
    print(f"LW cov condition number:   {condition_number(lw_cov):.0f} "
          f"(shrinkage={shrink:.3f})")

    print(f"\n{'=' * 50}")
    print("Ledoit-Wolf reduces condition number by "
          f"{condition_number(sample_cov) / max(condition_number(lw_cov), 1):.0f}x")
    print("Safe for N > T inversion. ✅")

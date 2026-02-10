"""
Optimized Wasserstein - Numba JIT compilation
GARANTIT les mêmes résultats que scipy.stats.wasserstein_distance
"""

import numpy as np
from numba import njit

@njit
def wasserstein_1d_numba(u: np.ndarray, v: np.ndarray) -> float:
    """Helper function for wasserstein 1d numba."""
    
    n = len(u)
    m = len(v)
    
    # Sort
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    
    # Merge sorted arrays with tracking
    all_values = np.empty(n + m)
    u_weights = np.empty(n + m)
    v_weights = np.empty(n + m)
    
    i = 0  # index in u_sorted
    j = 0  # index in v_sorted
    k = 0  # index in merged arrays
    
    while i < n and j < m:
        if u_sorted[i] < v_sorted[j]:
            all_values[k] = u_sorted[i]
            u_weights[k] = 1.0 / n
            v_weights[k] = 0.0
            i += 1
        elif u_sorted[i] > v_sorted[j]:
            all_values[k] = v_sorted[j]
            u_weights[k] = 0.0
            v_weights[k] = 1.0 / m
            j += 1
        else:
            # Equal values
            all_values[k] = u_sorted[i]
            u_weights[k] = 1.0 / n
            v_weights[k] = 1.0 / m
            i += 1
            j += 1
        k += 1
    
    # Add remaining elements
    while i < n:
        all_values[k] = u_sorted[i]
        u_weights[k] = 1.0 / n
        v_weights[k] = 0.0
        i += 1
        k += 1
    
    while j < m:
        all_values[k] = v_sorted[j]
        u_weights[k] = 0.0
        v_weights[k] = 1.0 / m
        j += 1
        k += 1
    
    # Trim arrays
    all_values = all_values[:k]
    u_weights = u_weights[:k]
    v_weights = v_weights[:k]
    
    # Compute cumulative weights
    u_cumweights = np.cumsum(u_weights)
    v_cumweights = np.cumsum(v_weights)
    
    # Compute Wasserstein distance as area between CDFs
    wass_dist = 0.0
    for idx in range(k - 1):
        delta = all_values[idx + 1] - all_values[idx]
        cdf_diff = abs(u_cumweights[idx] - v_cumweights[idx])
        wass_dist += delta * cdf_diff
    
    return wass_dist


@njit
def compute_pairwise_wasserstein_numba(data: np.ndarray) -> np.ndarray:
    """Compute pairwise wasserstein numba."""
    
    n_obs, n_tickers = data.shape
    avg_distances = np.zeros(n_tickers)
    
    for i in range(n_tickers):
        total_dist = 0.0
        count = 0
        
        for j in range(n_tickers):
            if i != j:
                dist = wasserstein_1d_numba(data[:, i], data[:, j])
                total_dist += dist
                count += 1
        
        avg_distances[i] = total_dist / count if count > 0 else 0.0
    
    return avg_distances


def test_numba_vs_scipy():
    """Helper function for test numba vs scipy."""
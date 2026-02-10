"""
HMM Optimisé - VERSION EXACTE DU NOTEBOOK
"""

import numpy as np
from hmmlearn import hmm


def fit_optimized_hmm(
    """Fit optimized hmm."""
    wass_features: np.ndarray,
    n_components: int = 3,
    persistence: float = 0.90,
    smooth_window: int = 20,
    covariance_type: str = "diag",
):
    """
    VERSION EXACTE: fit_robust_hmm_microstructure du notebook.
    
    1. Covariance diagonale
    2. Persistance 90%
    3. Smoothing window=20
    """
    
    # Standardisation
    X = (wass_features - np.mean(wass_features, axis=0)) / (np.std(wass_features, axis=0) + 1e-9)
    
    # Fit HMM diagonal
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=1000,
        random_state=42,
        init_params='stmc'
    )
    model.fit(X)
    
    # Forcer persistance
    transmat_persistent = np.ones((n_components, n_components)) * (1 - persistence) / (n_components - 1)
    np.fill_diagonal(transmat_persistent, persistence)
    model.transmat_ = transmat_persistent
    
    # Prédire
    states_raw = model.predict(X)
    
    # Smoothing
    states_smooth = states_raw.copy()
    n = len(states_raw)
    
    for i in range(smooth_window, n - smooth_window):
        window_states = states_raw[i-smooth_window:i+smooth_window]
        majority = np.bincount(window_states).argmax()
        states_smooth[i] = majority
    
    # Affichage
    unique, counts = np.unique(states_smooth, return_counts=True)
    print("\n" + "="*80)
    print("DISTRIBUTION DES RÉGIMES (HMM OPTIMISÉ)")
    print("="*80)
    for s, c in zip(unique, counts):
        print(f"Régime {s}: {c:,} obs ({c/len(states_smooth)*100:.1f}%)")
    
    n_transitions = np.sum(np.diff(states_smooth) != 0)
    avg_duration = len(states_smooth) / (n_transitions + 1) * 0.5
    print(f"\nDurée moyenne: {avg_duration:.1f}s")
    print(f"Transitions: {n_transitions}")
    
    return model, states_smooth


def get_state_probabilities(
    """Get state probabilities."""
    model: hmm.GaussianHMM,
    wass_features: np.ndarray
) -> np.ndarray:
    """
    Extrait les probabilités d'appartenance à chaque régime.

    CRUCIAL pour le Méta-HMM : au lieu de labels discrets (0, 1, 2),
    on utilise les probabilités continues P(état | observations).

    Args:
        model: HMM fitted
        wass_features: Features Wasserstein

    Returns:
        np.ndarray de shape (n_obs, n_components)
        Chaque ligne somme à 1 (distribution de probabilité)
    """
    # Standardisation (même que fit)
    X = (wass_features - np.mean(wass_features, axis=0)) / (np.std(wass_features, axis=0) + 1e-9)

    # Algorithme forward-backward pour obtenir les probabilités
    # Retourne P(état_i à temps t | toutes les observations)
    state_probs = model.predict_proba(X)

    return state_probs


def fit_optimized_hmm_with_probs(
    """Fit optimized hmm with probs."""
    wass_features: np.ndarray,
    n_components: int = 3,
    persistence: float = 0.90,
    smooth_window: int = 20,
    covariance_type: str = "diag",
):
    """
    Comme fit_optimized_hmm mais retourne aussi les probabilités.

    Returns:
        model: HMM fitted
        states_smooth: Labels d'état lissés
        state_probs: Probabilités d'appartenance (n_obs, n_components)
    """
    # Fit standard
    model, states_smooth = fit_optimized_hmm(
        wass_features,
        n_components=n_components,
        persistence=persistence,
        smooth_window=smooth_window,
        covariance_type=covariance_type,
    )

    # Extraction des probabilités
    state_probs = get_state_probabilities(model, wass_features)

    return model, states_smooth, state_probs

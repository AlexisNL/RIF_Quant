# -*- coding: utf-8 -*-
"""
Local (per-ticker) HMM with persistence forcing and majority-vote smoothing.

Design mirrors MetaHMM for a symmetric interface:
    hmm = LocalHMM(n_regimes=3, persistence=0.90, smooth_window=20)
    hmm.fit(wass_features)
    states = hmm.predict()
    probs  = hmm.predict_proba()

Backward-compatible functions are kept at the bottom of the module so that
existing call sites (run_hierarchical_contagion, optimize_hierarchical_parameters)
continue to work without modification.
"""

from __future__ import annotations

import numpy as np
from hmmlearn import hmm


class LocalHMM:
    """
    Per-ticker Hidden Markov Model with persistence forcing and smoothing.

    The model is fitted once via :meth:`fit`. Discrete state labels and
    posterior probabilities are stored as attributes after fitting.

    Parameters
    ----------
    n_regimes : int
        Number of hidden states (regimes).
    persistence : float
        Probability of staying in the same regime (forced post-fit).
    smooth_window : int
        Half-window for majority-vote smoothing of raw state predictions.
    covariance_type : str
        HMM covariance structure passed to hmmlearn (``"diag"`` or ``"full"``).
    random_state : int
        Seed for reproducibility.

    Attributes
    ----------
    model_ : hmm.GaussianHMM
        Fitted hmmlearn model (available after :meth:`fit`).
    states_ : np.ndarray, shape (n_obs,)
        Smoothed discrete regime labels (available after :meth:`fit`).
    probs_ : np.ndarray, shape (n_obs, n_regimes)
        Posterior regime probabilities P(state | obs) (available after :meth:`fit`).
    """

    def __init__(
        self,
        n_regimes: int = 3,
        persistence: float = 0.90,
        smooth_window: int = 20,
        covariance_type: str = "diag",
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.persistence = persistence
        self.smooth_window = smooth_window
        self.covariance_type = covariance_type
        self.random_state = random_state

        # Set after fit
        self.model_: hmm.GaussianHMM | None = None
        self.states_: np.ndarray | None = None
        self.probs_: np.ndarray | None = None
        self._X_mean: np.ndarray | None = None
        self._X_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray) -> "LocalHMM":
        """
        Fit the HMM on Wasserstein features, apply persistence forcing and
        majority-vote smoothing.

        Parameters
        ----------
        features : np.ndarray, shape (n_obs, n_features)
            Raw (un-standardised) Wasserstein distance features.

        Returns
        -------
        self
            Allows method chaining: ``hmm.fit(X).predict()``.
        """
        # --- Standardise (stored for transform in predict_proba) ---
        self._X_mean = np.mean(features, axis=0)
        self._X_std = np.std(features, axis=0) + 1e-9
        X = (features - self._X_mean) / self._X_std

        # --- Fit Gaussian HMM ---
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=1000,
            random_state=self.random_state,
            init_params="stmc",
        )
        model.fit(X)

        # --- Force persistence on transition matrix ---
        off_diag = (1.0 - self.persistence) / (self.n_regimes - 1)
        transmat = np.full((self.n_regimes, self.n_regimes), off_diag)
        np.fill_diagonal(transmat, self.persistence)
        model.transmat_ = transmat

        # --- Predict raw states & apply majority-vote smoothing ---
        states_raw = model.predict(X)
        states_smooth = self._majority_vote_smooth(states_raw)

        self.model_ = model
        self.states_ = states_smooth
        self.probs_ = model.predict_proba(X)

        self._print_summary()
        return self

    def predict(self) -> np.ndarray:
        """Return smoothed discrete regime labels (requires prior :meth:`fit`)."""
        self._check_fitted()
        return self.states_

    def predict_proba(self) -> np.ndarray:
        """
        Return posterior regime probabilities P(state | obs).

        Shape: (n_obs, n_regimes). Each row sums to 1.
        """
        self._check_fitted()
        return self.probs_

    @property
    def is_converged(self) -> bool:
        """True if the hmmlearn EM algorithm converged."""
        self._check_fitted()
        return bool(self.model_.monitor_.converged)

    @property
    def transmat_(self) -> np.ndarray:
        """Forced transition matrix (n_regimes × n_regimes)."""
        self._check_fitted()
        return self.model_.transmat_

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _majority_vote_smooth(self, states_raw: np.ndarray) -> np.ndarray:
        """Replace each state with the majority vote in a sliding window."""
        states_smooth = states_raw.copy()
        n = len(states_raw)
        w = self.smooth_window
        for i in range(w, n - w):
            window = states_raw[i - w : i + w]
            states_smooth[i] = np.bincount(window).argmax()
        return states_smooth

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("LocalHMM is not fitted yet. Call .fit(features) first.")

    def _print_summary(self) -> None:
        unique, counts = np.unique(self.states_, return_counts=True)
        print("\n" + "=" * 80)
        print("DISTRIBUTION DES RÉGIMES (HMM OPTIMISÉ)")
        print("=" * 80)
        for s, c in zip(unique, counts):
            print(f"Régime {s}: {c:,} obs ({c / len(self.states_) * 100:.1f}%)")
        n_transitions = np.sum(np.diff(self.states_) != 0)
        avg_duration = len(self.states_) / (n_transitions + 1) * 0.5
        print(f"\nDurée moyenne: {avg_duration:.1f}s")
        print(f"Transitions: {n_transitions}")


# ---------------------------------------------------------------------------
# Backward-compatible functional API
# ---------------------------------------------------------------------------
# These thin wrappers preserve the original call signatures used in
# run_hierarchical_contagion.py and optimize_hierarchical_parameters.py.
# They can be removed once those scripts are migrated to LocalHMM directly.

def fit_optimized_hmm(
    wass_features: np.ndarray,
    n_components: int = 3,
    persistence: float = 0.90,
    smooth_window: int = 20,
    covariance_type: str = "diag",
):
    """Backward-compatible wrapper — prefer ``LocalHMM`` for new code."""
    local_hmm = LocalHMM(
        n_regimes=n_components,
        persistence=persistence,
        smooth_window=smooth_window,
        covariance_type=covariance_type,
    ).fit(wass_features)
    return local_hmm.model_, local_hmm.states_


def get_state_probabilities(
    model: hmm.GaussianHMM,
    wass_features: np.ndarray,
) -> np.ndarray:
    """Backward-compatible wrapper — prefer ``LocalHMM.predict_proba()``."""
    X = (wass_features - np.mean(wass_features, axis=0)) / (
        np.std(wass_features, axis=0) + 1e-9
    )
    return model.predict_proba(X)


def fit_optimized_hmm_with_probs(
    wass_features: np.ndarray,
    n_components: int = 3,
    persistence: float = 0.90,
    smooth_window: int = 20,
    covariance_type: str = "diag",
):
    """Backward-compatible wrapper — prefer ``LocalHMM`` for new code."""
    local_hmm = LocalHMM(
        n_regimes=n_components,
        persistence=persistence,
        smooth_window=smooth_window,
        covariance_type=covariance_type,
    ).fit(wass_features)
    return local_hmm.model_, local_hmm.states_, local_hmm.probs_

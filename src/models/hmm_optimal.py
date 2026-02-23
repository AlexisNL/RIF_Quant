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
        n_iter: int = 1000,
        verbose: bool = True,
    ) -> None:
        self.n_regimes = n_regimes
        self.persistence = persistence
        self.smooth_window = smooth_window
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_iter = n_iter
        self.verbose = verbose
        self.model_: hmm.GaussianHMM | None = None
        self.states_: np.ndarray | None = None
        self.probs_: np.ndarray | None = None
        self._X_mean: np.ndarray | None = None
        self._X_std: np.ndarray | None = None

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
        self._X_mean = np.mean(features, axis=0)
        self._X_std = np.std(features, axis=0) + 1e-9
        X = (features - self._X_mean) / self._X_std

        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params="stmc",
        )
        model.fit(X)

        off_diag = (1.0 - self.persistence) / (self.n_regimes - 1)
        transmat = np.full((self.n_regimes, self.n_regimes), off_diag)
        np.fill_diagonal(transmat, self.persistence)
        model.transmat_ = transmat

        states_raw = model.predict(X)
        states_smooth = self._majority_vote_smooth(states_raw)

        self.model_ = model
        self.states_ = states_smooth
        self.probs_ = model.predict_proba(X)

        if self.verbose:
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

    def _majority_vote_smooth(self, states_raw: np.ndarray) -> np.ndarray:
        """Replace each state with the majority vote in a sliding window.

        w=0 is a no-op (returns a copy without any smoothing).
        Uses numpy stride tricks to avoid a Python loop.
        """
        w = self.smooth_window
        if w == 0:
            return states_raw.copy()

        from numpy.lib.stride_tricks import sliding_window_view
        n = len(states_raw)
        states_smooth = states_raw.copy()
        if n <= 2 * w:
            return states_smooth

        windows = sliding_window_view(states_raw, 2 * w + 1)   # shape (n-2w, 2w+1)
        modes = np.array([np.bincount(row).argmax() for row in windows])
        states_smooth[w : n - w] = modes
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

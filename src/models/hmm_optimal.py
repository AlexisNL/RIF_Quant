from __future__ import annotations

import logging

import numpy as np
from hmmlearn import hmm

logger = logging.getLogger(__name__)


class LocalHMM:
    """
    Per-ticker Hidden Markov Model with persistence forcing and smoothing.

    The model is fitted via :meth:`fit` using multiple random restarts (``n_init``),
    each running ``n_iter`` EM iterations. The restart with the highest
    log-likelihood is kept before persistence forcing and smoothing are applied.

    Setting ``n_init=5, n_iter=200`` (the defaults) keeps the same total EM
    budget as the former ``n_init=1, n_iter=1000`` while exploring the parameter
    space more thoroughly and avoiding degenerate local optima.

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
        Base seed; restart *i* uses seed ``random_state + i`` for full
        reproducibility across all restarts.
    n_iter : int
        Maximum EM iterations per restart.
    n_init : int
        Number of independent random restarts. The restart with the highest
        log-likelihood (``model.score(X)``) is selected.
    verbose : bool
        Whether to log the regime distribution summary after fitting.

    Attributes
    ----------
    model_ : hmm.GaussianHMM
        Best fitted hmmlearn model (available after :meth:`fit`).
    states_ : np.ndarray, shape (n_obs,)
        Smoothed discrete regime labels (available after :meth:`fit`).
    probs_ : np.ndarray, shape (n_obs, n_regimes)
        Posterior regime probabilities P(state | obs) (available after :meth:`fit`).
    best_score_ : float
        Log-likelihood per sample of the selected restart.
    best_init_ : int
        Index (0-based) of the restart that achieved ``best_score_``.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        persistence: float = 0.90,
        smooth_window: int = 20,
        covariance_type: str = "diag",
        random_state: int = 42,
        n_iter: int = 200,
        n_init: int = 5,
        verbose: bool = True,
    ) -> None:
        self.n_regimes = n_regimes
        self.persistence = persistence
        self.smooth_window = smooth_window
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_init = n_init
        self.verbose = verbose
        self.model_: hmm.GaussianHMM | None = None
        self.states_: np.ndarray | None = None
        self.probs_: np.ndarray | None = None
        self.best_score_: float | None = None
        self.best_init_: int | None = None
        self._X_mean: np.ndarray | None = None
        self._X_std: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "LocalHMM":
        """
        Fit the HMM on Wasserstein features, apply persistence forcing and
        majority-vote smoothing.

        Runs ``n_init`` independent EM restarts (seeds ``random_state`` to
        ``random_state + n_init - 1``) and keeps the model with the highest
        log-likelihood before applying persistence forcing.

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

        best_model: hmm.GaussianHMM | None = None
        best_score = -np.inf
        best_init = 0

        for i in range(self.n_init):
            candidate = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state + i,
                init_params="stmc",
            )
            candidate.fit(X)
            try:
                score = candidate.score(X)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_model = candidate
                best_init = i

        if best_model is None:
            raise RuntimeError("All HMM restarts failed to converge.")

        off_diag = (1.0 - self.persistence) / (self.n_regimes - 1)
        transmat = np.full((self.n_regimes, self.n_regimes), off_diag)
        np.fill_diagonal(transmat, self.persistence)
        best_model.transmat_ = transmat

        states_raw = best_model.predict(X)
        states_smooth = self._majority_vote_smooth(states_raw)

        self.model_ = best_model
        self.states_ = states_smooth
        self.probs_ = best_model.predict_proba(X)
        self.best_score_ = best_score
        self.best_init_ = best_init

        if self.verbose:
            self._log_summary()
        return self

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

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted model to new out-of-sample data.

        Uses the same standardisation (mean/std) fitted on the training data
        and the same majority-vote smoothing window, so regime labels are
        consistent between train and test sets.

        Parameters
        ----------
        features : np.ndarray, shape (n_obs, n_features)
            Raw (un-standardised) Wasserstein features for the test period.

        Returns
        -------
        np.ndarray, shape (n_obs,)
            Smoothed regime labels decoded via Viterbi on the fitted model.
        """
        self._check_fitted()
        X = (features - self._X_mean) / self._X_std
        states_raw = self.model_.predict(X)
        return self._majority_vote_smooth(states_raw)

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("LocalHMM is not fitted yet. Call .fit(features) first.")

    def _log_summary(self) -> None:
        unique, counts = np.unique(self.states_, return_counts=True)
        monitor = getattr(self.model_, "monitor_", None)
        n_iter_actual = len(monitor.history) if monitor is not None else "?"
        converged = bool(monitor.converged) if monitor is not None else None
        if converged is False:
            logger.warning(
                "HMM did not converge within %d iterations (best restart %d/%d) "
                "-- consider raising n_iter",
                self.n_iter, self.best_init_ + 1, self.n_init,
            )
        logger.info(
            "--- regime distribution (restart %d/%d | iters %s/%d | converged=%s | score=%.4f) ---",
            self.best_init_ + 1, self.n_init,
            n_iter_actual, self.n_iter,
            converged, self.best_score_,
        )
        for s, c in zip(unique, counts):
            logger.info("  regime %d: %s obs (%.1f%%)", s, f"{c:,}", c / len(self.states_) * 100)
        n_transitions = np.sum(np.diff(self.states_) != 0)
        avg_duration = len(self.states_) / (n_transitions + 1) * 0.5
        logger.info("  avg duration: %.1fs | transitions: %d", avg_duration, n_transitions)

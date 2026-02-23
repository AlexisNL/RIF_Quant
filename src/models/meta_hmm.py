"""
Hierarchical Meta-HMM — two-level architecture
===============================================

Key design: second-order HMM for contagion detection.

Architecture
------------
Level 1 (Local):  one HMM per asset → posterior probabilities P(state | asset)
Level 2 (Global): Meta-HMM observes all posteriors → sectoral regime

Advantages
----------
1. Resolves label switching: the Meta-HMM realigns local state semantics.
2. Filters noise: isolated transitions not confirmed by other assets are ignored.
3. Detects contagion: identifies when multiple assets change regime simultaneously.
4. Sectoral rotation: distinguishes local stress from systemic stress.

Observed variables for the Meta-HMM
-------------------------------------
- State probabilities from each local HMM  (n_tickers × n_regimes)
- Allows the model to capture complex co-occurrence patterns.

Example
-------
If HMM(AAPL) says "80% stress", HMM(GOOG) says "70% stress",
but HMM(MSFT) says "10% stress", the Meta-HMM can detect
"partial tech contagion" vs "generalised panic".
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from hmmlearn import hmm

logger = logging.getLogger(__name__)


class MetaHMM:
    """
    Hierarchical Meta-HMM for sectoral regime detection.

    Observes the state probabilities of local HMMs and detects
    contagion patterns and sectoral rotation.
    """

    def __init__(
        self,
        n_global_regimes: int = 3,
        persistence: float = 0.95,
        covariance_type: str = 'diag'
    ):
        """
        Initialise the Meta-HMM.

        Parameters
        ----------
        n_global_regimes : int
            Number of global (sectoral) regimes.
        persistence : float
            Self-transition probability (diagonal of the transition matrix).
        covariance_type : str
            Covariance structure: ``'diag'``, ``'full'``, or ``'tied'``.
        """
        self.n_global_regimes = n_global_regimes
        self.persistence = persistence
        self.covariance_type = covariance_type
        self.model = None
        self.is_fitted = False

    def fit(
        self,
        local_state_probs: Dict[str, np.ndarray],
        tickers: List[str]
    ) -> 'MetaHMM':
        """
        Fit the Meta-HMM on local HMM state probabilities.

        Parameters
        ----------
        local_state_probs : dict
            ``{ticker: state_probs}`` where ``state_probs`` has shape
            ``(n_obs, n_local_regimes)``.
        tickers : list of str
            Ordered list of tickers.

        Returns
        -------
        self
        """
        # Concatenate state probabilities from all assets
        # Final shape: (n_obs, n_tickers × n_local_regimes)
        prob_arrays = [local_state_probs[ticker] for ticker in tickers]
        X_meta = np.hstack(prob_arrays)

        logger.info(
            "Meta-HMM feature dimensions: %d assets × %d local regimes = %d features, %d observations",
            len(tickers), prob_arrays[0].shape[1], X_meta.shape[1], X_meta.shape[0],
        )

        # Standardise features
        self.feature_mean = np.mean(X_meta, axis=0)
        self.feature_std = np.std(X_meta, axis=0) + 1e-9
        X_scaled = (X_meta - self.feature_mean) / self.feature_std

        # Fit global HMM
        logger.info("Fitting Meta-HMM with %d global regimes...", self.n_global_regimes)

        self.model = hmm.GaussianHMM(
            n_components=self.n_global_regimes,
            covariance_type=self.covariance_type,
            n_iter=1000,
            random_state=42,
            init_params='stmc'
        )
        self.model.fit(X_scaled)

        # Force persistence (more stable global regimes)
        transmat_persistent = np.ones(
            (self.n_global_regimes, self.n_global_regimes)
        ) * (1 - self.persistence) / (self.n_global_regimes - 1)
        np.fill_diagonal(transmat_persistent, self.persistence)
        self.model.transmat_ = transmat_persistent

        self.is_fitted = True
        logger.info("OK Meta-HMM fitted with persistence = %.2f", self.persistence)

        return self

    def predict_global_states(
        self,
        local_state_probs: Dict[str, np.ndarray],
        tickers: List[str],
        smooth_window: int = 30
    ) -> np.ndarray:
        """
        Predict global (sectoral) states from local HMM posteriors.

        Parameters
        ----------
        local_state_probs : dict
            State probabilities from local HMMs.
        tickers : list of str
            Ordered list of tickers.
        smooth_window : int
            Majority-vote smoothing half-window (larger for global regimes).

        Returns
        -------
        np.ndarray
            Smoothed global state sequence.
        """
        if not self.is_fitted:
            raise ValueError("Meta-HMM not fitted. Call .fit() first.")

        # Concatenate and standardise
        prob_arrays = [local_state_probs[ticker] for ticker in tickers]
        X_meta = np.hstack(prob_arrays)
        X_scaled = (X_meta - self.feature_mean) / self.feature_std

        # Viterbi decoding
        global_states_raw = self.model.predict(X_scaled)

        # Majority-vote smoothing (more aggressive for global regimes)
        global_states_smooth = global_states_raw.copy()
        n = len(global_states_raw)

        for i in range(smooth_window, n - smooth_window):
            window_states = global_states_raw[i - smooth_window : i + smooth_window]
            majority = np.bincount(window_states).argmax()
            global_states_smooth[i] = majority

        # Log regime distribution
        unique, counts = np.unique(global_states_smooth, return_counts=True)
        for s, c in zip(unique, counts):
            logger.info(
                "Global regime %d: %d obs (%.1f%%)", s, c,
                c / len(global_states_smooth) * 100,
            )

        n_transitions = np.sum(np.diff(global_states_smooth) != 0)
        avg_duration = len(global_states_smooth) / (n_transitions + 1) * 0.5
        logger.info("Average duration: %.1fs | Transitions: %d", avg_duration, n_transitions)

        return global_states_smooth

    def predict_global_probs(
        self,
        local_state_probs: Dict[str, np.ndarray],
        tickers: List[str]
    ) -> np.ndarray:
        """
        Posterior probabilities over global regimes.

        Parameters
        ----------
        local_state_probs : dict
            State probabilities from local HMMs.
        tickers : list of str
            Ordered list of tickers.

        Returns
        -------
        np.ndarray, shape (n_obs, n_global_regimes)
        """
        if not self.is_fitted:
            raise ValueError("Meta-HMM not fitted.")

        prob_arrays = [local_state_probs[ticker] for ticker in tickers]
        X_meta = np.hstack(prob_arrays)
        X_scaled = (X_meta - self.feature_mean) / self.feature_std

        return self.model.predict_proba(X_scaled)

    def get_transition_matrix(self) -> np.ndarray:
        """Return the fitted transition matrix."""
        return self.model.transmat_

    def visualize_regime_hierarchy(
        self,
        local_states: Dict[str, np.ndarray],
        global_states: np.ndarray,
        tickers: List[str],
        timestamps=None,
        save_path=None,
    ):
        """Visualize agreement between local regimes and global regime."""
        n_tickers = len(tickers)
        fig, axes = plt.subplots(n_tickers + 1, 1, figsize=(16, 3 * (n_tickers + 1)), sharex=True)

        if timestamps is None:
            timestamps = np.arange(len(global_states))

        # Global regime on top
        ax = axes[0]
        im = ax.imshow(
            global_states.reshape(1, -1), aspect='auto', cmap='viridis',
            vmin=0, vmax=self.n_global_regimes - 1, interpolation='nearest',
        )
        ax.set_ylabel('Global\nRegime', fontweight='bold', fontsize=10)
        ax.set_yticks([])
        ax.set_title(
            'Hierarchy: Global Regime (Meta-HMM) vs Local Regimes (per-asset HMM)',
            fontweight='bold', fontsize=12, pad=20,
        )

        # Local regimes
        for i, ticker in enumerate(tickers, start=1):
            ax = axes[i]
            im = ax.imshow(
                local_states[ticker].reshape(1, -1), aspect='auto', cmap='viridis',
                vmin=0, vmax=2, interpolation='nearest',
            )
            ax.set_ylabel(ticker, fontweight='bold', fontsize=10)
            ax.set_yticks([])

        axes[-1].set_xlabel('Time (observations)', fontweight='bold', fontsize=10)

        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
        cbar.set_label('Regime', fontweight='bold', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("OK Hierarchy plot saved: %s", save_path)

        return fig

    def compute_regime_synchronization(
        self,
        local_states: Dict[str, np.ndarray],
        global_states: np.ndarray,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Measure synchronization between local and global regimes.

        Parameters
        ----------
        local_states : dict
            ``{ticker: state_sequence}``.
        global_states : np.ndarray
            Global state sequence.
        tickers : list of str

        Returns
        -------
        pd.DataFrame
            Synchronization metrics per ticker, sorted by leadership score.
        """
        results = []

        for ticker in tickers:
            local = local_states[ticker]

            # Co-transitions: how often the ticker changes simultaneously with the global regime
            local_transitions = np.diff(local) != 0
            global_transitions = np.diff(global_states) != 0
            co_transitions = np.sum(local_transitions & global_transitions)
            total_global_transitions = np.sum(global_transitions)

            # Synchronization rate
            if total_global_transitions > 0:
                sync_rate = co_transitions / total_global_transitions
            else:
                sync_rate = 0.0

            # Leadership: does the ticker lead the global regime?
            # Count how often local_transition[t] precedes global_transition[t+lag]
            max_lag = 10
            lead_count = 0
            lag_count = 0

            for lag in range(1, max_lag + 1):
                # Lead: local precedes global
                if lag < len(local_transitions):
                    lead_count += np.sum(local_transitions[:-lag] & global_transitions[lag:])

                # Lag: local follows global
                if lag < len(global_transitions):
                    lag_count += np.sum(global_transitions[:-lag] & local_transitions[lag:])

            if lead_count + lag_count > 0:
                leadership_score = (lead_count - lag_count) / (lead_count + lag_count)
            else:
                leadership_score = 0.0

            results.append({
                'ticker': ticker,
                'sync_rate': sync_rate,
                'co_transitions': co_transitions,
                'total_transitions_global': total_global_transitions,
                'leadership_score': leadership_score,
                'leads': lead_count,
                'lags': lag_count
            })

        sync_df = pd.DataFrame(results)
        sync_df = sync_df.sort_values('leadership_score', ascending=False)

        return sync_df


def fit_hierarchical_hmm_pipeline(
    local_state_probs: Dict[str, np.ndarray],
    local_states: Dict[str, np.ndarray],
    tickers: List[str],
    n_global_regimes: int = 3,
    persistence: float = 0.95,
    smooth_window: int = 30
) -> Tuple[MetaHMM, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    End-to-end pipeline for the hierarchical HMM.

    Parameters
    ----------
    local_state_probs : dict
        State probabilities from local HMMs.
    local_states : dict
        Viterbi state sequences from local HMMs.
    tickers : list of str
    n_global_regimes : int
    persistence : float
    smooth_window : int

    Returns
    -------
    meta_hmm : MetaHMM
        Fitted Meta-HMM.
    global_states : np.ndarray
        Smoothed global state sequence.
    global_probs : np.ndarray
        Global regime posterior probabilities.
    sync_df : pd.DataFrame
        Local-to-global synchronization metrics.
    """
    meta_hmm = MetaHMM(n_global_regimes=n_global_regimes, persistence=persistence)
    meta_hmm.fit(local_state_probs, tickers)

    global_states = meta_hmm.predict_global_states(
        local_state_probs, tickers, smooth_window=smooth_window
    )
    global_probs = meta_hmm.predict_global_probs(local_state_probs, tickers)

    sync_df = meta_hmm.compute_regime_synchronization(local_states, global_states, tickers)

    logger.info("Local → Global synchronization:\n%s",
                sync_df[['ticker', 'sync_rate', 'leadership_score']].to_string(index=False))

    leader = sync_df.iloc[0]
    logger.info(
        "OK Patient Zero (contagion leader): %s | leadership=%.3f | sync=%.1f%%",
        leader['ticker'], leader['leadership_score'], leader['sync_rate'] * 100,
    )

    return meta_hmm, global_states, global_probs, sync_df


if __name__ == "__main__":
    """Test the Meta-HMM with synthetic data."""

    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    n_obs = 1000
    n_tickers = 5
    n_local_regimes = 3

    # Simulated correlated state probabilities (mimicking contagion)
    base_probs = np.random.dirichlet(np.ones(n_local_regimes), size=n_obs)

    local_probs = {}
    for i in range(n_tickers):
        # Each asset has slightly different but correlated probabilities
        noise = np.random.normal(0, 0.1, size=(n_obs, n_local_regimes))
        probs = np.abs(base_probs + noise)
        probs = probs / probs.sum(axis=1, keepdims=True)
        local_probs[f'TICK{i}'] = probs

    meta_hmm = MetaHMM(n_global_regimes=3, persistence=0.95)
    meta_hmm.fit(local_probs, [f'TICK{i}' for i in range(n_tickers)])

    meta_hmm.predict_global_states(local_probs, [f'TICK{i}' for i in range(n_tickers)])

    logger.info("OK Meta-HMM test passed.")

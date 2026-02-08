"""
Méta-HMM Hiérarchique - Architecture à 2 niveaux
=================================================

INNOVATION MAJEURE : HMM de second ordre pour détection de contagion.

Architecture :
--------------
Niveau 1 (Local) : HMM par actif → Probabilités P(état | actif)
Niveau 2 (Global) : Méta-HMM observe toutes les probabilités → Régime sectoriel

Avantages :
-----------
1. Résout le "Label Switching" : Le Méta-HMM réaligne les sémantiques locales
2. Filtre le bruit : Ignore les transitions isolées non confirmées
3. Détecte la contagion : Identifie quand plusieurs actifs changent ensemble
4. Rotation sectorielle : Distingue stress local vs stress systémique

Variables observées par le Méta-HMM :
--------------------------------------
- Probabilités d'état de chaque HMM local (n_tickers × n_regimes)
- Permet de capturer des patterns complexes de co-occurrence

Exemple :
---------
Si HMM(AAPL) dit "80% stress", HMM(GOOG) dit "70% stress",
mais HMM(MSFT) dit "10% stress", le Méta-HMM peut détecter
une "Contagion partielle Tech" vs "Panique généralisée".
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns


class MetaHMM:
    """
    Méta-HMM hiérarchique pour détection de régimes sectoriels.

    Observe les probabilités d'état des HMM locaux et détecte
    des patterns de contagion et rotation sectorielle.
    """

    def __init__(
        self,
        n_global_regimes: int = 3,
        persistence: float = 0.95,
        covariance_type: str = 'diag'
    ):
        """
        Initialise le Méta-HMM.

        Args:
            n_global_regimes: Nombre de régimes globaux (sectoriels)
            persistence: Probabilité de rester dans le même régime
            covariance_type: Type de covariance ('diag', 'full', 'tied')
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
        Fit le Méta-HMM sur les probabilités d'état des HMM locaux.

        Args:
            local_state_probs: Dict {ticker: state_probs array (n_obs, n_regimes)}
            tickers: Liste des tickers

        Returns:
            self (fitted)
        """
        print("\n" + "="*70)
        print("FIT MÉTA-HMM GLOBAL")
        print("="*70)

        # Concaténation des probabilités de tous les actifs
        # Shape finale : (n_obs, n_tickers × n_regimes_local)
        prob_arrays = [local_state_probs[ticker] for ticker in tickers]
        X_meta = np.hstack(prob_arrays)

        print(f"\nDimensions des features Méta-HMM :")
        print(f"  - Nombre d'actifs : {len(tickers)}")
        print(f"  - Régimes locaux par actif : {prob_arrays[0].shape[1]}")
        print(f"  - Features totales : {X_meta.shape[1]} ({len(tickers)} × {prob_arrays[0].shape[1]})")
        print(f"  - Observations : {X_meta.shape[0]}")

        # Standardisation (optionnel, mais recommandé)
        self.feature_mean = np.mean(X_meta, axis=0)
        self.feature_std = np.std(X_meta, axis=0) + 1e-9
        X_scaled = (X_meta - self.feature_mean) / self.feature_std

        # Fit HMM global
        print(f"\nFitting Méta-HMM avec {self.n_global_regimes} régimes globaux...")

        self.model = hmm.GaussianHMM(
            n_components=self.n_global_regimes,
            covariance_type=self.covariance_type,
            n_iter=1000,
            random_state=42,
            init_params='stmc'
        )
        self.model.fit(X_scaled)

        # Forcer persistance (régimes globaux plus stables)
        transmat_persistent = np.ones(
            (self.n_global_regimes, self.n_global_regimes)
        ) * (1 - self.persistence) / (self.n_global_regimes - 1)
        np.fill_diagonal(transmat_persistent, self.persistence)
        self.model.transmat_ = transmat_persistent

        self.is_fitted = True

        print(f"✓ Méta-HMM fitted avec persistance = {self.persistence}")

        return self

    def predict_global_states(
        self,
        local_state_probs: Dict[str, np.ndarray],
        tickers: List[str],
        smooth_window: int = 30
    ) -> np.ndarray:
        """
        Prédit les états globaux (sectoriels) à partir des probabilités locales.

        Args:
            local_state_probs: Probabilités d'état des HMM locaux
            tickers: Liste des tickers
            smooth_window: Fenêtre de lissage (plus grande pour niveaux globaux)

        Returns:
            États globaux (np.ndarray)
        """
        if not self.is_fitted:
            raise ValueError("Méta-HMM non fitted. Appelez .fit() d'abord.")

        # Concaténation
        prob_arrays = [local_state_probs[ticker] for ticker in tickers]
        X_meta = np.hstack(prob_arrays)

        # Standardisation
        X_scaled = (X_meta - self.feature_mean) / self.feature_std

        # Prédiction
        global_states_raw = self.model.predict(X_scaled)

        # Lissage (plus agressif pour niveaux globaux)
        global_states_smooth = global_states_raw.copy()
        n = len(global_states_raw)

        for i in range(smooth_window, n - smooth_window):
            window_states = global_states_raw[i-smooth_window:i+smooth_window]
            majority = np.bincount(window_states).argmax()
            global_states_smooth[i] = majority

        # Statistiques
        print("\n" + "="*70)
        print("DISTRIBUTION DES RÉGIMES GLOBAUX (MÉTA-HMM)")
        print("="*70)

        unique, counts = np.unique(global_states_smooth, return_counts=True)
        for s, c in zip(unique, counts):
            print(f"Régime Global {s}: {c:,} obs ({c/len(global_states_smooth)*100:.1f}%)")

        n_transitions = np.sum(np.diff(global_states_smooth) != 0)
        avg_duration = len(global_states_smooth) / (n_transitions + 1) * 0.5
        print(f"\nDurée moyenne : {avg_duration:.1f}s")
        print(f"Transitions : {n_transitions}")

        return global_states_smooth

    def predict_global_probs(
        self,
        local_state_probs: Dict[str, np.ndarray],
        tickers: List[str]
    ) -> np.ndarray:
        """
        Probabilités d'appartenance aux régimes globaux.

        Args:
            local_state_probs: Probabilités d'état des HMM locaux
            tickers: Liste des tickers

        Returns:
            Probabilités globales (n_obs, n_global_regimes)
        """
        if not self.is_fitted:
            raise ValueError("Méta-HMM non fitted.")

        # Concaténation
        prob_arrays = [local_state_probs[ticker] for ticker in tickers]
        X_meta = np.hstack(prob_arrays)

        # Standardisation
        X_scaled = (X_meta - self.feature_mean) / self.feature_std

        # Probabilités
        global_probs = self.model.predict_proba(X_scaled)

        return global_probs

    def get_transition_matrix(self) -> np.ndarray:
        """Retourne la matrice de transition du Méta-HMM."""
        if not self.is_fitted:
            raise ValueError("Méta-HMM non fitted.")

        return self.model.transmat_

    def visualize_regime_agreement(
        self,
        local_states: Dict[str, np.ndarray],
        global_states: np.ndarray,
        tickers: List[str],
        timestamps: pd.Index = None,
        save_path: str = None
    ):
        """
        Visualise l'accord entre régimes locaux et régime global.

        Args:
            local_states: Dict {ticker: states array}
            global_states: États globaux du Méta-HMM
            tickers: Liste des tickers
            timestamps: Index temporel
            save_path: Chemin de sauvegarde (optionnel)
        """
        n_tickers = len(tickers)
        fig, axes = plt.subplots(n_tickers + 1, 1, figsize=(16, 3*(n_tickers+1)), sharex=True)

        if timestamps is None:
            timestamps = np.arange(len(global_states))

        # Régime global en haut
        ax = axes[0]
        regime_colors = global_states.reshape(1, -1)
        im = ax.imshow(regime_colors, aspect='auto', cmap='viridis',
                       vmin=0, vmax=self.n_global_regimes-1, interpolation='nearest')
        ax.set_ylabel('Régime\nGlobal', fontweight='bold', fontsize=10)
        ax.set_yticks([])
        ax.set_title('Hiérarchie : Régime Global (Méta-HMM) vs Régimes Locaux (HMM par actif)',
                     fontweight='bold', fontsize=12, pad=20)

        # Régimes locaux
        for i, ticker in enumerate(tickers, start=1):
            ax = axes[i]
            regime_colors = local_states[ticker].reshape(1, -1)
            im = ax.imshow(regime_colors, aspect='auto', cmap='viridis',
                           vmin=0, vmax=2, interpolation='nearest')  # Assume 3 régimes locaux
            ax.set_ylabel(ticker, fontweight='bold', fontsize=10)
            ax.set_yticks([])

        # Configuration
        axes[-1].set_xlabel('Temps (observations)', fontweight='bold', fontsize=10)

        # Colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='vertical',
                           fraction=0.02, pad=0.01)
        cbar.set_label('Régime', fontweight='bold', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualisation sauvegardée : {save_path}")

        return fig

    def compute_regime_synchronization(
        self,
        local_states: Dict[str, np.ndarray],
        global_states: np.ndarray,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Mesure la synchronisation entre régimes locaux et global.

        Args:
            local_states: Dict {ticker: states}
            global_states: États globaux
            tickers: Liste des tickers

        Returns:
            DataFrame avec métriques de synchronisation
        """
        results = []

        for ticker in tickers:
            local = local_states[ticker]

            # Co-transitions : combien de fois le ticker change en même temps que le global
            local_transitions = np.diff(local) != 0
            global_transitions = np.diff(global_states) != 0
            co_transitions = np.sum(local_transitions & global_transitions)
            total_global_transitions = np.sum(global_transitions)

            # Taux de synchronisation
            if total_global_transitions > 0:
                sync_rate = co_transitions / total_global_transitions
            else:
                sync_rate = 0.0

            # Leadership : est-ce que le ticker anticipe le global ?
            # Compte combien de fois local_transition[t] précède global_transition[t+lag]
            max_lag = 10  # ±5 secondes à 500ms
            lead_count = 0
            lag_count = 0

            for lag in range(1, max_lag+1):
                # Lead : local précède global
                if lag < len(local_transitions):
                    lead_count += np.sum(local_transitions[:-lag] & global_transitions[lag:])

                # Lag : local suit global
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
    Pipeline complet pour HMM hiérarchique.

    Args:
        local_state_probs: Probabilités des HMM locaux
        local_states: États des HMM locaux
        tickers: Liste des tickers
        n_global_regimes: Nombre de régimes globaux
        persistence: Persistance du Méta-HMM
        smooth_window: Fenêtre de lissage

    Returns:
        meta_hmm: Méta-HMM fitted
        global_states: États globaux
        global_probs: Probabilités globales
        sync_df: DataFrame de synchronisation
    """
    # Fit Méta-HMM
    meta_hmm = MetaHMM(
        n_global_regimes=n_global_regimes,
        persistence=persistence
    )
    meta_hmm.fit(local_state_probs, tickers)

    # Prédiction
    global_states = meta_hmm.predict_global_states(
        local_state_probs,
        tickers,
        smooth_window=smooth_window
    )

    global_probs = meta_hmm.predict_global_probs(
        local_state_probs,
        tickers
    )

    # Synchronisation
    sync_df = meta_hmm.compute_regime_synchronization(
        local_states,
        global_states,
        tickers
    )

    print("\n" + "="*70)
    print("SYNCHRONISATION RÉGIMES LOCAUX → GLOBAL")
    print("="*70)
    print(sync_df[['ticker', 'sync_rate', 'leadership_score']].to_string(index=False))

    # Identification du "Patient Zéro"
    leader = sync_df.iloc[0]
    print(f"\n✓ 'Patient Zéro' (leader de contagion) : {leader['ticker']}")
    print(f"  Leadership score : {leader['leadership_score']:.3f}")
    print(f"  Sync rate : {leader['sync_rate']:.1%}")

    return meta_hmm, global_states, global_probs, sync_df


if __name__ == "__main__":
    """Test du Méta-HMM."""

    # Simulation de données
    np.random.seed(42)
    n_obs = 1000
    n_tickers = 5
    n_local_regimes = 3

    # Probabilités d'état simulées (corrélées pour simuler contagion)
    base_probs = np.random.dirichlet(np.ones(n_local_regimes), size=n_obs)

    local_probs = {}
    for i in range(n_tickers):
        # Chaque actif a des probas légèrement différentes mais corrélées
        noise = np.random.normal(0, 0.1, size=(n_obs, n_local_regimes))
        probs = base_probs + noise
        probs = np.abs(probs)  # Positif
        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalisation
        local_probs[f'TICK{i}'] = probs

    # Test du Méta-HMM
    meta_hmm = MetaHMM(n_global_regimes=3, persistence=0.95)
    meta_hmm.fit(local_probs, [f'TICK{i}' for i in range(n_tickers)])

    global_states = meta_hmm.predict_global_states(
        local_probs,
        [f'TICK{i}' for i in range(n_tickers)]
    )

    print("\n✓ Test du Méta-HMM réussi !")

"""
Métriques de Contagion - Transfer Entropy & Regime Correlation
===============================================================

Mesure la propagation de l'information entre actifs via :
1. Transfer Entropy (TE) : Information causale dirigée
2. Corrélation de Régimes : Co-mouvement des états
3. Lead-Lag sur Probabilités : Anticip

ation temporelle

Innovation : Mesure si les actifs **changent de comportement** ensemble,
pas juste si leurs prix bougent ensemble (corrélation classique).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import entropy
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')


def compute_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    k: int = 1,
    bins: int = 10
) -> float:
    """
    Calcule l'entropie de transfert de source → target.

    TE(X → Y) mesure l'information que X apporte sur le futur de Y,
    au-delà de ce que Y sait déjà sur son propre passé.

    TE(X→Y) = I(Y_future ; X_past | Y_past)

    Args:
        source: Série source (cause potentielle)
        target: Série target (effet potentiel)
        k: Ordre temporel (lag)
        bins: Nombre de bins pour discrétisation

    Returns:
        Transfer entropy (en nats, ≥ 0)
    """
    n = len(source)

    # Préparation des séries décalées
    target_future = target[k:]
    target_past = target[:-k]
    source_past = source[:-k]

    # Discrétisation (histogrammes)
    target_future_binned = np.digitize(target_future, np.linspace(target_future.min(), target_future.max(), bins))
    target_past_binned = np.digitize(target_past, np.linspace(target_past.min(), target_past.max(), bins))
    source_past_binned = np.digitize(source_past, np.linspace(source_past.min(), source_past.max(), bins))

    # Calcul des probabilités jointes
    # P(Y_future, Y_past, X_past)
    joint_all = np.histogramdd(
        np.array([target_future_binned, target_past_binned, source_past_binned]).T,
        bins=(bins, bins, bins)
    )[0] / len(target_future_binned)

    # P(Y_future, Y_past)
    joint_target = np.histogramdd(
        np.array([target_future_binned, target_past_binned]).T,
        bins=(bins, bins)
    )[0] / len(target_future_binned)

    # P(Y_past, X_past)
    joint_past = np.histogramdd(
        np.array([target_past_binned, source_past_binned]).T,
        bins=(bins, bins)
    )[0] / len(target_past_binned)

    # P(Y_past)
    prob_target_past = np.bincount(target_past_binned, minlength=bins+1)[1:] / len(target_past_binned)

    # Transfer Entropy via entropies
    # TE = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    # Équivalent à une information mutuelle conditionnelle

    te = 0.0
    for i in range(bins):
        for j in range(bins):
            for k_bin in range(bins):
                p_all = joint_all[i, j, k_bin]
                p_target = joint_target[i, j]
                p_past = joint_past[j, k_bin]
                p_y_past = prob_target_past[j]

                if p_all > 0 and p_target > 0 and p_past > 0 and p_y_past > 0:
                    te += p_all * np.log((p_all * p_y_past) / (p_target * p_past))

    return max(te, 0)  # TE est toujours ≥ 0


def compute_transfer_entropy_matrix(
    state_probs: Dict[str, np.ndarray],
    tickers: List[str],
    k: int = 1,
    bins: int = 10
) -> pd.DataFrame:
    """
    Matrice de transfer entropy entre tous les actifs.

    Args:
        state_probs: Dict {ticker: proba array (n_obs, n_regimes)}
        tickers: Liste des tickers
        k: Lag temporel
        bins: Bins pour discrétisation

    Returns:
        DataFrame (source × target) avec TE values
    """
    print("\n" + "="*70)
    print("CALCUL DE LA MATRICE DE TRANSFER ENTROPY")
    print("="*70)
    print(f"  Lag = {k} ({k*0.5:.1f}s), Bins = {bins}")

    # On utilise la proba du régime de stress (régime 1 ou 2)
    # Ou la somme des probas de stress
    stress_probs = {}
    for ticker in tickers:
        probs = state_probs[ticker]
        if probs.shape[1] == 3:
            # Somme des régimes 1 et 2 (stress)
            stress_probs[ticker] = probs[:, 1] + probs[:, 2]
        else:
            # Utilise la dernière colonne
            stress_probs[ticker] = probs[:, -1]

    # Calcul de la matrice TE
    te_matrix = np.zeros((len(tickers), len(tickers)))

    for i, source_ticker in enumerate(tickers):
        for j, target_ticker in enumerate(tickers):
            if i != j:
                te = compute_transfer_entropy(
                    stress_probs[source_ticker],
                    stress_probs[target_ticker],
                    k=k,
                    bins=bins
                )
                te_matrix[i, j] = te

    te_df = pd.DataFrame(
        te_matrix,
        index=tickers,
        columns=tickers
    )

    print(f"\n✓ Matrice TE calculée")
    print(f"  TE moyen : {te_df.values[te_df.values > 0].mean():.4f} nats")
    print(f"  TE max : {te_df.values.max():.4f} nats")

    # Top 5 relations
    print(f"\nTop 5 relations de Transfer Entropy :")
    te_flat = []
    for i, source in enumerate(tickers):
        for j, target in enumerate(tickers):
            if i != j:
                te_flat.append({
                    'source': source,
                    'target': target,
                    'te': te_matrix[i, j]
                })

    te_flat_df = pd.DataFrame(te_flat).sort_values('te', ascending=False)
    print(te_flat_df.head().to_string(index=False))

    return te_df


def compute_regime_correlation(
    state_probs: Dict[str, np.ndarray],
    tickers: List[str],
    max_lag: int = 10
) -> pd.DataFrame:
    """
    Corrélation croisée des probabilités de régime.

    Mesure si deux actifs changent de régime en même temps.

    Args:
        state_probs: Dict {ticker: proba array}
        tickers: Liste des tickers
        max_lag: Lag maximum à tester

    Returns:
        DataFrame avec corrélations et lags optimaux
    """
    print("\n" + "="*70)
    print("CORRÉLATION DE RÉGIMES (CROSS-CORRELATION)")
    print("="*70)
    print(f"  Lag maximum = ±{max_lag} (±{max_lag*0.5:.1f}s)")

    results = []

    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i < j:  # Éviter les doublons

                # Utilise la proba de stress (somme des régimes 1 et 2)
                probs1 = state_probs[ticker1]
                probs2 = state_probs[ticker2]

                if probs1.shape[1] == 3:
                    stress1 = probs1[:, 1] + probs1[:, 2]
                    stress2 = probs2[:, 1] + probs2[:, 2]
                else:
                    stress1 = probs1[:, -1]
                    stress2 = probs2[:, -1]

                # Corrélation croisée
                correlation = correlate(stress1, stress2, mode='same')
                correlation = correlation / (np.std(stress1) * np.std(stress2) * len(stress1))

                # Trouver le lag du max
                center = len(correlation) // 2
                search_start = max(0, center - max_lag)
                search_end = min(len(correlation), center + max_lag + 1)

                local_corr = correlation[search_start:search_end]
                max_idx = np.argmax(np.abs(local_corr))
                optimal_lag = max_idx - (center - search_start)
                max_corr = local_corr[max_idx]

                results.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'max_correlation': max_corr,
                    'optimal_lag': optimal_lag,
                    'lag_seconds': optimal_lag * 0.5,
                    'zero_lag_corr': correlation[center]
                })

    corr_df = pd.DataFrame(results)
    corr_df = corr_df.sort_values('max_correlation', ascending=False, key=abs)

    print(f"\n✓ Corrélations de régimes calculées")
    print(f"  Corrélation moyenne (lag=0) : {corr_df['zero_lag_corr'].mean():.3f}")
    print(f"  Corrélation max : {corr_df['max_correlation'].abs().max():.3f}")

    print(f"\nTop 5 paires corrélées :")
    print(corr_df.head()[['ticker1', 'ticker2', 'max_correlation', 'optimal_lag', 'lag_seconds']].to_string(index=False))

    return corr_df


def identify_patient_zero(
    te_matrix: pd.DataFrame,
    sync_df: pd.DataFrame
) -> Dict:
    """
    Identifie le "Patient Zéro" de la contagion.

    Combine :
    1. Transfer Entropy sortante (qui cause le plus)
    2. Leadership score (qui anticipe le global)

    Args:
        te_matrix: Matrice de Transfer Entropy
        sync_df: DataFrame de synchronisation du Méta-HMM

    Returns:
        Dict avec le patient zéro et les métriques
    """
    print("\n" + "="*70)
    print("IDENTIFICATION DU 'PATIENT ZÉRO'")
    print("="*70)

    # TE sortante moyenne (combien je cause les autres)
    te_outgoing = te_matrix.sum(axis=1) / (len(te_matrix) - 1)  # Moyenne hors diagonale
    te_outgoing_df = te_outgoing.to_frame('te_outgoing').reset_index()
    te_outgoing_df.columns = ['ticker', 'te_outgoing']

    # Merge avec leadership
    combined = sync_df.merge(te_outgoing_df, on='ticker')

    # Score combiné (normalisation puis somme)
    combined['te_norm'] = (combined['te_outgoing'] - combined['te_outgoing'].min()) / (combined['te_outgoing'].max() - combined['te_outgoing'].min() + 1e-9)
    combined['leadership_norm'] = (combined['leadership_score'] - combined['leadership_score'].min()) / (combined['leadership_score'].max() - combined['leadership_score'].min() + 1e-9)

    combined['contagion_score'] = combined['te_norm'] + combined['leadership_norm']
    combined = combined.sort_values('contagion_score', ascending=False)

    patient_zero = combined.iloc[0]

    print(f"\n✓ Patient Zéro identifié : {patient_zero['ticker']}")
    print(f"  Contagion Score : {patient_zero['contagion_score']:.3f}")
    print(f"  Transfer Entropy sortante : {patient_zero['te_outgoing']:.4f} nats")
    print(f"  Leadership Score : {patient_zero['leadership_score']:.3f}")
    print(f"  Sync Rate : {patient_zero['sync_rate']:.1%}")

    print(f"\nRanking complet des actifs (par potentiel de contagion) :")
    print(combined[['ticker', 'contagion_score', 'te_outgoing', 'leadership_score']].to_string(index=False))

    return {
        'patient_zero': patient_zero['ticker'],
        'contagion_score': patient_zero['contagion_score'],
        'te_outgoing': patient_zero['te_outgoing'],
        'leadership_score': patient_zero['leadership_score'],
        'ranking': combined
    }


def visualize_contagion_network(
    te_matrix: pd.DataFrame,
    patient_zero_info: Dict,
    save_path: str = None
):
    """
    Visualise le réseau de contagion avec Transfer Entropy.

    Args:
        te_matrix: Matrice de TE
        patient_zero_info: Info du patient zéro
        save_path: Chemin de sauvegarde
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        print("\n" + "="*70)
        print("VISUALISATION DU RÉSEAU DE CONTAGION")
        print("="*70)

        # Créer graphe dirigé
        G = nx.DiGraph()
        tickers = te_matrix.index.tolist()
        G.add_nodes_from(tickers)

        # Ajouter arêtes (seuil : TE > moyenne)
        te_threshold = te_matrix.values[te_matrix.values > 0].mean()

        for source in tickers:
            for target in tickers:
                te_value = te_matrix.loc[source, target]
                if te_value > te_threshold:
                    G.add_edge(source, target, weight=te_value)

        # Visualisation
        fig, ax = plt.subplots(figsize=(12, 12))

        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Couleur des nœuds (patient zéro en rouge)
        node_colors = []
        for node in G.nodes():
            if node == patient_zero_info['patient_zero']:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')

        # Taille des nœuds proportionnelle au TE sortant
        te_outgoing = te_matrix.sum(axis=1)
        node_sizes = [3000 * (1 + te_outgoing[node] / te_outgoing.max()) for node in G.nodes()]

        # Dessiner
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                              edgecolors='black', linewidths=2, ax=ax)

        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)

        # Arêtes avec largeur proportionnelle au TE
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1

        for (u, v), w in zip(edges, weights):
            width = 5 * (w / max_weight)
            nx.draw_networkx_edges(G, pos, [(u, v)], width=width,
                                  edge_color='gray', alpha=0.6,
                                  arrowsize=20, ax=ax,
                                  connectionstyle='arc3,rad=0.1')

        ax.set_title(f'Réseau de Contagion (Transfer Entropy)\nPatient Zéro : {patient_zero_info["patient_zero"]}',
                    fontweight='bold', fontsize=14)
        ax.axis('off')

        # Légende
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=15, label=f'Patient Zéro ({patient_zero_info["patient_zero"]})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                  markersize=15, label='Autres actifs'),
            Line2D([0], [0], color='gray', linewidth=3, label='TE > moyenne (causalité)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Réseau de contagion sauvegardé : {save_path}")

        return fig

    except ImportError:
        print("⚠ networkx non disponible, visualisation ignorée")
        return None


if __name__ == "__main__":
    """Test des métriques de contagion."""

    # Données simulées
    np.random.seed(42)
    n_obs = 1000
    tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'INTC']

    # Simuler des probabilités de stress avec contagion
    # AAPL cause GOOG, qui cause les autres
    aapl_stress = np.random.rand(n_obs)
    goog_stress = 0.7 * np.roll(aapl_stress, 2) + 0.3 * np.random.rand(n_obs)
    msft_stress = 0.6 * np.roll(goog_stress, 1) + 0.4 * np.random.rand(n_obs)
    amzn_stress = 0.5 * np.roll(goog_stress, 3) + 0.5 * np.random.rand(n_obs)
    intc_stress = 0.4 * np.roll(msft_stress, 1) + 0.6 * np.random.rand(n_obs)

    # Conversion en probas de régime (3 régimes)
    state_probs = {}
    for ticker, stress in zip(tickers, [aapl_stress, goog_stress, msft_stress, amzn_stress, intc_stress]):
        probs = np.zeros((n_obs, 3))
        probs[:, 0] = 1 - stress  # Calme
        probs[:, 1] = stress * 0.6  # Stress modéré
        probs[:, 2] = stress * 0.4  # Stress élevé
        state_probs[ticker] = probs

    # Test Transfer Entropy
    te_matrix = compute_transfer_entropy_matrix(state_probs, tickers, k=2, bins=10)

    print("\n✓ Test des métriques de contagion réussi !")

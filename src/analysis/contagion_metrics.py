from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import correlate

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def _bin_series(x: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(0, dtype=int)
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    edges = np.linspace(xmin, xmax, bins + 1)
    return np.digitize(x, edges[1:-1], right=False)


def _transfer_entropy_from_binned(
    target_future_binned: np.ndarray,
    target_past_binned: np.ndarray,
    source_past_binned: np.ndarray,
    bins: int,
) -> float:
    n = len(target_future_binned)
    if n == 0:
        return 0.0

    joint_all = np.histogramdd(
        np.stack([target_future_binned, target_past_binned, source_past_binned], axis=1),
        bins=(bins, bins, bins),
        range=((0, bins), (0, bins), (0, bins)),
    )[0]
    joint_target = np.histogramdd(
        np.stack([target_future_binned, target_past_binned], axis=1),
        bins=(bins, bins),
        range=((0, bins), (0, bins)),
    )[0]
    joint_past = np.histogramdd(
        np.stack([target_past_binned, source_past_binned], axis=1),
        bins=(bins, bins),
        range=((0, bins), (0, bins)),
    )[0]
    prob_target_past = np.bincount(target_past_binned, minlength=bins)

    p_all    = joint_all    / n
    p_target = joint_target / n
    p_past   = joint_past   / n
    p_y_past = prob_target_past / n

    p_y_past_cube  = p_y_past.reshape(1, bins, 1)
    p_target_cube  = p_target[:, :, None]
    p_past_cube    = p_past[None, :, :]
    mask = (p_all > 0) & (p_target_cube > 0) & (p_past_cube > 0) & (p_y_past_cube > 0)
    if not np.any(mask):
        return 0.0

    num = p_all * p_y_past_cube
    den = p_target_cube * p_past_cube
    return float(max(np.sum(p_all[mask] * np.log(num[mask] / den[mask])), 0.0))


def _block_shuffle(arr: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return arr
    if block_size <= 1 or block_size >= n:
        return rng.permutation(arr)
    blocks = [arr[i : i + block_size] for i in range(0, n, block_size)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))], axis=0)


class ContagionAnalyzer:
    """
    Transfer Entropy contagion analysis and Patient Zero identification.

    All shared hyper-parameters (bins, surrogates, alpha) are set once at
    construction; intermediate results are stored as ``attr_`` attributes.

    Parameters
    ----------
    n_bins : int
        Histogram bins for TE discretisation.
    n_surrogates : int
        Number of block-shuffle surrogates for significance testing.
    block_size : int
        Block size for surrogate generation.
    alpha : float
        Significance threshold for TE p-values.
    random_state : int
        RNG seed for reproducibility.

    Attributes
    ----------
    te_matrix_ : pd.DataFrame or None
        Matrice TE apres ``compute_te_matrix_significance``.
    te_k_summary_ : pd.DataFrame or None
        k-grid summary after ``compute_te_matrix_significance``.
    regime_corr_ : pd.DataFrame or None
        Regime correlation table after ``compute_regime_correlation``.
    patient_zero_info_ : dict or None
        Patient Zero results after ``identify_patient_zero``.
    """

    def __init__(
        self,
        n_bins: int = 10,
        n_surrogates: int = 100,
        block_size: int = 30,
        alpha: float = 0.05,
        random_state: int = 0,
    ) -> None:
        self.n_bins = n_bins
        self.n_surrogates = n_surrogates
        self.block_size = block_size
        self.alpha = alpha
        self.random_state = random_state
        self.te_matrix_: Optional[pd.DataFrame] = None
        self.te_k_summary_: Optional[pd.DataFrame] = None
        self.regime_corr_: Optional[pd.DataFrame] = None
        self.patient_zero_info_: Optional[Dict] = None

    def compute_te_matrix_significance(
        self,
        state_probs: Dict[str, np.ndarray],
        tickers: List[str],
        k_grid: Optional[List[int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        All-pairs TE matrix with block-shuffle significance test and k selection.

        Tries every k in ``k_grid``, picks the one maximising (n_significant, z_mean).
        Results stored in ``self.te_matrix_`` and ``self.te_k_summary_``.

        Returns
        -------
        te_df_best : pd.DataFrame
            TE matrix for the best k.
        summary_df : pd.DataFrame
            Rows: k, n_significant, z_mean — sorted descending.
        """
        if not k_grid:
            k_grid = [1]

        logger.info("--- transfer entropy with significance (surrogates) ---")
        logger.info(
            "k_grid=%s, bins=%d, surrogates=%d, block_size=%d",
            k_grid, self.n_bins, self.n_surrogates, self.block_size,
        )

        stress = self._extract_stress_probs(state_probs, tickers)
        binned = {t: _bin_series(stress[t], self.n_bins) for t in tickers}
        rng = np.random.default_rng(self.random_state)

        summaries = []
        best_k = None
        best_score = (-np.inf, -np.inf)
        te_best = None

        for k in k_grid:
            te_matrix = np.zeros((len(tickers), len(tickers)))
            p_matrix  = np.full((len(tickers), len(tickers)), np.nan)
            z_matrix  = np.full((len(tickers), len(tickers)), np.nan)

            for i, src in enumerate(tickers):
                src_full = binned[src]
                surrogates = (
                    [_block_shuffle(src_full, self.block_size, rng) for _ in range(self.n_surrogates)]
                    if self.n_surrogates > 0 else []
                )

                for j, tgt in enumerate(tickers):
                    if i == j:
                        continue
                    tgt_full = binned[tgt]
                    n = min(len(src_full), len(tgt_full))
                    if n <= k:
                        continue

                    src_t = src_full[:n]
                    tgt_t = tgt_full[:n]
                    tf, tp, sp = tgt_t[k:], tgt_t[:-k], src_t[:-k]

                    te_obs = _transfer_entropy_from_binned(tf, tp, sp, self.n_bins)
                    te_matrix[i, j] = te_obs

                    if not surrogates:
                        continue

                    surr_vals = np.array([
                        _transfer_entropy_from_binned(tf, tp, s[:n][:-k], self.n_bins)
                        for s in surrogates
                    ])
                    mu    = float(np.mean(surr_vals))
                    sigma = float(np.std(surr_vals, ddof=1))
                    z = (te_obs - mu) / (sigma + 1e-9)
                    p = (1.0 + float(np.sum(surr_vals >= te_obs))) / (1.0 + self.n_surrogates)
                    z_matrix[i, j] = z
                    p_matrix[i, j] = p

            off_diag = ~np.eye(len(tickers), dtype=bool)
            n_sig  = int(np.sum((p_matrix < self.alpha) & off_diag))
            z_mean = float(np.nanmean(z_matrix[off_diag]))
            summaries.append({"k": k, "n_significant": n_sig, "z_mean": z_mean})

            score = (n_sig, z_mean)
            if score > best_score:
                best_score = score
                best_k = k
                te_best = pd.DataFrame(te_matrix, index=tickers, columns=tickers)

            logger.debug("k=%d: n_sig=%d, z_mean=%.3f", k, n_sig, z_mean)

        summary_df = pd.DataFrame(summaries).sort_values(
            ["n_significant", "z_mean"], ascending=False
        )
        logger.info("best k=%s (n_sig=%d, z_mean=%.3f)", best_k, best_score[0], best_score[1])

        self.te_matrix_ = te_best
        self.te_k_summary_ = summary_df
        return te_best, summary_df

    def compute_regime_correlation(
        self,
        state_probs: Dict[str, np.ndarray],
        tickers: List[str],
        max_lag: int = 10,
    ) -> pd.DataFrame:
        """
        Cross-correlation of stress probabilities between all ticker pairs.

        Results stored in ``self.regime_corr_``.
        """
        logger.info("--- regime cross-correlation ---")
        logger.info("max lag: +/-%d (+/-%.1fs)", max_lag, max_lag * 0.5)

        stress = self._extract_stress_probs(state_probs, tickers)
        rows = []

        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if i >= j:
                    continue
                s1, s2 = stress[t1], stress[t2]
                corr = correlate(s1, s2, mode="same")
                corr /= np.std(s1) * np.std(s2) * len(s1)

                center = len(corr) // 2
                start  = max(0, center - max_lag)
                end    = min(len(corr), center + max_lag + 1)
                local  = corr[start:end]

                best_idx   = int(np.argmax(np.abs(local)))
                opt_lag    = best_idx - (center - start)
                rows.append(
                    {
                        "ticker1": t1,
                        "ticker2": t2,
                        "max_correlation": local[best_idx],
                        "optimal_lag": opt_lag,
                        "lag_seconds": opt_lag * 0.5,
                        "zero_lag_corr": corr[center],
                    }
                )

        corr_df = pd.DataFrame(rows).sort_values(
            "max_correlation", ascending=False, key=abs
        )
        self.regime_corr_ = corr_df

        logger.info(
            "regime correlations computed: mean zero-lag=%.3f, max=%.3f",
            corr_df["zero_lag_corr"].mean(),
            corr_df["max_correlation"].abs().max(),
        )
        logger.debug(
            "top 5 correlated pairs:\n%s",
            corr_df.head()[["ticker1", "ticker2", "max_correlation", "optimal_lag", "lag_seconds"]].to_string(index=False),
        )

        return corr_df

    def identify_patient_zero(
        self,
        sync_df: pd.DataFrame,
        te_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Identify the Patient Zero of contagion.

        Combines normalised TE outgoing + normalised leadership score.
        If ``te_matrix`` is not provided, uses ``self.te_matrix_``.

        Results stored in ``self.patient_zero_info_``.
        """
        if te_matrix is None:
            if self.te_matrix_ is None:
                raise RuntimeError(
                    "No te_matrix available. Call compute_te_matrix* first or pass te_matrix."
                )
            te_matrix = self.te_matrix_

        logger.info("--- patient zero identification ---")

        te_out = (te_matrix.sum(axis=1) / (len(te_matrix) - 1)).rename("te_outgoing")
        combined = sync_df.merge(te_out.reset_index().rename(columns={"index": "ticker"}), on="ticker")

        def _norm(col):
            mn, mx = col.min(), col.max()
            return (col - mn) / (mx - mn + 1e-9)

        combined["te_norm"]         = _norm(combined["te_outgoing"])
        combined["leadership_norm"] = _norm(combined["leadership_score"])
        combined["contagion_score"] = combined["te_norm"] + combined["leadership_norm"]
        combined = combined.sort_values("contagion_score", ascending=False)

        pz = combined.iloc[0]
        logger.info(
            "patient zero: %s | contagion_score=%.3f, te_outgoing=%.4f nats, "
            "leadership=%.3f, sync=%.1f%%",
            pz["ticker"], pz["contagion_score"], pz["te_outgoing"],
            pz["leadership_score"], pz["sync_rate"] * 100,
        )
        logger.debug(
            "full ranking:\n%s",
            combined[["ticker", "contagion_score", "te_outgoing", "leadership_score"]].to_string(index=False),
        )

        result = {
            "patient_zero":    pz["ticker"],
            "contagion_score": pz["contagion_score"],
            "te_outgoing":     pz["te_outgoing"],
            "leadership_score": pz["leadership_score"],
            "ranking":         combined,
        }
        self.patient_zero_info_ = result
        return result

    def plot_network(
        self,
        te_matrix: Optional[pd.DataFrame] = None,
        patient_zero_info: Optional[Dict] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualise the contagion network with Transfer Entropy as edge weights.

        Falls back to ``self.te_matrix_`` / ``self.patient_zero_info_`` when
        arguments are not provided.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        if te_matrix is None:
            te_matrix = self.te_matrix_
        if patient_zero_info is None:
            patient_zero_info = self.patient_zero_info_
        if te_matrix is None or patient_zero_info is None:
            raise RuntimeError(
                "te_matrix and patient_zero_info required. "
                "Run compute_te_matrix* and identify_patient_zero first."
            )

        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            logger.info("--- contagion network visualisation ---")

            G = nx.DiGraph()
            tickers = te_matrix.index.tolist()
            G.add_nodes_from(tickers)

            threshold = te_matrix.values[te_matrix.values > 0].mean()
            for src in tickers:
                for tgt in tickers:
                    if te_matrix.loc[src, tgt] > threshold:
                        G.add_edge(src, tgt, weight=float(te_matrix.loc[src, tgt]))

            fig, ax = plt.subplots(figsize=(12, 12))
            pos = nx.spring_layout(G, k=1, iterations=50)

            pz = patient_zero_info["patient_zero"]
            node_colors = ["red" if n == pz else "lightblue" for n in G.nodes()]
            te_out = te_matrix.sum(axis=1)
            node_sizes = [3000 * (1 + te_out[n] / te_out.max()) for n in G.nodes()]

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                                   edgecolors="black", linewidths=2, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

            edges  = list(G.edges())
            weights = [G[u][v]["weight"] for u, v in edges]
            max_w = max(weights) if weights else 1.0
            for (u, v), w in zip(edges, weights):
                nx.draw_networkx_edges(
                    G, pos, [(u, v)],
                    width=5 * w / max_w,
                    edge_color="gray", alpha=0.6, arrowsize=20, ax=ax,
                    connectionstyle="arc3,rad=0.1",
                )

            ax.set_title(
                f"Réseau de Contagion (Transfer Entropy)\nPatient Zéro : {pz}",
                fontweight="bold", fontsize=14,
            )
            ax.axis("off")
            ax.legend(
                handles=[
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                           markersize=15, label=f"Patient Zéro ({pz})"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="lightblue",
                           markersize=15, label="Autres actifs"),
                    Line2D([0], [0], color="gray", linewidth=3, label="TE > moyenne"),
                ],
                loc="upper left", fontsize=10,
            )
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info("contagion network saved: %s", save_path)

            return fig

        except ImportError:
            logger.warning("networkx not available, skipping network visualisation")
            return None

    @staticmethod
    def _extract_stress_probs(
        state_probs: Dict[str, np.ndarray],
        tickers: List[str],
    ) -> Dict[str, np.ndarray]:
        """Sum non-calm regime posteriors into a single stress probability series."""
        out = {}
        for t in tickers:
            p = state_probs[t]
            out[t] = p[:, 1] + p[:, 2] if p.shape[1] == 3 else p[:, -1]
        return out

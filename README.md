# Détection de changements de régime et contagion en HFT

Ce dépôt implémente une méthodologie de détection de régimes microstructurels via HMM, optimisés sur des métriques de carnet d’ordres (Price, OFI, OBI) transformées par la distance de Wasserstein temporelle. L’approche est appliquée à 5 tickers du secteur Tech pour estimer des régimes locaux, puis un méta‑HMM global (et un HMM global direct sur les features) pour analyser contagion, lead‑lag et causalité dirigée.

## Idée principale

1. Normalisation robuste par MAD sur fenêtres glissantes.
2. Features de changement de régime via Wasserstein temporel (avant vs après, même actif/métrique).
3. HMM locaux par ticker, paramètres optimisés par ticker.
4. Méta‑HMM global sur les probabilités locales.
5. HMM global direct sur les features Wasserstein agrégées.
6. Diagnostics ARI vs KMeans et MMD pour robustesse et sur‑apprentissage.
7. Analyses lead‑lag et contagion (local↔global et ticker↔ticker).

## Scripts

- `scripts/optimize_hierarchical_parameters.py`
  - Optimise les paramètres HMM locaux par ticker.
  - Optionnellement optimise le méta‑HMM global et le HMM global direct.
  - Sauve les meilleurs paramètres en `.txt` + `.csv` dans `data/results/`.
- `scripts/run_hierarchical_contagion.py`
  - Lance la pipeline complète hiérarchique.
  - Charge les meilleurs paramètres depuis les `.txt` (si présents).
  - Produit toutes les sorties CSV/PNG dans `data/results/`.

## Quickstart

```bash
# Installer les dépendances
pip install -r requirements.txt

# (Optionnel) Accélérer Wasserstein
pip install numba

# 1) Optimiser les paramètres (par défaut: locaux + global méta + global direct)
python scripts/optimize_hierarchical_parameters.py

# 2) Lancer la pipeline finale
python scripts/run_hierarchical_contagion.py
```

## Données

Les fichiers LOBSTER doivent être placés dans `data/raw/`.
Format attendu (exemple) :

```
data/raw/
AAPL_2012-06-21_34200000_57600000_orderbook_5.csv
AAPL_2012-06-21_34200000_57600000_message_5.csv
INTC_2012-06-21_34200000_57600000_orderbook_5.csv
INTC_2012-06-21_34200000_57600000_message_5.csv
... (GOOG, AMZN, MSFT)
```

## Configuration

Le fichier `src/config.py` centralise les paramètres clés :

- `TICKERS`, `ANALYSIS_DATE`
- `MAD_WINDOW`
- `WASSERSTEIN_WINDOW`
- `N_REGIMES_LOCAL`, `N_REGIMES_GLOBAL`
- `HMM_PERSISTENCE_LOCAL`, `HMM_PERSISTENCE_GLOBAL`
- `HMM_SMOOTHING_LOCAL`, `HMM_SMOOTHING_GLOBAL`
- `HMM_COV_FULL_CORR_THRESHOLD`
- Paramètres MMD (`MMD_WINDOW`, `MMD_STEP`, etc.)

Les paramètres optimisés sont chargés automatiquement depuis `data/results/best_parameters_hierarchical*.txt`.

## Sorties principales

Les sorties sont enregistrées dans `data/results/` (gitignored) :

- États locaux et globaux
- Synchronisation local→global
- Lead‑lag local vs global
- Lead‑lag ticker↔ticker (avec p‑values)
- Transfer Entropy
- MMD diagnostics (local et global)
- Event study GOOG
- Visualisations (heatmaps, timelines, concordance)

## Métriques de validation

- **ARI vs KMeans** : mesure d’accord entre segmentation HMM et clustering.
- **MMD** : séparation de distributions entre régimes, par ticker et au niveau global.

## Structure du dépôt

Voir `REPO_STRUCTURE.md` pour une vue d’ensemble à jour.

## Notes

- La pipeline est conçue pour une journée HFT (21/06/2012) et 5 tickers Tech.
- Les résultats dépendent fortement des paramètres (MAD, Wasserstein, persistance/smoothing).

---
Dernière mise à jour : 2026-02-07

# Guide de démarrage rapide

Ce guide couvre la pipeline hiérarchique actuelle (HMM locaux + méta‑HMM global + HMM global direct).

## Installation

```bash
pip install -r requirements.txt

# Optionnel: accélération Wasserstein
pip install numba
```

## Exécution standard

```bash
# Optimisation des paramètres (locaux + global méta + global direct)
python scripts/optimize_hierarchical_parameters.py

# Pipeline complète hiérarchique
python scripts/run_hierarchical_contagion.py
```

## Optimisation ciblée

Le script d’optimisation peut lancer un sous‑ensemble :

```bash
# Uniquement les HMM locaux
python scripts/optimize_hierarchical_parameters.py --locals-only

# Uniquement le méta‑HMM global
python scripts/optimize_hierarchical_parameters.py --meta-only

# Uniquement le HMM global direct
python scripts/optimize_hierarchical_parameters.py --direct-only
```

## Sorties générées

Les sorties sont créées dans `data/results/` :

- Paramètres optimisés en `.txt` et `.csv`
- États locaux et globaux
- Synchronisation et leadership
- Lead‑lag local↔global et ticker↔ticker (p‑values + heatmaps)
- MMD diagnostics local/global
- Transfer Entropy
- Event study GOOG

## Données requises

Placer les fichiers LOBSTER dans `data/raw/` :

```
data/raw/
AAPL_2012-06-21_34200000_57600000_orderbook_5.csv
AAPL_2012-06-21_34200000_57600000_message_5.csv
... (INTC, GOOG, AMZN, MSFT)
```

## Troubleshooting

- ImportError: vérifier `pip install -r requirements.txt`
- Exécution lente: installer `numba` et réduire `WASSERSTEIN_WINDOW` dans `src/config.py`
- Données manquantes: vérifier `data/raw/` et `ANALYSIS_DATE`

---
Version: 2.0
Date: 2026-02-07

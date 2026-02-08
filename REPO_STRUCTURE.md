# Structure du dépôt

```
RIF/
README.md
QUICKSTART.md
REPO_STRUCTURE.md

scripts/
  optimize_hierarchical_parameters.py
  run_hierarchical_contagion.py

src/
  config.py
  data/
  features/
  models/
  analysis/
  visualization/

data/
  raw/        (LOBSTER, non versionné)
  processed/  (intermédiaire, non versionné)
  results/    (sorties, non versionné)

paper/        (si utilisé)
docs/         (notes internes)
```

Notes:

- `data/results/` est ignoré par git.
- Les scripts actifs sont ceux de `scripts/` uniquement.

---
Dernière mise à jour : 2026-02-07

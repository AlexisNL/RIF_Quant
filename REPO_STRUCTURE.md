# Repository Structure

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
  raw/        (LOBSTER, not versioned)
  processed/  (intermediate, not versioned)
  results/    (outputs, not versioned)

paper/        (if used)
docs/         (internal notes)
```

Notes:

- `data/results/` is ignored by git.
- Active scripts are in `scripts/` only.

---
Last update: 2026-02-07

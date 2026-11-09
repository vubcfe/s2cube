# Downstream experiments

Run after finishing the `netherlands` and `kenya` examples. These scripts need scikit-learn and
PyTorch, which s2cube does not depend on.

```bash
pip install scikit-learn torch
python run_experiment.py --project ../netherlands/project.yaml --out results.csv
python run_experiment.py --project ../kenya/project.yaml --out results_kenya.csv
python side_experiments.py --project ../netherlands/project.yaml --out side.json \
  --only unmeasured correlogram purity guard parcel_fit
```

`run_experiment.py` compares corrected and uncorrected cubes under spatial and random splits with
a Random Forest and a small temporal CNN. `--features dates --splits spatial` repeats it on
individual clear dates. `side_experiments.py` runs the sensitivity checks, and needs `dates.csv`
from `s2cube info --refinement`.

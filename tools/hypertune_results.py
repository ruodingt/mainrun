"""Print best trial and trial state counts from Optuna study."""
import json
from collections import Counter
from pathlib import Path
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

db = Path(__file__).parent.parent / "experiments" / "hypertune_v2" / "optuna_v2.db"
study = optuna.load_study(study_name="hypertune_v2", storage=f"sqlite:///{db}")

counts = Counter(t.state.name for t in study.trials)
print(f"Trial states: {dict(counts)}")

completed = [t for t in study.trials if t.state.name == "COMPLETE"]
if not completed:
    print("No completed trials yet.")
else:
    t = study.best_trial
    print(f"Completed:    {len(completed)}")
    print(f"Best val_loss: {t.value:.4f}  (trial #{t.number})")
    print(f"Best params:   {json.dumps(t.params, indent=2)}")


import wandb
import pandas as pd
from pathlib import Path

# Your project settings
PROJECT_DIR = Path(__file__).resolve().parent
ENTITY = "darjih924-pandit-deendayal-energy-university"
PROJECT = "hindi-transliteration"
SWEEP_ID = "t7cy1fvm"

print("Connecting to Weights & Biases...")

api = wandb.Api()
sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")

rows = []

for run in sweep.runs:
    config = {
        key: value
        for key, value in run.config.items()
        if not key.startswith("_")
    }

    summary = dict(run.summary)

    row = {
        "run_id": run.id,
        "run_name": run.name,
        **config,
        "best_dev_character_accuracy": summary.get(
            "best_dev_character_accuracy",
            summary.get("dev_character_accuracy")
        ),
        "best_dev_loss": summary.get(
            "best_dev_loss",
            summary.get("dev_loss")
        ),
        "epochs_run": summary.get("epochs_run"),
        "trainable_parameters": summary.get("trainable_parameters"),
    }

    rows.append(row)

df = pd.DataFrame(rows)

output_file = PROJECT_DIR / "sweep_results.csv"
df.to_csv(output_file, index=False)

print(f"\nExported {len(df)} runs.")
print(f"CSV saved to: {output_file}")
print("\nSweep results:")
print(df.to_string(index=False))

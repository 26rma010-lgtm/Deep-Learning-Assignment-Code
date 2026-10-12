
import wandb
from sweep_config import sweep_config

PROJECT = "hindi-transliteration"

if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep_config,
        project=PROJECT,
    )

    print("\nNew sweep created successfully!")
    print(f"Sweep ID: {sweep_id}")
    print(
        "Sweep URL: "
        f"https://wandb.ai/darjih924-pandit-deendayal-energy-university/"
        f"{PROJECT}/sweeps/{sweep_id}"
    )

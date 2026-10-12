
import wandb
from train import sweep_train

SWEEP_ID = "t7cy1fvm"
PROJECT = "hindi-transliteration"

if __name__ == "__main__":
    print("Starting the new W&B hyperparameter sweep...")

    wandb.agent(
        SWEEP_ID,
        function=sweep_train,
        project=PROJECT,
        count=5,
    )

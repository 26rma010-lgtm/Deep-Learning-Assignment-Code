
import pandas as pd
import matplotlib.pyplot as plt

models = ["RNN", "GRU", "LSTM"]

for model in models:
    file_name = f"history_{model.lower()}.csv"
    df = pd.read_csv(file_name)

    # Graph 1: Training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Training Loss")
    plt.plot(df["epoch"], df["dev_loss"], label="Validation Loss")
    plt.title(f"{model}: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model.lower()}_loss.png", dpi=300)
    plt.close()

    # Graph 2: Training and validation character accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["epoch"],
        df["train_character_accuracy"],
        label="Training Accuracy"
    )
    plt.plot(
        df["epoch"],
        df["dev_character_accuracy"],
        label="Validation Accuracy"
    )
    plt.title(f"{model}: Training vs Validation Character Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Character Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model.lower()}_accuracy.png", dpi=300)
    plt.close()

print("All six training graphs saved successfully!")

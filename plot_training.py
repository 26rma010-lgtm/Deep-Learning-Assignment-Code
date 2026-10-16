
import json
import matplotlib.pyplot as plt
from pathlib import Path

history_path = Path("results/training_history.json")

with open(history_path, "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = [record["epoch"] for record in history]
train_loss = [record["train_loss"] for record in history]
dev_loss = [record["dev_loss"] for record in history]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.plot(epochs, dev_loss, marker="s", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training and Validation Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = Path("results/loss_curve.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Graph saved successfully: {output_path.resolve()}")

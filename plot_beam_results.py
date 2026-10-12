
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "beam_search_results.csv"

df = pd.read_csv(INPUT_FILE).sort_values("beam_size")

# Plot 1: Whole-word accuracy
plt.figure(figsize=(8, 5))
plt.plot(
    df["beam_size"],
    df["whole_word_accuracy"],
    marker="o"
)
plt.xlabel("Beam size")
plt.ylabel("Whole-word accuracy (%)")
plt.title("GRU Transliteration: Accuracy vs Beam Size")
plt.xticks(df["beam_size"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "beam_accuracy_comparison.png",
    dpi=300
)
plt.close()

# Plot 2: Character similarity
plt.figure(figsize=(8, 5))
plt.plot(
    df["beam_size"],
    df["normalized_character_similarity"],
    marker="o"
)
plt.xlabel("Beam size")
plt.ylabel("Character similarity (%)")
plt.title("GRU Transliteration: Character Similarity vs Beam Size")
plt.xticks(df["beam_size"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "beam_similarity_comparison.png",
    dpi=300
)
plt.close()

# Plot 3: Character error rate
plt.figure(figsize=(8, 5))
plt.plot(
    df["beam_size"],
    df["character_error_rate"],
    marker="o"
)
plt.xlabel("Beam size")
plt.ylabel("Character error rate (%)")
plt.title("GRU Transliteration: CER vs Beam Size")
plt.xticks(df["beam_size"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "beam_cer_comparison.png",
    dpi=300
)
plt.close()

print("Created beam-search plots:")
print("beam_accuracy_comparison.png")
print("beam_similarity_comparison.png")
print("beam_cer_comparison.png")

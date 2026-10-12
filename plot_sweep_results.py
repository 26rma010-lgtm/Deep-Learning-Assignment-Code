
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.plotting import parallel_coordinates

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "sweep_results.csv"

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("sweep_results.csv contains no runs.")

# Short labels for the x-axis
df["run_label"] = [
    f"{i + 1}: {cell}"
    for i, cell in enumerate(df["cell_type"])
]

# --------------------------------------------------
# 1. Accuracy vs. run
# --------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(df["run_label"], df["best_dev_character_accuracy"])
plt.xlabel("Run number and cell type")
plt.ylabel("Best development character accuracy (%)")
plt.title("W&B Sweep: Accuracy by Run")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "sweep_accuracy_by_run.png", dpi=300)
plt.close()

# --------------------------------------------------
# 2. Parallel coordinates plot
# Normalize numerical hyperparameters to 0-1
# so their scales are comparable.
# --------------------------------------------------
parallel_columns = [
    "cell_type",
    "embedding_size",
    "hidden_size",
    "encoder_layers",
    "decoder_layers",
    "dropout",
    "learning_rate",
    "best_dev_character_accuracy",
]

parallel_df = df[parallel_columns].copy()

numeric_columns = [
    "embedding_size",
    "hidden_size",
    "encoder_layers",
    "decoder_layers",
    "dropout",
    "learning_rate",
    "best_dev_character_accuracy",
]

for column in numeric_columns:
    minimum = parallel_df[column].min()
    maximum = parallel_df[column].max()

    if maximum > minimum:
        parallel_df[column] = (
            (parallel_df[column] - minimum) / (maximum - minimum)
        )
    else:
        parallel_df[column] = 0.5

plt.figure(figsize=(14, 6))
parallel_coordinates(
    parallel_df,
    class_column="cell_type",
    alpha=0.8,
    linewidth=1.5
)
plt.title("Parallel Coordinates: Sweep Hyperparameters")
plt.ylabel("Normalized value (0 to 1)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "sweep_parallel_coordinates.png", dpi=300)
plt.close()

# --------------------------------------------------
# 3. Correlation table and heatmap
# Use numerical variables only.
# --------------------------------------------------
correlation_columns = [
    "embedding_size",
    "hidden_size",
    "encoder_layers",
    "decoder_layers",
    "dropout",
    "learning_rate",
    "best_dev_character_accuracy",
    "best_dev_loss",
    "trainable_parameters",
]

correlation_df = df[correlation_columns]
correlation = correlation_df.corr()

correlation.to_csv(PROJECT_DIR / "sweep_correlation_table.csv")

plt.figure(figsize=(10, 8))
image = plt.imshow(correlation, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(image, label="Pearson correlation")
plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=75,
    ha="right"
)
plt.yticks(range(len(correlation.index)), correlation.index)

for i in range(len(correlation.index)):
    for j in range(len(correlation.columns)):
        plt.text(
            j, i, f"{correlation.iloc[i, j]:.2f}",
            ha="center", va="center", fontsize=7
        )

plt.title("Hyperparameter and Performance Correlations")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "sweep_correlation_heatmap.png", dpi=300)
plt.close()

print("\nAnalysis completed!")
print("Created:")
print("- sweep_accuracy_by_run.png")
print("- sweep_parallel_coordinates.png")
print("- sweep_correlation_table.csv")
print("- sweep_correlation_heatmap.png")

print("\nCorrelation with development character accuracy:")
print(
    correlation["best_dev_character_accuracy"]
    .sort_values(ascending=False)
    .to_string()
)

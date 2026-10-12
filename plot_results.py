
import pandas as pd
import matplotlib.pyplot as plt

# Read the test results
df = pd.read_csv("test_results.csv")

# Convert accuracy values to percentages
accuracy = df["whole_word_accuracy"] * 100

# Create the graph
plt.figure(figsize=(8, 5))
plt.bar(df["model"], accuracy)

plt.title("Test Whole-Word Accuracy: RNN vs GRU vs LSTM")
plt.xlabel("Model")
plt.ylabel("Whole-Word Accuracy (%)")
plt.ylim(0, 100)

for i, value in enumerate(accuracy):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.savefig("test_accuracy_comparison.png", dpi=300)
plt.show()

print("Graph saved as test_accuracy_comparison.png")

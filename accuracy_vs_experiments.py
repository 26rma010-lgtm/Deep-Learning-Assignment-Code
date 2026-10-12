
import matplotlib.pyplot as plt
from datetime import datetime

# Replace these timestamps with the actual Created timestamps
# from your W&B runs, ordered oldest to newest.
created = [
    "2026-10-11 10:00:00",
    "2026-10-11 10:10:00",
    "2026-10-11 10:20:00",
    "2026-10-11 10:30:00",
    "2026-10-11 10:40:00",
]

# Replace/reorder these values to match those timestamps.
accuracy = [49.5349, 83.97876, 30.54493, 52.04183, 44.67608]

times = [
    datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
    for t in created
]

best_index = accuracy.index(max(accuracy))

plt.figure(figsize=(10, 5.5))
plt.plot(times, accuracy, marker="o", linewidth=2,
         label="Development character accuracy")

plt.scatter(
    times[best_index], accuracy[best_index],
    marker="*", s=180, label="Best configuration"
)

plt.annotate(
    f"Best: {accuracy[best_index]:.2f}%\n"
    f"Experiment {best_index + 1}",
    (times[best_index], accuracy[best_index]),
    xytext=(12, -40),
    textcoords="offset points",
    arrowprops={"arrowstyle": "->"}
)

plt.xlabel("Run creation time (Created)")
plt.ylabel("Development character accuracy (%)")
plt.title("Accuracy vs. Experiment Creation Time")
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig("accuracy_vs_created.png", dpi=300, bbox_inches="tight")
plt.savefig("accuracy_vs_created.pdf", bbox_inches="tight")
plt.show()

print("Best accuracy:", round(max(accuracy), 2), "%")
print("Experiments run through the first best result:", best_index + 1)

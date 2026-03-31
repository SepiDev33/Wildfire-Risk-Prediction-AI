import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import confusion_matrix
import os

os.makedirs("../performance", exist_ok=True)

def load_dataset():
    dataset = fetch_ucirepo(id=547)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.astype(str).str.strip().str.lower()
    y = y.replace({"fire": 1, "not fire": 0, "notfire": 0})
    y = pd.to_numeric(y, errors="coerce")
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    X.columns = [str(col).strip().lower().replace(" ", "_") for col in X.columns]
    return X, y

model_results = {
    "Logistic Regression": {"accuracy": 0.9184, "f1": 0.92},
    "Decision Tree":        {"accuracy": 0.9388, "f1": 0.94},
    "Random Forest":        {"accuracy": 0.9388, "f1": 0.94},
}

names = list(model_results.keys())
accuracies = [v["accuracy"] * 100 for v in model_results.values()]
f1_scores  = [v["f1"] * 100  for v in model_results.values()]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy",  color="#378ADD")
bars2 = ax.bar(x + width/2, f1_scores,  width, label="F1-Score",  color="#1D9E75")

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Comparison — Wildfire Risk Prediction", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11)
ax.set_ylim(85, 100)
ax.legend(fontsize=11)
ax.bar_label(bars1, fmt="%.2f%%", padding=3, fontsize=10)
ax.bar_label(bars2, fmt="%.2f%%", padding=3, fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()
print("Saved: performance/model_comparison.png")

model = joblib.load("wildfire_best_model.joblib")
X, y = load_dataset()
predictions = model.predict(X)
cm = confusion_matrix(y, predictions)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Fire", "Fire"],
            yticklabels=["No Fire", "Fire"],
            linewidths=0.5, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Confusion Matrix — Best Model (Decision Tree)", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved: performance/confusion_matrix.png")

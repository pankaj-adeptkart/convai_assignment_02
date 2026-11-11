from pathlib import Path
import numpy as np
import pandas as pd

path = Path("data/task4_claim_evaluation.csv")
df = pd.read_csv(path)

required = [
    "model_confidence",
    "evaluator_correct",
    "source_supports_claim",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["model_confidence"] = pd.to_numeric(
    df["model_confidence"], errors="coerce"
)
df["evaluator_correct"] = pd.to_numeric(
    df["evaluator_correct"], errors="coerce"
)

df = df.dropna(
    subset=["model_confidence", "evaluator_correct"]
).copy()

if df.empty:
    raise ValueError("No independently evaluated claims are available.")

if not df["model_confidence"].between(0, 1).all():
    raise ValueError("Confidence values must be between 0 and 1.")

if not df["evaluator_correct"].isin([0, 1]).all():
    raise ValueError("Correctness labels must be 0 or 1.")

bins = np.linspace(0.0, 1.0, 11)
df["confidence_bin"] = pd.cut(
    df["model_confidence"],
    bins=bins,
    include_lowest=True
)

ece = 0.0
rows = []

for interval, group in df.groupby(
    "confidence_bin", observed=True
):
    if group.empty:
        continue

    mean_confidence = group["model_confidence"].mean()
    mean_accuracy = group["evaluator_correct"].mean()
    weight = len(group) / len(df)

    ece += weight * abs(mean_accuracy - mean_confidence)

    rows.append({
        "bin": str(interval),
        "count": len(group),
        "mean_confidence": mean_confidence,
        "empirical_accuracy": mean_accuracy,
    })

summary = pd.DataFrame(rows)
summary.to_csv(
    "data/task4_ece_bin_summary.csv",
    index=False
)

print("Evaluated claims:", len(df))
print(f"Human-labelled ECE: {ece:.4f}")
print("Saved data/task4_ece_bin_summary.csv")
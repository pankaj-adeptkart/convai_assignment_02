from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "human_evaluation_template.csv"
OUTPUT_FILE = DATA_DIR / "human_evaluation_summary.csv"

df = pd.read_csv(INPUT_FILE)

score_columns = [
    "factual_correctness",
    "citation_support",
    "completeness",
    "context_retention",
]

for column in score_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Evaluation records:", len(df))

summary = (
    df.groupby("task_id")[score_columns]
    .agg(["mean", "count"])
)

summary.to_csv(OUTPUT_FILE)

print("\nHuman evaluation summary:")
print(summary)

print(f"\nSaved: {OUTPUT_FILE}")
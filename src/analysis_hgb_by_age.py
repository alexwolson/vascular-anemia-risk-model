"""
Analyse pre-operative haemoglobin (HEMO) distribution by age band.

Outputs
-------
output/task4_hgb_by_age/hgb_by_age_stats.csv   -- descriptive statistics per age band
output/task4_hgb_by_age/hgb_by_age_extreme.csv  -- counts of extreme low HEMO values
output/task4_hgb_by_age/hgb_by_age_violin.png   -- violin plot (300 DPI)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "merged_vqi_2012_2020.parquet"
OUT_DIR = ROOT / "output" / "task4_hgb_by_age"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Age-band definitions
# ---------------------------------------------------------------------------
AGE_BINS = [0, 40, 50, 60, 70, 80, 91]  # 91 so that AGE==90 is included
AGE_LABELS = ["Under 40", "40-49", "50-59", "60-69", "70-79", "80+"]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print(f"Loading {DATA_PATH} ...")
df = pd.read_parquet(DATA_PATH)
print(f"  Total rows: {len(df):,}")
print(f"  AGE range : {df['AGE'].min()} - {df['AGE'].max()}")
print(f"  HEMO range: {df['HEMO'].min()} - {df['HEMO'].max()}")
print(f"  HEMO null : {df['HEMO'].isna().sum():,}")

# Drop rows where HEMO is missing (cannot compute statistics)
df = df.dropna(subset=["HEMO"]).copy()
print(f"  Rows after dropping HEMO nulls: {len(df):,}")

# ---------------------------------------------------------------------------
# 2. Create age bands
# ---------------------------------------------------------------------------
df["age_band"] = pd.cut(
    df["AGE"],
    bins=AGE_BINS,
    labels=AGE_LABELS,
    right=False,          # [lower, upper)
    include_lowest=True,
)

print(f"\nAge-band distribution:")
print(df["age_band"].value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# 3. Summary statistics table  (Task items 4)
# ---------------------------------------------------------------------------
def compute_stats(group: pd.Series) -> pd.Series:
    """Return descriptive stats for a HEMO series."""
    return pd.Series(
        {
            "n": int(group.count()),
            "min": group.min(),
            "max": group.max(),
            "mean": group.mean(),
            "SD": group.std(),
            "median": group.median(),
            "Q1": group.quantile(0.25),
            "Q3": group.quantile(0.75),
            "IQR": group.quantile(0.75) - group.quantile(0.25),
        }
    )


stats = (
    df.groupby("age_band", observed=False)["HEMO"]
    .apply(compute_stats)
    .unstack()
)

# Enforce column order
stats = stats[["n", "min", "max", "mean", "SD", "median", "Q1", "Q3", "IQR"]]
stats["n"] = stats["n"].astype(int)

stats_path = OUT_DIR / "hgb_by_age_stats.csv"
stats.to_csv(stats_path)
print(f"\nSummary statistics saved to {stats_path}")
print(stats.to_string(float_format="%.2f"))

# ---------------------------------------------------------------------------
# 4. Extreme low-HEMO counts  (Task items 5-6)
# ---------------------------------------------------------------------------
extreme_rows = []
for label in AGE_LABELS:
    band = df.loc[df["age_band"] == label, "HEMO"]
    n_total = len(band)
    n_lt7 = int((band < 7).sum())
    n_lt5 = int((band < 5).sum())
    extreme_rows.append(
        {
            "age_band": label,
            "n_total": n_total,
            "n_HEMO_lt7": n_lt7,
            "pct_HEMO_lt7": 100.0 * n_lt7 / n_total if n_total else np.nan,
            "n_HEMO_lt5": n_lt5,
            "pct_HEMO_lt5": 100.0 * n_lt5 / n_total if n_total else np.nan,
        }
    )

extreme = pd.DataFrame(extreme_rows).set_index("age_band")
extreme_path = OUT_DIR / "hgb_by_age_extreme.csv"
extreme.to_csv(extreme_path)
print(f"\nExtreme-low HEMO counts saved to {extreme_path}")
print(extreme.to_string(float_format="%.2f"))

# ---------------------------------------------------------------------------
# 5. Violin plot  (Task item 9)
# ---------------------------------------------------------------------------
# Filter HEMO to a clinically plausible range for visualisation.
# Values outside 2-20 g/dL are almost certainly sentinel/data-entry artefacts.
HEMO_PLOT_MIN = 2.0
HEMO_PLOT_MAX = 20.0
plot_df = df.loc[df["HEMO"].between(HEMO_PLOT_MIN, HEMO_PLOT_MAX)].copy()
n_excluded = len(df) - len(plot_df)
print(
    f"\nViolin plot: filtered HEMO to [{HEMO_PLOT_MIN}, {HEMO_PLOT_MAX}] g/dL  "
    f"({n_excluded:,} rows excluded as likely sentinel / outlier values)"
)

# Publication-quality style
sns.set_theme(style="whitegrid", font_scale=1.1)
fig, ax = plt.subplots(figsize=(10, 6))

sns.violinplot(
    data=plot_df,
    x="age_band",
    y="HEMO",
    hue="age_band",
    order=AGE_LABELS,
    hue_order=AGE_LABELS,
    inner="quartile",
    linewidth=1.0,
    palette="muted",
    legend=False,
    ax=ax,
)

ax.set_xlabel("Age Group", fontsize=13, labelpad=10)
ax.set_ylabel("Pre-operative Haemoglobin (g/dL)", fontsize=13, labelpad=10)
ax.set_title("Distribution of Pre-operative Haemoglobin by Age Group", fontsize=14, pad=12)
ax.tick_params(axis="both", labelsize=11)

# Add a note about filtering
fig.text(
    0.5,
    -0.02,
    f"Note: HEMO values outside [{HEMO_PLOT_MIN}, {HEMO_PLOT_MAX}] g/dL excluded "
    f"from plot ({n_excluded:,} observations; likely sentinel / data-entry artefacts).",
    ha="center",
    fontsize=9,
    style="italic",
    color="grey",
)

fig.tight_layout()
violin_path = OUT_DIR / "hgb_by_age_violin.png"
fig.savefig(violin_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Violin plot saved to {violin_path}")

print("\nDone.")

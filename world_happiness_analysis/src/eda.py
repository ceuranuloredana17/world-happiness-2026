"""
World Happiness Report 2026 — Exploratory Data Analysis

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "world_happiness_2026.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = "viridis"
sns.set_theme(style="whitegrid", palette=PALETTE)

FEATURES = ["gdp_per_capita", "social_support", "healthy_life_expectancy",
            "freedom", "generosity", "corruption"]


# ── Load & validate ───────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✔ Loaded {len(df)} rows, {df.shape[1]} columns")
    print(df.dtypes, "\n")
    print("Missing values:\n", df.isnull().sum(), "\n")
    return df


# ── Descriptive stats ─────────────────────────────────────────────────────────
def descriptive_stats(df: pd.DataFrame) -> None:
    print("=" * 50)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 50)
    print(df.describe().round(3).to_string())
    print()

    print("Top 10 Happiest Countries:")
    print(df.nlargest(10, "score")[["rank", "country", "region", "score"]].to_string(index=False))
    print()

    print("Bottom 10 Countries:")
    print(df.nsmallest(10, "score")[["rank", "country", "region", "score"]].to_string(index=False))
    print()

    print("Average score by region:")
    region_avg = df.groupby("region")["score"].mean().sort_values(ascending=False).round(3)
    print(region_avg.to_string())
    print()


# ── Plot 1: Top & Bottom 10 ───────────────────────────────────────────────────
def plot_top_bottom(df: pd.DataFrame) -> None:
    top = df.nlargest(10, "score")[["country", "score"]]
    bottom = df.nsmallest(10, "score")[["country", "score"]]
    combined = pd.concat([top, bottom])
    colors = ["#2ecc71"] * 10 + ["#e74c3c"] * 10

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(combined["country"], combined["score"], color=colors)
    ax.axvline(df["score"].mean(), color="navy", linestyle="--", linewidth=1.2, label="Global Mean")
    ax.set_xlabel("Happiness Score", fontsize=12)
    ax.set_title("Top 10 & Bottom 10 Countries — Happiness Score 2026", fontsize=14, fontweight="bold")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_bottom_countries.png"), dpi=150)
    plt.close()
    print("✔ Saved: top_bottom_countries.png")


# ── Plot 2: Correlation heatmap ───────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr_cols = ["score"] + FEATURES
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    # upper triangle mask
    import numpy as np
    mask_np = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask_np, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Correlation Matrix — Happiness Features 2026", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print("✔ Saved: correlation_heatmap.png")


# ── Plot 3: GDP vs Score scatter (colored by region) ─────────────────────────
def plot_gdp_vs_score(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    regions = df["region"].unique()
    cmap = plt.cm.get_cmap("tab10", len(regions))

    for i, region in enumerate(regions):
        sub = df[df["region"] == region]
        ax.scatter(sub["gdp_per_capita"], sub["score"], label=region,
                   color=cmap(i), alpha=0.8, edgecolors="white", linewidth=0.4, s=70)

    # Annotate top 5 + bottom 5
    for _, row in pd.concat([df.nlargest(5, "score"), df.nsmallest(5, "score")]).iterrows():
        ax.annotate(row["country"], (row["gdp_per_capita"], row["score"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("GDP per Capita (log scale)", fontsize=12)
    ax.set_ylabel("Happiness Score", fontsize=12)
    ax.set_title("GDP per Capita vs Happiness Score — 2026", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="Region")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gdp_vs_score.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ Saved: gdp_vs_score.png")


# ── Plot 4: Feature contribution stacked bar (top 20) ────────────────────────
def plot_feature_contributions(df: pd.DataFrame) -> None:
    top20 = df.nlargest(20, "score").set_index("country")
    fig, ax = plt.subplots(figsize=(12, 7))
    top20[FEATURES].plot(kind="bar", stacked=True, ax=ax,
                         colormap="Spectral", edgecolor="none")
    ax.set_xlabel("Country", fontsize=11)
    ax.set_ylabel("Contribution to Score", fontsize=11)
    ax.set_title("Feature Contributions — Top 20 Countries 2026", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_contributions_top20.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ Saved: feature_contributions_top20.png")


# ── Plot 5: Average score by region (horizontal bar) ─────────────────────────
def plot_region_scores(df: pd.DataFrame) -> None:
    region_avg = df.groupby("region")["score"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("coolwarm", len(region_avg))
    ax.barh(region_avg.index, region_avg.values, color=colors)
    ax.axvline(df["score"].mean(), color="black", linestyle="--", linewidth=1, label="Global Mean")
    ax.set_xlabel("Average Happiness Score", fontsize=12)
    ax.set_title("Average Happiness Score by Region — 2026", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "region_scores.png"), dpi=150)
    plt.close()
    print("✔ Saved: region_scores.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    descriptive_stats(df)
    plot_top_bottom(df)
    plot_correlation_heatmap(df)
    plot_gdp_vs_score(df)
    plot_feature_contributions(df)
    plot_region_scores(df)
    print("\n✅ EDA complete. All charts saved to /outputs/")

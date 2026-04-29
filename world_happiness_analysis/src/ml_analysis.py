"""
World Happiness Report 2026 — Machine Learning Analysis
  • Linear Regression  →  predict happiness score
  • Random Forest      →  feature importance
  • KMeans Clustering  →  group countries by happiness profile
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "..", "data", "world_happiness_2026.csv")
OUTPUT_DIR = os.path.join(BASE, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

FEATURES = ["gdp_per_capita", "social_support", "healthy_life_expectancy",
            "freedom", "generosity", "corruption"]


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
def run_regression(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  LINEAR REGRESSION — Predict Happiness Score")
    print("=" * 55)

    X = df[FEATURES]
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    print(f"  R²  (test set)      : {r2:.4f}")
    print(f"  R²  (5-fold CV avg) : {cv_r2:.4f}")
    print(f"  MSE (test set)      : {mse:.4f}")
    print(f"  RMSE                : {np.sqrt(mse):.4f}")

    # Coefficients
    coef_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=False)
    print("\n  Coefficients:")
    print(coef_df.to_string(index=False))

    # Plot: Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(y_test, y_pred, color="#3498db", edgecolors="white",
               linewidth=0.5, alpha=0.85, s=80)
    lims = [min(y_test.min(), y_pred.min()) - 0.2,
            max(y_test.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Score", fontsize=12)
    ax.set_ylabel("Predicted Score", fontsize=12)
    ax.set_title(f"Actual vs Predicted\nR² = {r2:.3f}", fontsize=13, fontweight="bold")
    ax.legend()

    # Plot: Coefficients
    ax2 = axes[1]
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["Coefficient"]]
    ax2.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Coefficient Value", fontsize=12)
    ax2.set_title("Linear Regression Coefficients", fontsize=13, fontweight="bold")

    plt.suptitle("Linear Regression — World Happiness 2026", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ml_regression.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✔ Saved: ml_regression.png")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RANDOM FOREST — Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
def run_random_forest(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  RANDOM FOREST — Feature Importance")
    print("=" * 55)

    X = df[FEATURES]
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  R²  (test set) : {r2:.4f}")
    print(f"  RMSE           : {rmse:.4f}")

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=True)

    print("\n  Feature Importances:")
    print(importance_df.sort_values("Importance", ascending=False).to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color=cmap)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Random Forest — Feature Importance\nR² = {r2:.3f}  |  RMSE = {rmse:.3f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ml_feature_importance.png"), dpi=150)
    plt.close()
    print("  ✔ Saved: ml_feature_importance.png")

    return rf


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KMEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
def run_clustering(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  KMEANS CLUSTERING — Country Happiness Profiles")
    print("=" * 55)

    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    inertias = []
    k_range = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Fit best k = 4
    K = 4
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_labels = {0: "Struggling", 1: "Developing", 2: "Thriving", 3: "Flourishing"}
    # Auto-assign labels by cluster mean score
    cluster_means = df.groupby("cluster")["score"].mean().sort_values()
    label_map = {cluster_id: list(cluster_labels.values())[i]
                 for i, cluster_id in enumerate(cluster_means.index)}
    df["cluster_label"] = df["cluster"].map(label_map)

    print("\n  Cluster Summary:")
    summary = df.groupby("cluster_label").agg(
        Countries=("country", "count"),
        Avg_Score=("score", "mean"),
        Min_Score=("score", "min"),
        Max_Score=("score", "max")
    ).round(3)
    print(summary.to_string())

    print("\n  Sample countries per cluster:")
    for label in ["Flourishing", "Thriving", "Developing", "Struggling"]:
        sample = df[df["cluster_label"] == label]["country"].head(4).tolist()
        print(f"    {label:12s}: {', '.join(sample)}")

    # ── Plot 1: Elbow curve ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inertias, marker="o", color="#3498db", linewidth=2)
    axes[0].axvline(K, color="red", linestyle="--", linewidth=1.2, label=f"Chosen k={K}")
    axes[0].set_xlabel("Number of Clusters (k)", fontsize=12)
    axes[0].set_ylabel("Inertia", fontsize=12)
    axes[0].set_title("Elbow Method — Optimal k", fontsize=13, fontweight="bold")
    axes[0].legend()

    # ── Plot 2: GDP vs Score, colored by cluster ──────────────────────────────
    cluster_colors = {"Flourishing": "#2ecc71", "Thriving": "#3498db",
                      "Developing": "#f39c12", "Struggling": "#e74c3c"}
    ax2 = axes[1]
    for label, grp in df.groupby("cluster_label"):
        ax2.scatter(grp["gdp_per_capita"], grp["score"],
                    color=cluster_colors[label], label=label,
                    alpha=0.85, edgecolors="white", linewidth=0.4, s=75)

    ax2.set_xlabel("GDP per Capita", fontsize=12)
    ax2.set_ylabel("Happiness Score", fontsize=12)
    ax2.set_title("KMeans Clusters — GDP vs Score", fontsize=13, fontweight="bold")
    ax2.legend(title="Cluster")

    plt.suptitle("KMeans Clustering — World Happiness 2026", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ml_clustering.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✔ Saved: ml_clustering.png")

    # ── Plot 3: Cluster radar / mean features heatmap ─────────────────────────
    cluster_profile = df.groupby("cluster_label")[FEATURES].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cluster_profile, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Cluster Feature Profiles — Mean Values", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Cluster", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ml_cluster_profiles.png"), dpi=150)
    plt.close()
    print("  ✔ Saved: ml_cluster_profiles.png")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    run_regression(df)
    run_random_forest(df)
    run_clustering(df)
    print("\n✅ ML analysis complete. Charts saved to /outputs/")

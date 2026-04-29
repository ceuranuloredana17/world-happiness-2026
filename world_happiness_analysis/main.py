"""
World Happiness Report 2026 — Run Everything
Usage:
    python main.py          # EDA + ML
    python main.py --dash   # EDA + ML + interactive dashboard
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.eda import load_data, descriptive_stats, plot_top_bottom, \
    plot_correlation_heatmap, plot_gdp_vs_score, \
    plot_feature_contributions, plot_region_scores
from src.ml_analysis import run_regression, run_random_forest, run_clustering


def main():
    print("━" * 55)
    print("  WORLD HAPPINESS REPORT 2026 — Full Analysis")
    print("━" * 55)

    df = load_data()

    print("\n📊 Phase 1 — Exploratory Data Analysis")
    descriptive_stats(df)
    plot_top_bottom(df)
    plot_correlation_heatmap(df)
    plot_gdp_vs_score(df)
    plot_feature_contributions(df)
    plot_region_scores(df)

    print("\n🤖 Phase 2 — Machine Learning")
    run_regression(df)
    run_random_forest(df)
    df_clustered = run_clustering(df)

    print("\n✅ All done! Charts saved to /outputs/")

    if "--dash" in sys.argv:
        print("\n🚀 Starting interactive dashboard...")
        from src.dashboard import load_data as load_dash, build_app
        df2 = load_dash()
        app = build_app(df2)
        app.run(debug=True)


if __name__ == "__main__":
    main()

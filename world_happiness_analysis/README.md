# 🌍 World Happiness Report 2026 — Data Analysis & ML

A full-stack **data science project** covering exploratory analysis, machine learning,
and an interactive dashboard — built on the World Happiness Report 2026 dataset (147 countries).

---

## 🚀 Quick Start

```bash
git clone https://github.com/ceuranuloredana17/world-happiness-2026.git
cd world-happiness-2026

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run EDA + ML (saves charts to /outputs/)
python main.py

# Run EDA + ML + interactive dashboard
python main.py --dash
```

---

## 📁 Project Structure

```
world_happiness_analysis/
├── data/
│   └── world_happiness_2026.csv     # 147 countries, 10 features
├── src/
│   ├── eda.py                       # Phase 1 — Exploratory Data Analysis
│   ├── ml_analysis.py               # Phase 2 — Machine Learning
│   └── dashboard.py                 # Phase 2 — Interactive Plotly Dashboard
├── outputs/                         # Auto-generated charts
├── main.py                          # Entry point — run everything
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Phase 1 — Exploratory Data Analysis

| Chart | Description |
|---|---|
| `top_bottom_countries.png` | Top 10 & Bottom 10 with global mean reference |
| `correlation_heatmap.png` | Feature correlation matrix |
| `gdp_vs_score.png` | GDP vs Score scatter, colored by region |
| `feature_contributions_top20.png` | Stacked bar — factor breakdown for top 20 |
| `region_scores.png` | Average happiness by world region |

---

## 🤖 Phase 2 — Machine Learning

### Linear Regression
- Predicts happiness score from 6 socioeconomic features
- Metrics: R², RMSE, 5-fold cross-validation
- Outputs: Actual vs Predicted + coefficient chart

### Random Forest
- 200 estimators, compares R² vs linear baseline
- Feature importance ranking

### KMeans Clustering (k=4)
- Groups countries into: **Flourishing · Thriving · Developing · Struggling**
- Elbow method for optimal k selection

| Chart | Description |
|---|---|
| `ml_regression.png` | Actual vs Predicted + coefficients |
| `ml_feature_importance.png` | Random Forest feature importance |
| `ml_clustering.png` | Elbow curve + cluster scatter |
| `ml_cluster_profiles.png` | Mean feature values per cluster |

---

## 📈 Interactive Dashboard (Plotly Dash)

Run `python main.py --dash` → open **http://127.0.0.1:8050**

- 🗺 Choropleth world map — filterable by region
- 📊 Top 10 bar chart
- 🔵 Scatter plot — any feature vs score, OLS trendline
- 🕸 Radar chart — feature profiles for top 5 regions

---

## 🔑 Key Findings

- **Western Europe** leads globally; Finland, Iceland, Denmark are the top 3
- **GDP per capita** and **social support** are the strongest predictors (R² ≈ 0.97)
- **Random Forest** outperforms linear regression on unseen data
- **KMeans** reveals 4 distinct profiles aligned with development level
- **Afghanistan** ranks last (1.45); **Sub-Saharan Africa** is the lowest-scoring region

---

## 🛠 Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `plotly` · `dash` · `scikit-learn`

---



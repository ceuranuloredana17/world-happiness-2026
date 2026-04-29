"""
World Happiness Report 2026 — Interactive Dashboard (Plotly)
Run this script, then open http://127.0.0.1:8050 in your browser.

Install extra deps:
    pip install plotly dash
"""

import pandas as pd
import numpy as np
import os

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dash import Dash, dcc, html, Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("⚠  Dash/Plotly not installed. Run:  pip install plotly dash")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "..", "data", "world_happiness_2026.csv")

FEATURES = ["gdp_per_capita", "social_support", "healthy_life_expectancy",
            "freedom", "generosity", "corruption"]

FEATURE_LABELS = {
    "gdp_per_capita": "GDP per Capita",
    "social_support": "Social Support",
    "healthy_life_expectancy": "Healthy Life Expectancy",
    "freedom": "Freedom",
    "generosity": "Generosity",
    "corruption": "Low Corruption"
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def build_app(df: pd.DataFrame) -> "Dash":
    app = Dash(__name__, title="World Happiness 2026")

    regions = sorted(df["region"].unique())
    region_options = [{"label": "All Regions", "value": "ALL"}] + \
                     [{"label": r, "value": r} for r in regions]

    # ── Layout ────────────────────────────────────────────────────────────────
    app.layout = html.Div([

        # Header
        html.Div([
            html.H1("🌍 World Happiness Report 2026",
                    style={"margin": 0, "color": "white", "fontFamily": "Segoe UI, sans-serif"}),
            html.P("Interactive analysis of 147 countries across 10 regions",
                   style={"color": "#ccc", "margin": "4px 0 0", "fontFamily": "Segoe UI, sans-serif"})
        ], style={"background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
                  "padding": "24px 32px", "marginBottom": "24px"}),

        # Controls
        html.Div([
            html.Div([
                html.Label("Filter by Region", style={"fontWeight": "bold", "marginBottom": "6px", "display": "block"}),
                dcc.Dropdown(
                    id="region-filter",
                    options=region_options,
                    value="ALL",
                    clearable=False,
                    style={"width": "300px"}
                )
            ], style={"marginRight": "40px"}),

            html.Div([
                html.Label("X Axis (Scatter)", style={"fontWeight": "bold", "marginBottom": "6px", "display": "block"}),
                dcc.Dropdown(
                    id="x-axis",
                    options=[{"label": FEATURE_LABELS[f], "value": f} for f in FEATURES],
                    value="gdp_per_capita",
                    clearable=False,
                    style={"width": "260px"}
                )
            ])
        ], style={"display": "flex", "alignItems": "flex-end", "padding": "0 32px 20px"}),

        # Row 1: Map + Bar
        html.Div([
            html.Div([dcc.Graph(id="choropleth-map")],
                     style={"flex": "1.6", "marginRight": "16px"}),
            html.Div([dcc.Graph(id="top-bar")],
                     style={"flex": "1"})
        ], style={"display": "flex", "padding": "0 32px 16px"}),

        # Row 2: Scatter + Radar
        html.Div([
            html.Div([dcc.Graph(id="scatter-plot")],
                     style={"flex": "1.3", "marginRight": "16px"}),
            html.Div([dcc.Graph(id="region-radar")],
                     style={"flex": "1"})
        ], style={"display": "flex", "padding": "0 32px 32px"}),

    ], style={"fontFamily": "Segoe UI, sans-serif", "backgroundColor": "#f8f9fa", "minHeight": "100vh"})

    # ── Callbacks ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("choropleth-map", "figure"),
        Output("top-bar", "figure"),
        Output("scatter-plot", "figure"),
        Output("region-radar", "figure"),
        Input("region-filter", "value"),
        Input("x-axis", "value"),
    )
    def update_all(region_filter, x_axis):
        filtered = df if region_filter == "ALL" else df[df["region"] == region_filter]

        # 1. Choropleth map
        map_fig = px.choropleth(
            filtered,
            locations="country",
            locationmode="country names",
            color="score",
            hover_name="country",
            hover_data={"rank": True, "score": ":.2f", "region": True},
            color_continuous_scale="Viridis",
            title="Happiness Score by Country",
        )
        map_fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_colorbar=dict(title="Score"),
            plot_bgcolor="white", paper_bgcolor="white"
        )

        # 2. Top/Bottom 10 bar
        top = filtered.nlargest(10, "score")[["country", "score", "region"]]
        bar_fig = px.bar(
            top.sort_values("score"),
            x="score", y="country", orientation="h",
            color="score", color_continuous_scale="Viridis",
            title="Top 10 Happiest Countries",
            labels={"score": "Score", "country": ""},
        )
        bar_fig.update_layout(showlegend=False, plot_bgcolor="white",
                               paper_bgcolor="white", margin=dict(l=0, r=10, t=40, b=0))

        # 3. Scatter
        scatter_fig = px.scatter(
            filtered,
            x=x_axis, y="score",
            color="region",
            hover_name="country",
            hover_data={"rank": True, "score": ":.2f"},
            title=f"{FEATURE_LABELS[x_axis]} vs Happiness Score",
            trendline="ols",
            labels={x_axis: FEATURE_LABELS[x_axis], "score": "Happiness Score"},
            size_max=14,
            opacity=0.85
        )
        scatter_fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                   margin=dict(t=45, b=10))

        # 4. Radar by region
        region_means = df.groupby("region")[FEATURES].mean()
        top_regions = df.groupby("region")["score"].mean().nlargest(5).index
        radar_fig = go.Figure()
        labels = [FEATURE_LABELS[f] for f in FEATURES]
        for reg in top_regions:
            vals = region_means.loc[reg, FEATURES].tolist()
            vals_closed = vals + [vals[0]]
            radar_fig.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=labels + [labels[0]],
                fill="toself",
                name=reg,
                opacity=0.6
            ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
            title="Feature Profiles — Top 5 Regions",
            showlegend=True,
            paper_bgcolor="white",
            margin=dict(t=50, b=20)
        )

        return map_fig, bar_fig, scatter_fig, radar_fig

    return app


if __name__ == "__main__":
    if not DASH_AVAILABLE:
        print("Install with:  pip install plotly dash")
    else:
        df = load_data()
        app = build_app(df)
        print("\n✅ Dashboard running at http://127.0.0.1:8050")
        print("   Press Ctrl+C to stop.\n")
        app.run(debug=True)

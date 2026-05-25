"""Page 1 — Player Explorer: FBref data loading, PCA scatter, player radar chart."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc

from src.data import COMPETITION_REGISTRY
from src.pipeline import load_tournament_data, load_quality_scores

_RADAR_FEATURES = [
    "proportion_dribble_complete",
    "proportion_duel_won",
    "proportion_pass_cross",
    "proportion_pass_shot_assist",
    "proportion_goal",
    "proportion_50_50_won",
    "proportion_clearance_aerial_won",
    "proportion_interception_won",
    "proportion_foul_dangerous_play",
    "mean_shot_statsbomb_xg",
]

_RADAR_LABELS = [f.replace("proportion_", "").replace("mean_", "").replace("_", " ").title()
                 for f in _RADAR_FEATURES]

_COMP_OPTIONS = [
    {"label": k.replace("_", " ").title(), "value": k}
    for k in COMPETITION_REGISTRY
]

_POS_COLORS = {
    "center_back":      "#1f77b4",
    "right_back":       "#aec7e8",
    "left_back":        "#c5b0d5",
    "center_midfield":  "#2ca02c",
    "center_attacking_midfield": "#98df8a",
    "center_defensive_midfield": "#d62728",
    "right_wing":       "#ff7f0e",
    "left_wing":        "#ffbb78",
    "center_forward":   "#9467bd",
    "unknown":          "#bcbcbc",
}


def layout() -> html.Div:
    return html.Div([
        dcc.Store(id="pe-store"),

        html.H4("Player Explorer", className="mb-3"),

        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="pe-competition",
                    options=_COMP_OPTIONS,
                    value="copa_america_2024",
                    clearable=False,
                ),
                width=5,
            ),
            dbc.Col(
                dbc.Button("Load Data", id="pe-load-btn", color="primary", n_clicks=0),
                width="auto",
            ),
            dbc.Col(html.Div(id="pe-status", className="text-muted pt-2"), width=True),
        ], className="mb-3 align-items-center"),

        dbc.Row([
            dbc.Col(
                dcc.Loading(
                    dcc.Graph(id="pe-scatter", style={"height": "500px"},
                              config={"displayModeBar": False}),
                    type="circle",
                ),
                width=8,
            ),
            dbc.Col([
                dcc.Dropdown(
                    id="pe-player-search",
                    placeholder="Search player…",
                    className="mb-2",
                    options=[],
                ),
                dcc.Loading(
                    dcc.Graph(id="pe-radar", style={"height": "420px"},
                              config={"displayModeBar": False}),
                    type="dot",
                ),
            ], width=4),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.H6("Quality Scores", className="mb-2"),
                html.Div(id="pe-quality-table"),
            ]),
        ]),
    ])


def register_callbacks(app) -> None:

    @app.callback(
        Output("pe-store", "data"),
        Output("pe-status", "children"),
        Input("pe-load-btn", "n_clicks"),
        State("pe-competition", "value"),
        prevent_initial_call=True,
    )
    def load_data(n_clicks, competition):
        model_df, _ = load_tournament_data(competition)
        try:
            quality_df = load_quality_scores(competition, fbref_stats=None)
            q_json = quality_df.reset_index().to_json(orient="records")
        except Exception:
            q_json = None
        store = {
            "model_df": model_df.reset_index().to_json(orient="records"),
            "quality_json": q_json,
            "competition": competition,
        }
        return store, f"✓ Loaded {len(model_df)} players from {competition.replace('_', ' ').title()}"

    @app.callback(
        Output("pe-scatter", "figure"),
        Output("pe-player-search", "options"),
        Input("pe-store", "data"),
        prevent_initial_call=True,
    )
    def build_scatter(store):
        if not store:
            return go.Figure(), []

        df = pd.read_json(store["model_df"], orient="records")
        if "player" in df.columns:
            df = df.set_index("player")

        numeric = df.select_dtypes(include="number").fillna(0)
        if numeric.shape[0] < 2:
            return go.Figure(), []

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(numeric)

        # Determine position group from one-hot columns
        pos_cols = [c for c in df.columns if c.startswith("mode_position_")]

        def _pos_label(row):
            for c in pos_cols:
                if row[c] == 1:
                    return c.replace("mode_position_", "")
            return "unknown"

        labels = df[pos_cols].apply(_pos_label, axis=1) if pos_cols else pd.Series(
            "unknown", index=df.index
        )

        fig = go.Figure()
        seen_groups = labels.unique()
        for grp in list(_POS_COLORS) + [g for g in seen_groups if g not in _POS_COLORS]:
            mask = labels == grp
            if not mask.any():
                continue
            color = _POS_COLORS.get(grp, "#bcbcbc")
            fig.add_trace(go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=grp.replace("_", " ").title(),
                text=df.index[mask].tolist(),
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=8, color=color, opacity=0.75,
                            line=dict(width=0.5, color="white")),
            ))

        var_ratio = pca.explained_variance_ratio_
        fig.update_layout(
            title=f"Personality Space — PCA  ({var_ratio[0]:.0%} + {var_ratio[1]:.0%} variance)",
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(t=80),
        )

        player_options = [{"label": p, "value": p} for p in df.index.tolist()]
        return fig, player_options

    @app.callback(
        Output("pe-radar", "figure"),
        Input("pe-player-search", "value"),
        State("pe-store", "data"),
        prevent_initial_call=True,
    )
    def show_radar(player, store):
        if not player or not store:
            return go.Figure()

        df = pd.read_json(store["model_df"], orient="records")
        if "player" in df.columns:
            df = df.set_index("player")

        if player not in df.index:
            return go.Figure()

        available = [f for f in _RADAR_FEATURES if f in df.columns]
        labels = [_RADAR_LABELS[_RADAR_FEATURES.index(f)] for f in available]
        values = df.loc[player, available].fillna(0).tolist()

        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(31,119,180,0.2)",
            line=dict(color="#1f77b4", width=2),
            name=player,
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            template="plotly_white",
            title=dict(text=player, font=dict(size=13)),
            height=420,
            margin=dict(t=60, l=30, r=30, b=30),
        )
        return fig

    @app.callback(
        Output("pe-quality-table", "children"),
        Input("pe-store", "data"),
        prevent_initial_call=True,
    )
    def show_quality_table(store):
        if not store or not store.get("quality_json"):
            return dbc.Alert("Quality scores not available.", color="info", className="mt-2")

        df = pd.read_json(store["quality_json"], orient="records")
        score_cols = [c for c in ["player", "quality_score", "tm_score", "fbref_score",
                                   "sofascore_score", "market_value_eur", "position_group"]
                      if c in df.columns]
        df = df[score_cols].sort_values("quality_score", ascending=False).head(30)
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].round(3)

        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df.columns],
            sort_action="native",
            page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "12px", "padding": "6px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        )

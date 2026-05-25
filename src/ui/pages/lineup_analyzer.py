"""Page 2 — Lineup Matchup Analyzer: personality-based xG prediction + betting signal."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc

from src.data import COMPETITION_REGISTRY
from src.pipeline import betting_signal, predict_with_confidence
from src.ui.components import CAT_OPTIONS, lineup_builder, metric_card


class _MockModel:
    """
    Stand-in when no trained Keras model is loaded.
    xG ≈ weighted sum of category positions (more forwards → higher xG).
    Clearly marked as demo output in the UI.
    """
    _WEIGHTS = {**{i: 0.08 for i in range(1, 8)},
                **{i: 0.14 for i in range(8, 18)},
                **{i: 0.22 for i in range(18, 24)}}

    def predict(self, x, verbose=0):
        team = x[0, 0, :]
        xg = sum(self._WEIGHTS.get(int(round(v)), 0.12) for v in team)
        xg = max(0.3, xg + np.random.normal(0, 0.05))
        return np.array([[xg]])


_MOCK_MODEL = _MockModel()

_COMP_OPTIONS = [
    {"label": k.replace("_", " ").title(), "value": k}
    for k in COMPETITION_REGISTRY
]


def layout() -> html.Div:
    return html.Div([
        dcc.Store(id="la-results-store"),

        html.H4("Lineup Matchup Analyzer", className="mb-1"),
        dbc.Alert(
            "Demo mode — predictions use a rule-based mock model. "
            "Replace _MockModel with your trained Keras model for real xG.",
            color="warning",
            className="mb-3 py-2",
        ),

        # ── Lineup builders ───────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Input(id="la-home-name", placeholder="Home team name", className="mb-2"),
                lineup_builder("home", "Home Lineup (10 outfield)"),
            ], width=5),
            dbc.Col(
                dbc.Button(
                    "Analyze Matchup ▶",
                    id="la-run-btn",
                    color="success",
                    size="lg",
                    className="w-100",
                    n_clicks=0,
                    style={"marginTop": "2rem"},
                ),
                width=2,
                className="d-flex align-items-center justify-content-center",
            ),
            dbc.Col([
                dbc.Input(id="la-away-name", placeholder="Away team name", className="mb-2"),
                lineup_builder("away", "Away Lineup (10 outfield)"),
            ], width=5),
        ], className="mb-4"),

        # ── Prediction results ────────────────────────────────────────────
        dcc.Loading(html.Div(id="la-prediction-output"), type="circle"),

        html.Hr(),

        # ── Odds + betting signal ─────────────────────────────────────────
        html.H6("Betting Signal (optional)", className="mt-3 mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Home odds", html_for="la-odds-home", className="small"),
                dbc.Input(id="la-odds-home", type="number", min=1.01, step=0.01,
                          placeholder="e.g. 2.10"),
            ], width=2),
            dbc.Col([
                dbc.Label("Draw odds", html_for="la-odds-draw", className="small"),
                dbc.Input(id="la-odds-draw", type="number", min=1.01, step=0.01,
                          placeholder="e.g. 3.30"),
            ], width=2),
            dbc.Col([
                dbc.Label("Away odds", html_for="la-odds-away", className="small"),
                dbc.Input(id="la-odds-away", type="number", min=1.01, step=0.01,
                          placeholder="e.g. 3.80"),
            ], width=2),
            dbc.Col([
                dbc.Label("Kelly fraction", html_for="la-kelly", className="small"),
                dcc.Slider(id="la-kelly", min=0.05, max=1.0, step=0.05, value=0.25,
                           marks={0.25: "0.25×", 0.5: "0.5×", 1.0: "Full"},
                           tooltip={"placement": "bottom"}),
            ], width=4),
            dbc.Col(
                dbc.Button("Get Signal", id="la-signal-btn", color="info",
                           outline=True, n_clicks=0),
                width=2,
                className="d-flex align-items-end",
            ),
        ], className="mb-3"),
        html.Div(id="la-signal-output"),
    ])


def register_callbacks(app) -> None:

    @app.callback(
        Output("la-results-store", "data"),
        Output("la-prediction-output", "children"),
        Input("la-run-btn", "n_clicks"),
        State({"type": "home-slot", "index": ALL}, "value"),
        State({"type": "away-slot", "index": ALL}, "value"),
        State("la-home-name", "value"),
        State("la-away-name", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, home_cats, away_cats, home_name, away_name):
        home_cats = [v for v in (home_cats or []) if v is not None]
        away_cats = [v for v in (away_cats or []) if v is not None]

        if len(home_cats) < 10 or len(away_cats) < 10:
            missing = []
            if len(home_cats) < 10:
                missing.append(f"Home needs {10 - len(home_cats)} more")
            if len(away_cats) < 10:
                missing.append(f"Away needs {10 - len(away_cats)} more")
            alert = dbc.Alert(
                f"Please fill all 10 outfield slots. {' | '.join(missing)}.",
                color="danger",
            )
            return no_update, alert

        result = predict_with_confidence(
            model=_MOCK_MODEL,
            home_lineup=home_cats[:10],
            away_lineup=away_cats[:10],
            mae_estimate=0.3,
            confidence_level=0.90,
        )

        home_label = home_name or "Home"
        away_label = away_name or "Away"

        # ── Metric cards ──────────────────────────────────────────────────
        h_xg = result["adjusted_home_xg"]
        a_xg = result["adjusted_away_xg"]
        h_prob = result["home_win_prob"]
        d_prob = result["draw_prob"]
        a_prob = result["away_win_prob"]
        diff = result["expected_goal_diff"]

        cards = dbc.Row([
            dbc.Col(metric_card(f"{home_label} xG", f"{h_xg:.2f}",
                                "success" if h_xg >= a_xg else "secondary"), width=2),
            dbc.Col(metric_card("xG Diff", f"{diff:+.2f}",
                                "success" if diff > 0 else "danger"), width=2),
            dbc.Col(metric_card(f"{away_label} xG", f"{a_xg:.2f}",
                                "success" if a_xg > h_xg else "secondary"), width=2),
            dbc.Col(metric_card(f"{home_label} Win", f"{h_prob:.1%}", "primary"), width=2),
            dbc.Col(metric_card("Draw", f"{d_prob:.1%}", "secondary"), width=2),
            dbc.Col(metric_card(f"{away_label} Win", f"{a_prob:.1%}", "danger"), width=2),
        ], className="mb-3 g-2")

        # ── xG bar chart with confidence interval ─────────────────────────
        ci_fig = go.Figure()
        for label, xg, low, high, color in [
            (home_label, h_xg,
             result["adjusted_home_xg_low"], result["adjusted_home_xg_high"], "#1f77b4"),
            (away_label, a_xg,
             result["adjusted_away_xg_low"], result["adjusted_away_xg_high"], "#d62728"),
        ]:
            ci_fig.add_trace(go.Bar(
                x=[label], y=[xg],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[high - xg],
                    arrayminus=[xg - low],
                    visible=True,
                    color="#666",
                    thickness=2,
                    width=10,
                ),
                name=label,
                marker_color=color,
                text=[f"{xg:.2f}"],
                textposition="outside",
                width=0.4,
            ))
        ci_fig.update_layout(
            title="Expected Goals (90% CI)",
            yaxis_title="xG",
            template="plotly_white",
            height=300,
            showlegend=False,
            margin=dict(t=50, b=30),
        )

        # ── Win probability bar chart ──────────────────────────────────────
        prob_fig = go.Figure(go.Bar(
            x=[home_label, "Draw", away_label],
            y=[h_prob, d_prob, a_prob],
            marker_color=["#1f77b4", "#aec7e8", "#d62728"],
            text=[f"{p:.1%}" for p in [h_prob, d_prob, a_prob]],
            textposition="outside",
        ))
        prob_fig.add_hline(y=1/3, line_dash="dash", line_color="gray",
                           annotation_text="Equal (33%)", annotation_position="right")
        prob_fig.update_layout(
            title="Win / Draw / Loss Probabilities",
            yaxis=dict(title="Probability", tickformat=".0%", range=[0, 1]),
            template="plotly_white",
            height=300,
            showlegend=False,
            margin=dict(t=50, b=30),
        )

        charts = dbc.Row([
            dbc.Col(dcc.Graph(figure=ci_fig, config={"displayModeBar": False}), width=6),
            dbc.Col(dcc.Graph(figure=prob_fig, config={"displayModeBar": False}), width=6),
        ])

        # Persist result for the betting signal callback
        store_data = {
            "home_win_prob": h_prob,
            "draw_prob":     d_prob,
            "away_win_prob": a_prob,
            "adjusted_home_xg": h_xg,
            "adjusted_away_xg": a_xg,
            "expected_goal_diff": diff,
        }

        return store_data, html.Div([cards, charts])

    @app.callback(
        Output("la-signal-output", "children"),
        Input("la-signal-btn", "n_clicks"),
        State("la-results-store", "data"),
        State("la-odds-home", "value"),
        State("la-odds-draw", "value"),
        State("la-odds-away", "value"),
        State("la-kelly", "value"),
        State("la-home-name", "value"),
        State("la-away-name", "value"),
        prevent_initial_call=True,
    )
    def get_signal(n_clicks, result, h_odds, d_odds, a_odds, kelly_frac,
                   home_name, away_name):
        if not result:
            return dbc.Alert("Run analysis first.", color="info")
        if not all(isinstance(v, (int, float)) and v > 1.0
                   for v in [h_odds, d_odds, a_odds]):
            return dbc.Alert("Enter valid decimal odds (> 1.0) for all three outcomes.",
                             color="warning")

        home_label = home_name or "Home"
        away_label = away_name or "Away"

        sig = betting_signal(
            matchup_result=result,
            home_odds=float(h_odds),
            draw_odds=float(d_odds),
            away_odds=float(a_odds),
            kelly_fraction=float(kelly_frac or 0.25),
        )

        margin_pct = f"{sig['bookmaker_margin']:.1%}"

        if not sig["value_bet_found"]:
            return dbc.Alert(
                [
                    html.Strong("No value bet found. "),
                    f"Bookmaker margin: {margin_pct}. ",
                    "Model finds no outcome where edge exceeds the minimum threshold.",
                ],
                color="secondary",
            )

        best = sig["best_signal"]
        outcome_labels = {"home_win": f"{home_label} Win", "draw": "Draw",
                          "away_win": f"{away_label} Win"}
        outcome = outcome_labels.get(best["outcome"], best["outcome"])
        color = "success" if best["edge"] > 0.10 else "warning"

        rows = []
        for s in sig["signals"]:
            out_lbl = outcome_labels.get(s["outcome"], s["outcome"])
            rows.append(
                dbc.ListGroupItem(
                    dbc.Row([
                        dbc.Col(html.Strong(out_lbl), width=3),
                        dbc.Col(f"Edge: {s['edge']:+.1%}", width=3),
                        dbc.Col(f"Kelly: {s['kelly_fraction']:.1%}", width=3),
                        dbc.Col(f"Fair prob: {s['fair_prob']:.1%}", width=3),
                    ]),
                    color=("success" if s["outcome"] == best["outcome"] else None),
                )
            )

        return dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(html.Strong(f"Best bet: {outcome}"), width=6),
                    dbc.Col(f"Bookmaker margin: {margin_pct}", width=3,
                            className="text-muted text-end"),
                    dbc.Col(
                        dbc.Badge(
                            f"Edge {best['edge']:+.1%}",
                            color=color,
                            pill=True,
                        ),
                        width=3,
                        className="text-end",
                    ),
                ])
            ),
            dbc.CardBody(dbc.ListGroup(rows, flush=True)),
        ])

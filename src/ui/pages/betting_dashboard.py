"""Page 3 — Betting Signal Dashboard: historical backtest + Kelly criterion simulation."""

import base64
import io
import json

import numpy as np
import pandas as pd
from dash import Input, Output, State, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc

from src.backtest import (
    backtest_report,
    plot_bankroll_history,
    plot_cumulative_roi,
    plot_edge_distribution,
    run_backtest,
)
from src.data.odds_scraper import FOOTBALL_DATA_URLS, FootballDataScraper
from src.ui.components import metric_card

_COMP_OPTIONS = [
    {"label": k.replace("_", " ").title(), "value": k}
    for k in FOOTBALL_DATA_URLS
]

_SCHEMA_INFO = (
    "Model predictions CSV must have columns: "
    "home_win, draw, away_win  (probabilities 0–1, rows aligned with odds data). "
    "Odds CSV is downloaded automatically from football-data.co.uk."
)

_METRIC_COLS = {
    "roi": "ROI",
    "final_bankroll": "Final Bankroll",
    "max_drawdown": "Max Drawdown (%)",
    "sharpe_ratio": "Sharpe Ratio",
    "win_rate": "Win Rate",
    "n_bets": "# Bets",
    "avg_edge": "Avg Edge",
    "avg_kelly": "Avg Kelly",
}


def _demo_probs(n: int) -> pd.DataFrame:
    """Synthetic model predictions for demo purposes."""
    rng = np.random.default_rng(42)
    probs = rng.dirichlet(alpha=[2, 1.5, 1.5], size=n)
    return pd.DataFrame(probs, columns=["home_win", "draw", "away_win"])


def layout() -> html.Div:
    return html.Div([
        dcc.Store(id="bd-backtest-store"),
        dcc.Store(id="bd-probs-store"),

        html.H4("Betting Signal Dashboard", className="mb-1"),
        dbc.Alert(_SCHEMA_INFO, color="info", className="mb-3 py-2 small"),

        # ── Controls ──────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Label("Competition (historical odds)", className="small fw-bold"),
                dcc.Dropdown(
                    id="bd-competition",
                    options=_COMP_OPTIONS,
                    value="premier_league_2324",
                    clearable=False,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Starting bankroll (€)", className="small fw-bold"),
                dbc.Input(id="bd-bankroll", type="number", value=1000, min=100, step=100),
            ], width=2),
            dbc.Col([
                dbc.Label("Kelly fraction", className="small fw-bold"),
                dcc.Slider(id="bd-kelly", min=0.05, max=1.0, step=0.05, value=0.25,
                           marks={0.25: "¼", 0.5: "½", 1.0: "Full"},
                           tooltip={"placement": "bottom"}),
            ], width=3),
            dbc.Col([
                dbc.Label("Min edge", className="small fw-bold"),
                dcc.Slider(id="bd-min-edge", min=0.01, max=0.20, step=0.01, value=0.05,
                           marks={0.05: "5%", 0.10: "10%", 0.15: "15%"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], width=2),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Model predictions CSV (optional)", className="small fw-bold"),
                dcc.Upload(
                    id="bd-upload",
                    children=html.Div([
                        "Drag & Drop or ",
                        html.A("Select CSV", className="text-primary"),
                    ]),
                    style={
                        "width": "100%", "height": "50px", "lineHeight": "50px",
                        "borderWidth": "1px", "borderStyle": "dashed",
                        "borderRadius": "5px", "textAlign": "center",
                    },
                    accept=".csv",
                ),
                html.Div(id="bd-upload-status", className="text-muted small mt-1"),
            ], width=5),
            dbc.Col([
                dbc.Label(" ", className="small fw-bold d-block"),
                dbc.Row([
                    dbc.Col(
                        dbc.Button("Use Demo Predictions", id="bd-demo-btn",
                                   color="secondary", outline=True, n_clicks=0),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("▶  Run Backtest", id="bd-run-btn",
                                   color="primary", n_clicks=0),
                        width="auto",
                    ),
                ], className="g-2"),
            ], width=7, className="d-flex align-items-end"),
        ], className="mb-4"),

        dcc.Loading(html.Div(id="bd-results"), type="circle"),
    ])


def register_callbacks(app) -> None:

    @app.callback(
        Output("bd-probs-store", "data"),
        Output("bd-upload-status", "children"),
        Input("bd-upload", "contents"),
        State("bd-upload", "filename"),
        prevent_initial_call=True,
    )
    def parse_upload(contents, filename):
        if not contents:
            return no_update, no_update
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            required = {"home_win", "draw", "away_win"}
            missing = required - set(df.columns)
            if missing:
                return no_update, f"❌ Missing columns: {missing}"
            return df[list(required)].to_json(orient="records"), f"✓ Loaded {len(df)} rows from {filename}"
        except Exception as exc:
            return no_update, f"❌ Parse error: {exc}"

    @app.callback(
        Output("bd-probs-store", "data", allow_duplicate=True),
        Output("bd-upload-status", "children", allow_duplicate=True),
        Input("bd-demo-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def use_demo(n_clicks):
        df = _demo_probs(380)  # one Premier League season
        return df.to_json(orient="records"), "✓ Using 380-match synthetic demo predictions"

    @app.callback(
        Output("bd-backtest-store", "data"),
        Output("bd-results", "children"),
        Input("bd-run-btn", "n_clicks"),
        State("bd-competition", "value"),
        State("bd-bankroll", "value"),
        State("bd-kelly", "value"),
        State("bd-min-edge", "value"),
        State("bd-probs-store", "data"),
        prevent_initial_call=True,
    )
    def run_bt(n_clicks, competition, bankroll, kelly, min_edge, probs_json):
        if not probs_json:
            return no_update, dbc.Alert(
                "Load model predictions first (upload CSV or click 'Use Demo Predictions').",
                color="warning",
            )

        # Download historical odds
        try:
            scraper = FootballDataScraper()
            odds_df = scraper.get_historical_odds(competition)
        except Exception as exc:
            return no_update, dbc.Alert(f"Odds download failed: {exc}", color="danger")

        if odds_df.empty:
            return no_update, dbc.Alert(
                f"No historical odds available for {competition}.", color="warning"
            )

        model_probs = pd.read_json(probs_json, orient="records")

        result = run_backtest(
            model_probs=model_probs,
            odds_df=odds_df,
            bankroll=float(bankroll or 1000),
            kelly_fraction=float(kelly or 0.25),
            min_edge=float(min_edge or 0.05),
        )

        # ── Summary metrics ────────────────────────────────────────────────
        rpt = backtest_report(result)
        roi_color = "success" if result["roi"] >= 0 else "danger"

        summary_cards = dbc.Row([
            dbc.Col(metric_card("ROI", f"{result['roi']:+.1%}", roi_color), width=2),
            dbc.Col(metric_card("Final Bankroll",
                                f"€{result['final_bankroll']:,.0f}", roi_color), width=2),
            dbc.Col(metric_card("Max Drawdown", f"{result['max_drawdown']:.1f}%", "danger"),
                    width=2),
            dbc.Col(metric_card("Sharpe", f"{result['sharpe_ratio']:.2f}",
                                "success" if result["sharpe_ratio"] > 0 else "secondary"), width=2),
            dbc.Col(metric_card("Win Rate", f"{result['win_rate']:.1%}", "primary"), width=2),
            dbc.Col(metric_card("# Bets", str(result["n_bets"]), "secondary"), width=2),
        ], className="mb-3 g-2")

        # ── Figures ────────────────────────────────────────────────────────
        fig_bank = plot_bankroll_history(result)
        fig_edge = plot_edge_distribution(result)
        fig_roi  = plot_cumulative_roi(result)

        graphs = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_bank, config={"displayModeBar": False}), width=12),
        ], className="mb-2")
        graphs2 = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_edge, config={"displayModeBar": False}), width=6),
            dbc.Col(dcc.Graph(figure=fig_roi,  config={"displayModeBar": False}), width=6),
        ], className="mb-3")

        # ── Bet history table (last 20) ────────────────────────────────────
        hist = result["history"]
        table_section = html.Div()
        if not hist.empty:
            display_cols = ["date", "home_team", "away_team", "outcome", "bet_on",
                            "edge", "kelly_fraction", "bet_size", "pnl", "bankroll"]
            display_cols = [c for c in display_cols if c in hist.columns]
            recent = hist.tail(20)[display_cols].copy()
            recent["pnl"] = recent["pnl"].round(2)
            recent["bankroll"] = recent["bankroll"].round(2)
            table_section = html.Div([
                html.H6("Recent Bets (last 20)", className="mt-2 mb-1"),
                dash_table.DataTable(
                    data=recent.to_dict("records"),
                    columns=[{"name": c.replace("_", " ").title(), "id": c}
                             for c in recent.columns],
                    style_table={"overflowX": "auto"},
                    style_cell={"fontSize": "12px", "padding": "5px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                    style_data_conditional=[
                        {"if": {"filter_query": "{pnl} > 0"},
                         "backgroundColor": "#d4edda", "color": "#155724"},
                        {"if": {"filter_query": "{pnl} < 0"},
                         "backgroundColor": "#f8d7da", "color": "#721c24"},
                    ],
                    page_size=20,
                ),
            ])

        # Summary text
        summary_alert = dbc.Alert(result["summary"], color="info", className="mb-3 py-2 small")

        store_data = {
            "roi": result["roi"],
            "n_bets": result["n_bets"],
            "competition": competition,
        }

        return store_data, html.Div([summary_alert, summary_cards, graphs, graphs2, table_section])

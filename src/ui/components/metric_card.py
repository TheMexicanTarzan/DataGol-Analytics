from dash import html
import dash_bootstrap_components as dbc


def metric_card(label: str, value: str, color: str = "primary") -> dbc.Card:
    """Single-metric Bootstrap card for dashboards."""
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
            html.H4(value, className=f"text-{color} mb-0 fw-bold"),
        ]),
        className="text-center shadow-sm h-100",
    )

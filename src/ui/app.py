"""Dash application factory for DataGol Analytics."""

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html


def _navbar() -> dbc.NavbarSimple:
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Player Explorer",    href="/")),
            dbc.NavItem(dbc.NavLink("Lineup Analyzer",    href="/lineup-analyzer")),
            dbc.NavItem(dbc.NavLink("Betting Dashboard",  href="/betting-dashboard")),
        ],
        brand="⚽ DataGol Analytics",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-0",
    )


def create_app() -> Dash:
    """
    Build and return the configured Dash application.

    Pages are imported here (after the Dash instance exists) so their
    @app.callback decorators register against this specific app object.
    """
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,  # required: not all page IDs exist at startup
        title="DataGol Analytics",
    )

    # Import pages now that the app exists; each calls register_callbacks(app)
    from src.ui.pages import player_explorer, lineup_analyzer, betting_dashboard

    player_explorer.register_callbacks(app)
    lineup_analyzer.register_callbacks(app)
    betting_dashboard.register_callbacks(app)

    app.layout = dbc.Container(
        [
            dcc.Location(id="url"),
            _navbar(),
            html.Div(id="page-content", className="py-3"),
        ],
        fluid=True,
        className="px-4",
    )

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def render_page(pathname: str):
        if pathname == "/lineup-analyzer":
            return lineup_analyzer.layout()
        if pathname == "/betting-dashboard":
            return betting_dashboard.layout()
        return player_explorer.layout()

    return app

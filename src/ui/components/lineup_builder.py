from dash import dcc, html
import dash_bootstrap_components as dbc

# Personality category options (1-7 defenders, 8-17 midfielders, 18-23 forwards)
CAT_OPTIONS = (
    [{"label": f"Category {i}  — Defender",    "value": i} for i in range(1, 8)]
    + [{"label": f"Category {i} — Midfielder", "value": i} for i in range(8, 18)]
    + [{"label": f"Category {i} — Forward",    "value": i} for i in range(18, 24)]
)


def lineup_builder(team_id: str, title: str) -> html.Div:
    """
    10 outfield personality-category dropdowns + a GK name field for one team.

    IDs use the dict form so the lineup_analyzer callback can collect all
    values with a single pattern-matching Input:
        State({"type": f"{team_id}-slot", "index": ALL}, "value")
    """
    slots = [
        dcc.Dropdown(
            id={"type": f"{team_id}-slot", "index": i},
            options=CAT_OPTIONS,
            value=None,
            placeholder=f"Player {i + 1}",
            clearable=True,
            className="mb-1",
        )
        for i in range(10)
    ]
    return html.Div(
        [
            html.H6(title, className="fw-bold mb-2"),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("GK"),
                    dbc.Input(
                        id=f"{team_id}-gk",
                        type="text",
                        placeholder="Goalkeeper name (optional, for quality scoring)",
                    ),
                ],
                className="mb-2",
                size="sm",
            ),
            html.Div(slots, style={"maxHeight": "400px", "overflowY": "auto"}),
        ],
        className="border rounded p-2 bg-light",
    )

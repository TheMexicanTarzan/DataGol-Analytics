"""DataGol Analytics — Plotly Dash entry point."""

from src.ui.app import create_app

app = create_app()
server = app.server  # expose for gunicorn / production deployment

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

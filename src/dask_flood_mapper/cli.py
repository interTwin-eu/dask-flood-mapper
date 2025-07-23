"""Dask Flood Mapper CLI."""

# ruff: noqa: T201

import threading
import webbrowser

from dask_flood_mapper.app import app


def open_browser() -> None:
    """Open the web browser to the Dask Flood Mapper application."""
    webbrowser.open_new("http://127.0.0.1:5000")


def main() -> None:
    """Run the Dask Flood Mapper web application."""
    threading.Timer(1.5, open_browser).start()
    app.run(debug=False)


print("🧭 Flask template folder:", app.template_folder)
print("📁 Static folder:", app.static_folder)

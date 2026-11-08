# ui_agent.py -- the Agent tab
#
# A terminal on an agent (Claude Code, or another command line) running inside
# Arcana, in the Arcana workspace, with Arcana's MCP server and skills. Dash
# draws the frame; assets/agent_panel.js draws the launcher, the bar of running
# agents and the terminal, and talks to arcana/agent_term.py. The token on the
# panel is what lets this page, and only this page, type into the terminal.

from __future__ import annotations

from dash import Input, Output, html

try:
    from . import agent_term as _term
    from . import ui_style as _ui
    from . import workspace as _workspace
except ImportError:                                   # pragma: no cover
    import agent_term as _term
    import ui_style as _ui
    import workspace as _workspace

VISIBLE = {"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
           "padding": "10px 12px 12px", "gap": "8px"}
HIDDEN = {"display": "none"}


def layout() -> html.Div:
    return html.Div(
        id="agent-panel",
        style=HIDDEN,
        **{"data-token": _term.TOKEN},
        children=[
            html.Div([
                html.Div([
                    html.Span("AGENT", style={"fontWeight": "700", "fontSize": "12px",
                                              "letterSpacing": "1px", "color": _ui.INK}),
                    html.Span(id="agent-where", style={"color": _ui.INK_DIM, "fontSize": "11px",
                                                       "wordBreak": "break-all"}),
                ], style={"display": "flex", "alignItems": "baseline", "gap": "10px",
                          "flexWrap": "wrap"}),
                html.Div(id="agent-bar", className="agent-bar"),
            ], style={"display": "flex", "flexDirection": "column", "gap": "8px",
                      "flex": "0 0 auto"}),
            html.Div(id="agent-launcher", className="agent-launcher"),
            html.Div(id="agent-term-host", className="agent-term-host",
                     style={"display": "none"}),
        ],
    )


def register(app) -> None:
    _term.register(app.server)

    @app.callback(Output("agent-panel", "style"), Input("mode-select", "value"))
    def _show(mode):
        if mode != "agent":
            return HIDDEN
        # Refresh the workspace's guide, MCP config and skills whenever the tab
        # opens, so an agent started from it always meets the current ones.
        try:
            _workspace.prepare()
        except OSError:
            pass
        return VISIBLE

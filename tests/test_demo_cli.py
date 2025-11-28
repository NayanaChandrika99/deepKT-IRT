# ABOUTME: Verifies demo CLI exposes explain and gaming-check commands.
# ABOUTME: Ensures Typer app loads without side effects.

from scripts import demo_trace


def test_demo_cli_has_explain_and_gaming_commands():
    app = demo_trace.app
    command_names = {cmd.name or cmd.callback.__name__ for cmd in app.registered_commands}
    assert "explain" in command_names
    assert "gaming-check" in command_names or "gaming_check" in command_names

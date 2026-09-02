import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize("mode", ["all", "tracking", "editing"])
def test_main_entrypoint(mode):
    """CLI entrypoint passes correct mode to StartupWidget and adds the tracking widget."""

    viewer = MagicMock(name="viewer")
    motile_widget = MagicMock(name="motile_widget")

    with (
        patch("motile_tracker.__main__.napari.Viewer", return_value=viewer),
        patch("motile_tracker.__main__.napari.run"),
        patch("motile_tracker.__main__.StartupWidget") as mock_startup_widget,
        patch(
            "motile_tracker.__main__.MotileWidget", return_value=motile_widget
        ) as mock_motile_widget,
        patch.object(sys, "argv", ["prog", "--mode", mode]),
    ):
        from motile_tracker.__main__ import main

        main()

    mock_startup_widget.assert_called_once()
    args, kwargs = mock_startup_widget.call_args

    # First positional arg should be viewer
    assert args[0] is viewer

    # mode should match CLI flag
    assert kwargs["mode"] == mode

    # MotileWidget should be constructed with the viewer and added as a dock widget
    mock_motile_widget.assert_called_once_with(viewer)
    viewer.window.add_dock_widget.assert_called_once_with(
        motile_widget, name="Tracking"
    )

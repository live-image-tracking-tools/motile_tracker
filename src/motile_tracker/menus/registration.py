"""Widget contribution that adds the Motile tab to napari-track-edit's TrackingWidget.

Registration happens as an import-time side effect below, so by the time a user
opens this widget from the Plugins menu, napari-track-edit's TrackingWidget already
knows about the Motile tab and includes it alongside "Track from Scratch".

`MotileTrackingWidget` is a thin subclass (rather than a plain factory function)
because napari only auto-injects the viewer argument for widget contributions that
are QWidget subclasses; plain functions are called with no arguments.
"""

from napari import Viewer
from napari_track_edit.application_menus.main_app import Tracking_LauncherWidget
from napari_track_edit.application_menus.tracking_widget import (
    TrackingWidget,
    register_tracking_tab,
)
from qtpy.QtCore import QTimer

from motile_tracker.menus.motile_widget import MotileWidget

register_tracking_tab("Motile", MotileWidget)


class MotileTrackingWidget(Tracking_LauncherWidget):
    """Shows (or creates, on first use) napari-track-edit's shared Tracking widget,
    with the Motile tab selected."""

    def __init__(self, napari_viewer: Viewer):
        super().__init__(napari_viewer)
        # StartupWidget's own setup (switching tabs, removing self) runs on a
        # singleShot(0,...) timer, so queue behind it to select the Motile tab.
        QTimer.singleShot(0, lambda: self._select_motile_tab(napari_viewer))

    @staticmethod
    def _select_motile_tab(napari_viewer: Viewer) -> None:
        dock_widget = napari_viewer.window.dock_widgets.get("Tracking")
        if dock_widget is None:
            return
        tracking_widget = dock_widget.widget()
        if isinstance(tracking_widget, TrackingWidget):
            motile_tab = tracking_widget.tabs.get("Motile")
            if motile_tab is not None:
                tracking_widget.setCurrentWidget(motile_tab)

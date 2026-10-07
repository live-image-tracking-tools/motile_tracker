from __future__ import annotations

import pyqtgraph as pg
from napari_track_edit.data_views.views_coordinator.tracks_viewer import TracksViewer
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible, ensure_main_thread

from motile_tracker.backend import MotileGraph, get_solver_name

from .params_editor import AttributeWeightsEditor, SolverParamsEditor, TilingParamsEditor


class MotileParamsWidget(QGroupBox):
    """Widget for viewing live stats on the currently selected tracks, editing
    SolverParams and TilingParams, and requesting a solve.

    Does not solve itself: emits run_solver, for the caller (MotileWidget) to
    run on a background thread against whatever TracksViewer.tracks currently
    holds.
    """

    run_solver = Signal(object, object)  # solver_params, tiling_params

    def __init__(self, viewer):
        super().__init__(title="Motile Solver")
        self.viewer = viewer
        self.tracks_viewer = TracksViewer.get_instance(viewer)
        self.tracks_viewer.tracks_updated.connect(self.update_stats)

        self.stats_label = QLabel("")
        self.solver_params_widget = SolverParamsEditor()
        self.attribute_weights_widget = AttributeWeightsEditor()
        self.attribute_weights_widget.refresh_requested.connect(
            self._refresh_attribute_weights
        )
        self.tiling_params_widget = TilingParamsEditor()
        self.solver_label = QLabel("")
        self.gap_plot = self._plot_widget()

        run_btn = QPushButton(f"Run Tracking ({get_solver_name()})")
        run_btn.clicked.connect(self.emit_run_solver)
        run_btn.setToolTip("Might take minutes or longer for larger samples.")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stats_label)
        main_layout.addWidget(self.solver_params_widget)
        main_layout.addWidget(self.attribute_weights_widget)
        main_layout.addWidget(self.tiling_params_widget)
        main_layout.addWidget(run_btn)
        main_layout.addWidget(self._progress_widget())
        self.setLayout(main_layout)

        self.update_stats()

    def _refresh_attribute_weights(self) -> None:
        """Rebuild the attribute weight rows from the currently selected
        tracks' node/edge features."""
        self.attribute_weights_widget.refresh_from_tracks(self.tracks_viewer.tracks)

    def update_stats(self, *_) -> None:
        """Refresh the stats label from the currently selected tracks, and
        hide this whole widget when nothing is selected."""
        tracks = self.tracks_viewer.tracks
        if tracks is None:
            self.setVisible(False)
            return
        self.setVisible(True)
        num_nodes = tracks.graph_full.num_nodes()
        num_in_solution = tracks.graph_solution.num_nodes()
        status = tracks.status if isinstance(tracks, MotileGraph) else "n/a"
        self.stats_label.setText(
            f"Nodes: {num_nodes}  |  In solution: {num_in_solution}  |  "
            f"Status: {status}"
        )

    def emit_run_solver(self) -> None:
        """Emit run_solver with a copy of the current solver and tiling params.

        attribute_weights lives on a separate editor instance (so it can be
        rebuilt independently of the rest of SolverParams), so it is merged in
        here rather than living on self.solver_params_widget.solver_params.
        """
        solver_params = self.solver_params_widget.solver_params.copy()
        solver_params.attribute_weights = dict(
            self.attribute_weights_widget.solver_params.attribute_weights
        )
        tiling_params = self.tiling_params_widget.tiling_params.copy()
        self.run_solver.emit(solver_params, tiling_params)

    def _progress_widget(self) -> QWidget:
        """Create a widget containing solver progress and status.

        Returns:
            QWidget: A widget with a label indicating solver status and
                a collapsible graph of the solver gap.
        """
        widget = QWidget()
        layout = QVBoxLayout()

        collapsable_plot = QCollapsible("Graph of solver gap")
        collapsable_plot.layout().setContentsMargins(0, 0, 0, 0)
        collapsable_plot.addWidget(self.gap_plot)
        collapsable_plot.collapse(animate=False)

        layout.addWidget(self.solver_label)
        layout.addWidget(collapsable_plot)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        return widget

    def _plot_widget(self) -> pg.PlotWidget:
        """
        Returns:
            pg.PlotWidget: a widget containg an (empty) plot of the solver gap
        """
        gap_plot = pg.PlotWidget()
        gap_plot.setBackground((37, 41, 49))
        styles = {
            "color": "white",
        }
        gap_plot.plotItem.setLogMode(x=False, y=True)
        gap_plot.plotItem.setLabel("left", "Gap", **styles)
        gap_plot.plotItem.setLabel("bottom", "Solver round", **styles)
        return gap_plot

    def _set_solver_label(self, status: str):
        self.solver_label.setText("Solver status: " + status)

    @ensure_main_thread
    def solver_event_update(self, status: str, gaps: list[float] | None) -> None:
        """Update the solver status label and gap plot.

        Args:
            status (str): The current solver status.
            gaps (list[float] | None): The solver gaps so far, if any.
        """
        self._set_solver_label(status)
        self.gap_plot.getPlotItem().clear()
        if gaps is not None and len(gaps) > 0:
            try:
                self.gap_plot.getPlotItem().plot(range(len(gaps)), gaps)
            # note: catching pyqt graph exception about range(len(gaps))
            # and gaps being different lengths. Pyqtgraph uses a generic
            # Exception :( so we check the string
            except Exception as e:
                if "X and Y arrays" not in str(e):
                    raise e

    def reset_progress(self):
        self._set_solver_label("not running")
        self.gap_plot.getPlotItem().clear()

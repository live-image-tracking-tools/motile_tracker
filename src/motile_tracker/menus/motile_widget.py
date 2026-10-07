# do not put from __future__ import annotations as it breaks the injection

import logging

import numpy as np
from funtracks.data_model import Tracks
from funtracks.utils import ensure_unique_labels
from napari import Viewer
from napari.utils.notifications import show_warning
from napari_track_edit.data_views.views_coordinator.tracks_viewer import TracksViewer
from qtpy.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import thread_worker
from tracksdata.array import GraphArrayView

from motile_tracker.backend import (
    CandidateGraphParams,
    MotileGraph,
    SolverParams,
    TilingParams,
    build_candidate_graph,
    solve,
)

from .candidate_graph_widget import CandidateGraphWidget
from .motile_params_widget import MotileParamsWidget

logger = logging.getLogger(__name__)


class MotileWidget(QWidget):
    """A widget that controls the backend components of the motile tracker.

    Two independent sub-widgets: CandidateGraphWidget (edit params, build a
    candidate graph from the selected input layer) and MotileParamsWidget
    (edit solver/tiling params, run the solver against whatever tracks are
    currently selected in the TracksViewer).
    """

    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer: Viewer = viewer
        self.tracks_viewer = TracksViewer.get_instance(self.viewer)
        self.tracks_viewer.tracks_list.tracks_saved.connect(self._on_tracks_saved)
        self.tracks_viewer.tracks_list.tracks_loaded.connect(self._on_tracks_loaded)

        self.candidate_graph_widget = CandidateGraphWidget(self.viewer)
        self.candidate_graph_widget.build_candidate_graph.connect(
            self._build_candidate_graph
        )

        self.motile_params_widget = MotileParamsWidget(self.viewer)
        self.motile_params_widget.run_solver.connect(self._run_solver)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._title_widget())
        main_layout.addWidget(self.candidate_graph_widget)
        main_layout.addWidget(self.motile_params_widget)
        main_layout.addStretch()
        self.setLayout(main_layout)

    def _on_tracks_saved(self, tracks: Tracks, path) -> None:
        """Write motile run metadata (candidate graph/solver/tiling params,
        gaps) next to tracks that napari-track-edit just saved as a geff.

        napari-track-edit's TracksList writes only the geff store; it has no
        notion of MotileGraph, so this is the only place motile-specific data
        is ever saved. No-ops for tracks that are not a MotileGraph.
        """
        if not isinstance(tracks, MotileGraph):
            return
        tracks.save_metadata(path)

    def _on_tracks_loaded(self, tracks: Tracks, path) -> None:
        """Rewrap tracks that were loaded with motile run metadata as a
        MotileGraph, so its params/gaps are not silently dropped on load.

        napari-track-edit's TracksList loads plain tracks with no notion of
        MotileGraph. If the loaded path has motile metadata, replaces the
        plain tracks list entry with a MotileGraph wrapping the same graph.
        """
        if MotileGraph._load_params(path) is None:
            return

        tracks_list = self.tracks_viewer.tracks_list
        list_widget = tracks_list.tracks_list
        for row in range(list_widget.count() - 1, -1, -1):
            item = list_widget.item(row)
            button = list_widget.itemWidget(item)
            if button.tracks is tracks:
                name = button.name.text()
                tracks_list.remove_tracks(item)
                break
        else:
            return

        run = MotileGraph(
            graph=tracks.graph_full,
            ndim=tracks.ndim,
            scale=tracks.scale,
            time_attr=tracks.features.time_key,
            pos_attr=tracks.features.position_key,
            _features=tracks.features,
            _segmentation=tracks.segmentation,
        )
        run.load_metadata(path)
        tracks_list.add_tracks(run, name, select=True)

    def _build_candidate_graph(
        self, input_data: np.ndarray, params: CandidateGraphParams, scale: list[float]
    ) -> None:
        """Called when the candidate graph widget requests a build. Starts
        building in a separate thread to avoid blocking.
        """
        worker = self.build_candidate_run(input_data, params, scale)
        worker.returned.connect(self._on_build_candidate_complete)
        worker.start()

    @thread_worker
    def build_candidate_run(
        self,
        input_data: np.ndarray,
        candidate_graph_params: CandidateGraphParams,
        scale: list[float],
    ) -> MotileGraph:
        """Builds the candidate graph for the given input data and returns a
        MotileGraph wrapping it (not yet solved).
        """
        try:
            cand_graph = build_candidate_graph(input_data, candidate_graph_params, scale)
        except ValueError as e:
            if "Duplicate values found among nodes" in str(e):
                input_data = ensure_unique_labels(input_data)
                cand_graph = build_candidate_graph(
                    input_data, candidate_graph_params, scale
                )
            else:
                raise

        return MotileGraph(
            graph=cand_graph,
            scale=scale,
            candidate_graph_params=candidate_graph_params,
            status="candidate",
        )

    def _on_build_candidate_complete(self, run: MotileGraph) -> None:
        """Called when the candidate graph building thread returns. Adds the
        new candidate graph to the tracks list, which selects and displays it.
        """
        run.status = "candidate"
        self.tracks_viewer.tracks_list.add_tracks(run, "candidate graph", select=True)

    def _run_solver(
        self, solver_params: SolverParams, tiling_params: TilingParams
    ) -> None:
        """Called when the motile params widget requests a solve. Solves
        whatever tracks are currently selected in the TracksViewer, in a
        separate thread to avoid blocking.
        """
        tracks = self.tracks_viewer.tracks
        if tracks is None:
            show_warning("No tracks selected to solve")
            return
        worker = self.solve_with_motile(tracks, solver_params, tiling_params)
        worker.returned.connect(self._on_solve_complete)
        worker.start()

    @thread_worker
    def solve_with_motile(
        self,
        tracks: Tracks,
        solver_params: SolverParams,
        tiling_params: TilingParams,
    ) -> Tracks:
        """Runs the solver on the given tracks and recomputes track ids.

        Args:
            tracks: The tracks (candidate graph) to solve. Solved in place.
            solver_params: The solver parameters to use.
            tiling_params: The chunked/tiled solving parameters to use.

        Returns:
            Tracks: The same tracks object, now solved.
        """
        if isinstance(tracks, MotileGraph):
            tracks.status = "initializing"
            tracks.solver_params = solver_params
            tracks.tiling_params = tiling_params

        solve(
            tracks,
            solver_params,
            tiling_params,
            lambda event_data: self._on_solver_event(tracks, event_data),
        )

        if isinstance(tracks, MotileGraph):
            tracks.status = "done"

        # Solving replaces the graph's topology, so the tracklet/lineage ids
        # computed when the candidate graph was built are stale (or -1-sentinel
        # placeholders); recompute them from the newly solved topology.
        tracks.enable_features(
            [tracks.features.tracklet_key, tracks.features.lineage_key],
            recompute=True,
        )
        if "mask" in tracks.graph_solution.node_attr_keys():
            seg_shape = tracks.graph_solution.metadata.get("shape")
            if seg_shape is not None:
                tracks.segmentation = GraphArrayView(
                    graph=tracks.graph_solution,
                    shape=seg_shape,
                    attr_key="node_id",
                    offset=0,
                )

        if tracks.segmentation is not None:
            # recompute=False: area values are already on the graph nodes
            # because compute_graph_from_seg computes area during node extraction.
            tracks.enable_features(["area"], recompute=False)

        if tracks.graph_solution.num_nodes() == 0:
            show_warning(
                "No tracks found - try making your edge selection value more negative"
            )
        return tracks

    def _on_solver_event(self, tracks: Tracks, event_data: dict) -> None:
        """Parse the solver event and update status/gaps, then refresh the
        motile params widget's progress display.
        """
        if not isinstance(tracks, MotileGraph):
            return
        event_type = event_data["event_type"]
        if event_type in ["PRESOLVE", "PRESOLVEROUND"] and tracks.status != "presolving":
            tracks.status = "presolving"
            tracks.gaps = []  # try this to remove the weird initial gap for gurobi
            self.motile_params_widget.solver_event_update(tracks.status, tracks.gaps)
        elif event_type in ["MIPSOL", "BESTSOLFOUND"]:
            tracks.status = "solving"
            gap = event_data["gap"]
            if tracks.gaps is None:
                tracks.gaps = []
            tracks.gaps.append(gap)
            self.motile_params_widget.solver_event_update(tracks.status, tracks.gaps)

    def _on_solve_complete(self, tracks: Tracks) -> None:
        """Called when the solver thread returns. Refreshes the stats/progress
        display and the napari layers; the tracks object was solved in place
        so no new row is added to the tracks list.
        """
        status = tracks.status if isinstance(tracks, MotileGraph) else "done"
        gaps = tracks.gaps if isinstance(tracks, MotileGraph) else None
        self.motile_params_widget.solver_event_update(status, gaps)
        self.motile_params_widget.update_stats()

        # solve() mutates tracks.graph_full/graph_solution directly rather than
        # through the normal action-recording path, so the viewer's layers
        # never heard about the change. Only refresh if these are still the
        # tracks currently being viewed (the user may have switched away
        # during the background solve).
        if self.tracks_viewer.tracks is tracks:
            self.tracks_viewer._refresh(refresh_view=True)

    def _title_widget(self) -> QWidget:
        """Create the intro paragraph widget, with links to motile docs.

        Returns:
            QWidget: A widget describing the tracking tab and linking to motile docs
        """
        richtext = r"""<p>This tab uses the
        <a href="https://funkelab.github.io/motile/"><font color=yellow>motile</font></a> library to
        track objects with global optimization.</p>"""
        label = QLabel(richtext)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        return label

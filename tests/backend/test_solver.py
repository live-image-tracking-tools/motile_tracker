import numpy as np
import pytest
from funtracks.data_model import Tracks
from funtracks.utils.tracksdata_utils import assert_node_attrs_equal_with_masks

from motile_tracker.backend import SolverParams, solve
from motile_tracker.backend.solve import build_candidate_graph


def _tracks_from(input_data, solver_params, scale=None):
    """Build a Tracks object wrapping the candidate graph for input_data."""
    cand_graph = build_candidate_graph(input_data, solver_params, scale=scale)
    # A points list is an (N, D) array of D-dimensional points (e.g. t, y, x);
    # a segmentation's own ndim (t, [z], y, x) already matches the Tracks ndim.
    ndim = input_data.shape[1] if input_data.ndim == 2 else input_data.ndim
    return Tracks(cand_graph, ndim=ndim, time_attr="t")


# capsys is a pytest fixture that captures stdout and stderr output streams
def test_solve_2d(graph_2d, segmentation_2d):
    params = SolverParams()
    params.appear_cost = None
    tracks = _tracks_from(segmentation_2d, params)
    solve(tracks, params)
    soln_graph = tracks.graph_solution

    # remove nodes that don't make the solution
    # node 4 is too far from node 3
    # node 5 is two frames from node 4
    # node 6 is isolated and has no edges
    for node in [4, 5, 6]:
        graph_2d.remove_node(node)
    assert set(soln_graph.node_ids()) == set(graph_2d.node_ids())


def test_solve_3d(graph_3d, segmentation_3d):
    params = SolverParams()
    params.appear_cost = None
    tracks = _tracks_from(segmentation_3d, params)
    solve(tracks, params)
    assert set(tracks.graph_solution.node_ids()) == set(graph_3d.node_ids())


def test_solve_chunked(segmentation_3d):
    """Test that chunked solving produces same results as full solve."""
    # First solve without chunking
    params = SolverParams()
    params.appear_cost = None
    full_tracks = _tracks_from(segmentation_3d, params)
    solve(full_tracks, params)
    full_solution = full_tracks.graph_solution

    # Then solve with chunking
    params_chunked = SolverParams()
    params_chunked.appear_cost = None
    params_chunked.window_size = 3
    params_chunked.overlap_size = 1
    chunked_tracks = _tracks_from(segmentation_3d, params_chunked)
    solve(chunked_tracks, params_chunked)
    chunked_solution = chunked_tracks.graph_solution

    # Solutions should have the same nodes and edges
    assert set(full_solution.node_ids()) == set(chunked_solution.node_ids())
    assert_node_attrs_equal_with_masks(
        full_solution, chunked_solution, check_row_order=False
    )
    assert {tuple(e) for e in full_solution.edge_list()} == {
        tuple(e) for e in chunked_solution.edge_list()
    }


def test_solve_chunked_multiple_windows_no_boundary_discontinuity():
    """A single track spanning many windows should stay fully connected.

    Regression test: edges whose source frame was the last pinned frame of
    the previous window (e.g. source in [window_start, window_start +
    overlap_size)) were being dropped at every window boundary, because the
    aggregation step filtered edges by source time instead of target time.
    """
    n_frames = 12
    points = np.array([[t, 10.0 + t, 10.0, 10.0] for t in range(n_frames)])

    params_full = SolverParams()
    params_full.appear_cost = None
    params_full.iou_cost = None
    full_tracks = _tracks_from(points, params_full)
    solve(full_tracks, params_full)
    full_solution = full_tracks.graph_solution

    params_chunked = SolverParams()
    params_chunked.appear_cost = None
    params_chunked.iou_cost = None
    params_chunked.window_size = 4
    params_chunked.overlap_size = 2
    chunked_tracks = _tracks_from(points, params_chunked)
    solve(chunked_tracks, params_chunked)
    chunked_solution = chunked_tracks.graph_solution

    assert set(full_solution.node_ids()) == set(chunked_solution.node_ids())
    assert {tuple(e) for e in full_solution.edge_list()} == {
        tuple(e) for e in chunked_solution.edge_list()
    }


def test_solve_chunked_overlap_required():
    """Test that overlap_size must be at least 1."""
    params = SolverParams()
    params.window_size = 3

    with pytest.raises(ValueError, match="overlap_size must be at least 1"):
        params.overlap_size = 0


def test_solve_single_window(segmentation_3d):
    """Test solving just a single window for interactive testing."""
    params = SolverParams()
    params.appear_cost = None
    params.window_size = 3
    params.single_window_start = 1  # Start at frame 1

    tracks = _tracks_from(segmentation_3d, params)
    solve(tracks, params)
    solution = tracks.graph_solution

    # Should only have nodes from frames 1, 2, 3
    assert solution.num_nodes() > 0
    # Verify all nodes are within the window
    for node in solution.node_ids():
        node_time = solution.nodes[node]["t"]
        assert 1 <= node_time < 4, f"Node {node} has time {node_time}, expected 1-3"


def test_solve_single_window_start_0(segmentation_2d):
    """Window starting at frame 0 — no t-shift should be applied."""
    params = SolverParams()
    params.appear_cost = None
    params.window_size = 2
    params.single_window_start = 0

    tracks = _tracks_from(segmentation_2d, params)
    solve(tracks, params)
    solution = tracks.graph_solution

    assert solution.num_nodes() > 0
    for node in solution.node_ids():
        node_time = solution.nodes[node]["t"]
        assert 0 <= node_time < 2, f"Node {node} has time {node_time}, expected 0-1"


def test_solve_single_window_points():
    """Single-window mode with a points list (ndim==2 branch) as input."""
    # Columns: (t, y, x) — points close enough to form edges within default max_edge_distance
    points = np.array(
        [
            [0, 50.0, 50.0],
            [1, 51.0, 51.0],
            [2, 52.0, 52.0],
            [3, 53.0, 53.0],
        ]
    )
    params = SolverParams()
    params.appear_cost = None
    params.iou_cost = None  # points graphs have no iou edge attribute
    params.window_size = 2
    params.single_window_start = 1

    tracks = _tracks_from(points, params)
    solve(tracks, params)
    solution = tracks.graph_solution

    assert solution.num_nodes() > 0
    for node in solution.node_ids():
        node_time = solution.nodes[node]["t"]
        assert 1 <= node_time < 3, f"Node {node} has time {node_time}, expected 1-2"


def test_solve_single_window_invalid_start(segmentation_3d):
    """Test that invalid window_start raises ValueError."""

    params = SolverParams()
    params.appear_cost = None
    params.window_size = 3
    params.single_window_start = 100  # Beyond data range (5 frames)

    tracks = _tracks_from(segmentation_3d, params)
    with pytest.raises(ValueError, match="beyond last frame"):
        solve(tracks, params)

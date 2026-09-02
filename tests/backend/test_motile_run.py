import warnings

import numpy as np
from funtracks.import_export import import_from_geff, write_to_geff

from motile_tracker.backend import MotileRun, SolverParams


def test_save_metadata_writes_params_inside_the_geff(tmp_path, graph_2d):
    """Solver params live inside the store, not beside it."""
    run = MotileRun(graph=graph_2d, run_name="test", solver_params=SolverParams())
    path = tmp_path / "my_run.geff"
    write_to_geff(run, path, overwrite=True)

    run.save_metadata(path)

    assert (path / "solver_params.json").exists()
    assert (path / "attrs.json").exists()


def test_resave_metadata_is_quiet(tmp_path, graph_2d):
    """Overwriting a run's metadata must not warn about its own files."""
    run = MotileRun(graph=graph_2d, run_name="test", solver_params=SolverParams())
    path = tmp_path / "my_run.geff"
    write_to_geff(run, path, overwrite=True)
    run.save_metadata(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run.save_metadata(path)

    unrecognized = [
        w for w in caught if "not recognized as a component" in str(w.message)
    ]
    non_geff = [w for w in caught if "non-geff members" in str(w.message)]
    assert unrecognized == []
    assert non_geff == []


def test_resave_metadata_preserves_params(tmp_path, graph_2d):
    """Re-saving metadata over itself keeps the params readable."""
    run = MotileRun(graph=graph_2d, run_name="test", solver_params=SolverParams())
    path = tmp_path / "my_run.geff"
    write_to_geff(run, path, overwrite=True)
    run.save_metadata(path)
    run.save_metadata(path)

    assert (path / "solver_params.json").exists()
    tracks = import_from_geff(path)
    assert MotileRun.load_metadata(tracks, path).solver_params == run.solver_params


def test_load_metadata_run_dir_renamed_to_non_timestamp(tmp_path, graph_2d):
    """A run directory the user renamed must still load its metadata.

    The name and time come from the attrs file, so they survive a rename that
    _unpack_id could not parse.
    """
    run = MotileRun(graph=graph_2d, run_name="my_run", solver_params=SolverParams())
    path = tmp_path / "my_run.geff"
    write_to_geff(run, path, overwrite=True)
    run.save_metadata(path)
    renamed = path.rename(tmp_path / "not_a_timestamp")

    tracks = import_from_geff(renamed)
    loaded = MotileRun.load_metadata(tracks, renamed)

    assert loaded.run_name == "my_run"
    assert loaded.time == run.time


def test_load_metadata_falls_back_to_unpack_id_without_attrs(tmp_path, graph_2d):
    """Runs saved before the name/time were written to attrs still load by
    unpacking the timestamped directory name."""
    run = MotileRun(graph=graph_2d, run_name="test", solver_params=SolverParams())
    # reproduce the old layout: a directory named by _make_id
    path = tmp_path / run._make_id()
    write_to_geff(run, path, overwrite=True)
    run.save_metadata(path)
    (path / "attrs.json").unlink()

    tracks = import_from_geff(path)
    loaded = MotileRun.load_metadata(tracks, path)

    assert loaded.run_name == "test"
    # the directory-name timestamp only has second granularity
    assert loaded.time == run.time.replace(microsecond=0)


def test_resolve_name_and_time_falls_back_to_dir_stem(tmp_path):
    """With neither attrs nor a parseable directory name, the directory name
    is used and the time is left for __init__ to fill in."""
    time, name = MotileRun._resolve_name_and_time(tmp_path / "some_run", None)

    assert name == "some_run"
    assert time is None


def test_save_load_metadata(tmp_path, graph_2d):
    run_name = "test"
    scale = [1.0, 2.0, 3.0]
    run = MotileRun(
        graph=graph_2d,
        run_name=run_name,
        solver_params=SolverParams(),
        scale=scale,
    )
    path = tmp_path / "test.geff"
    write_to_geff(run, path, overwrite=True)
    run.save_metadata(path)

    tracks = import_from_geff(path)
    newrun = MotileRun.load_metadata(tracks, path)

    assert set(run.graph.node_ids()) == set(newrun.graph.node_ids())
    assert {tuple(e) for e in run.graph.edge_list()} == {
        tuple(e) for e in newrun.graph.edge_list()
    }
    assert run.run_name == newrun.run_name
    assert np.array_equal(np.asarray(run.segmentation), np.asarray(newrun.segmentation))
    # the time now round-trips exactly: it comes from the attrs file rather
    # than from the second-granularity timestamp in the directory name
    assert run.time == newrun.time
    assert run.gaps == newrun.gaps
    assert run.scale == newrun.scale
    assert run.solver_params == newrun.solver_params
    # Verify core accessor methods work on the loaded run
    # (regression: time_attr mismatch after load caused KeyError in get_time)
    node_ids = list(newrun.graph.node_ids())
    for node_id in node_ids:
        newrun.get_time(node_id)
        newrun.get_position(node_id)
        newrun.get_track_id(node_id)
    newrun.get_positions(node_ids, incl_time=True)

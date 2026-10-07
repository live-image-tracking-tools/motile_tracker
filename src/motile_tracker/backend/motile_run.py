from __future__ import annotations

import json
from pathlib import Path

import tracksdata as td
from funtracks.data_model import Tracks

from .solver_params import CandidateGraphParams, SolverParams, TilingParams

PARAMS_FILENAME = "solver_params.json"
CANDIDATE_GRAPH_PARAMS_FILENAME = "candidate_graph_params.json"
TILING_PARAMS_FILENAME = "tiling_params.json"
GAPS_FILENAME = "gaps.txt"

# Internal edge attr keys managed by tracksdata — not user-facing features
_TRACKSDATA_INTERNAL_EDGE_KEYS = frozenset({"edge_id", "source_id", "target_id"})


class MotileGraph(Tracks):
    """A Tracks wrapper for a motile run's graph (candidate or solved),
    carrying the params used to build/solve it plus solving bookkeeping
    (status, gaps).
    """

    def __init__(
        self,
        graph: td.graph.BaseGraph,
        time_attr: str = "t",
        pos_attr: str | tuple[str] | list[str] = "pos",
        scale: list[float] | None = None,
        ndim: int | None = None,
        candidate_graph_params: CandidateGraphParams | None = None,
        solver_params: SolverParams | None = None,
        tiling_params: TilingParams | None = None,
        gaps: list[float] | None = None,
        status: str = "done",
        _features=None,
        _segmentation=None,
    ):
        super().__init__(
            graph,
            time_attr=time_attr,
            pos_attr=pos_attr,
            scale=scale,
            ndim=ndim,
            features=_features,
            _segmentation=_segmentation,
        )
        self.candidate_graph_params = candidate_graph_params
        self.solver_params = solver_params
        self.tiling_params = tiling_params
        self.gaps = gaps
        self.status = status

    def save_metadata(self, path: str | Path) -> Path:
        """Save the run's metadata (candidate graph params, solver params,
        tiling params, gaps) inside the geff store at the provided path.

        Assumes the geff store at `path` already exists — writing the tracks
        themselves is the caller's responsibility (funtracks.import_export);
        this only writes the motile-specific data alongside it. A geff is a
        zarr directory, and writing a geff only replaces geff-controlled
        groups, so these files survive the tracks being saved again over the
        same store.

        Args:
            path (str | Path): The geff store to save the run's metadata into.

        Returns:
            (Path): The Path the metadata was saved to.
        """
        run_dir = Path(path)
        self._save_params(run_dir)
        self._save_list(list_to_save=self.gaps, run_dir=run_dir, filename=GAPS_FILENAME)
        return run_dir

    def load_metadata(self, path: str | Path) -> None:
        """Load saved run metadata (candidate graph params, solver params,
        tiling params, gaps) onto this MotileGraph in place.

        Args:
            path (str | Path): The geff store the run's metadata was saved
                into by :meth:`save_metadata`.
        """
        run_dir = Path(path)
        self.candidate_graph_params = self._load_candidate_graph_params(run_dir)
        self.solver_params = self._load_params(run_dir)
        self.tiling_params = self._load_tiling_params(run_dir)
        self.gaps = self._load_list(
            run_dir=run_dir, filename=GAPS_FILENAME, required=False
        )

    def _save_params(self, run_dir: Path):
        """Save the run parameters in the provided run directory. Currently
        dumps the parameters dict into json files. Skips writing a file for
        whichever params are None, which only happens for a run loaded from a
        directory that had no params file (see _load_params).

        Args:
            run_dir (Path): A directory in which to save the parameters files.
        """
        if self.solver_params is not None:
            params_file = run_dir / PARAMS_FILENAME
            with open(params_file, "w") as f:
                json.dump(self.solver_params.__dict__, f)
        if self.candidate_graph_params is not None:
            cand_params_file = run_dir / CANDIDATE_GRAPH_PARAMS_FILENAME
            with open(cand_params_file, "w") as f:
                json.dump(self.candidate_graph_params.__dict__, f)
        if self.tiling_params is not None:
            tiling_params_file = run_dir / TILING_PARAMS_FILENAME
            with open(tiling_params_file, "w") as f:
                json.dump(self.tiling_params.__dict__, f)

    @staticmethod
    def _load_params(run_dir: Path) -> SolverParams | None:
        """Load solver parameters from the parameters json file in the
        provided directory. Returns None if the file is absent, which is the
        case for v1 run directories and for runs saved by versions that
        wrapped imported (CSV/geff) tracks in a MotileRun with no solver
        params.

        Args:
            run_dir (Path): The directory in which to find the parameters file.

        Returns:
            SolverParams | None: The solver parameters, or None if no params
                file exists in the run directory.
        """
        params_file = run_dir / PARAMS_FILENAME
        if not params_file.is_file():
            return None
        with open(params_file) as f:
            params_dict = json.load(f)
        return SolverParams(**params_dict)

    @staticmethod
    def _load_candidate_graph_params(run_dir: Path) -> CandidateGraphParams | None:
        """Load candidate graph parameters from the parameters json file in
        the provided directory. Returns None if the file is absent.

        Args:
            run_dir (Path): The directory in which to find the parameters file.

        Returns:
            CandidateGraphParams | None: The candidate graph parameters, or
                None if no params file exists in the run directory.
        """
        params_file = run_dir / CANDIDATE_GRAPH_PARAMS_FILENAME
        if not params_file.is_file():
            return None
        with open(params_file) as f:
            params_dict = json.load(f)
        return CandidateGraphParams(**params_dict)

    @staticmethod
    def _load_tiling_params(run_dir: Path) -> TilingParams | None:
        """Load tiling parameters from the parameters json file in the
        provided directory. Returns None if the file is absent.

        Args:
            run_dir (Path): The directory in which to find the parameters file.

        Returns:
            TilingParams | None: The tiling parameters, or None if no params
                file exists in the run directory.
        """
        params_file = run_dir / TILING_PARAMS_FILENAME
        if not params_file.is_file():
            return None
        with open(params_file) as f:
            params_dict = json.load(f)
        return TilingParams(**params_dict)

    def _save_list(self, list_to_save: list | None, run_dir: Path, filename: str):
        if list_to_save is None:
            return
        list_file = run_dir / filename
        with open(list_file, "w") as f:
            f.write(",".join(map(str, list_to_save)))

    @staticmethod
    def _load_list(run_dir: Path, filename: str, required: bool = True) -> list[float]:
        list_file = run_dir / filename
        if list_file.is_file():
            with open(list_file) as f:
                file_content = f.read()
            if file_content == "":
                return None
            list_values = list(map(float, file_content.split(",")))
            return list_values
        elif required:
            raise FileNotFoundError(f"No content found at {list_file}")
        else:
            return None

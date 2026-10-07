from __future__ import annotations

import logging

import numpy as np
import polars as pl
import tracksdata as td
from funtracks.candidate_graph import (
    compute_graph_from_points_list,
    compute_graph_from_seg,
)

from .solver_params import CandidateGraphParams

logger = logging.getLogger(__name__)


def build_candidate_graph(
    input_data: np.ndarray,
    candidate_graph_params: CandidateGraphParams,
    scale: list | None = None,
    time_offset: int = 0,
) -> td.graph.BaseGraph:
    """Build the candidate graph from input data.

    If candidate_graph_params.single_window_start is set, input_data is sliced
    down to just that window (single_window_size frames, or the rest of the
    data if unset) before building, so the candidate graph only contains nodes
    from that window — useful for interactively testing parameters on a small
    portion of the data before running on the full dataset. Node times stay
    absolute via time_offset.
    """
    single_window_start = candidate_graph_params.single_window_start
    if single_window_start is not None:
        max_time = (
            input_data.shape[0] - 1
            if input_data.ndim != 2
            else int(input_data[:, 0].max())
        )
        if single_window_start > max_time:
            raise ValueError(
                f"single_window_start ({single_window_start}) is beyond "
                f"last frame ({max_time})"
            )
        single_window_size = candidate_graph_params.single_window_size
        window_end = (
            max_time + 1
            if single_window_size is None
            else min(single_window_start + single_window_size, max_time + 1)
        )
        if input_data.ndim == 2:
            row_mask = (input_data[:, 0] >= single_window_start) & (
                input_data[:, 0] < window_end
            )
            input_data = input_data[row_mask]
        else:
            # numpy slice is a view (no copy)
            input_data = input_data[single_window_start:window_end]
        time_offset = single_window_start

    if input_data.ndim == 2:
        cand_graph = compute_graph_from_points_list(
            input_data, candidate_graph_params.max_edge_distance, scale=scale
        )
    else:
        cand_graph = compute_graph_from_seg(
            input_data,
            candidate_graph_params.max_edge_distance,
            scale=scale,
            t_start=time_offset,
        )
    logger.debug("Cand graph has %d nodes", cand_graph.num_nodes())

    # A candidate graph holds every possible node/edge, not yet a solution —
    # Tracks defaults "solution" to True (correct for wrapping an already-solved
    # result), so it must be explicitly cleared here or graph_solution would show
    # the entire unsolved candidate tangle.
    if "solution" not in cand_graph.node_attr_keys():
        cand_graph.add_node_attr_key("solution", default_value=False, dtype=pl.Boolean)
    else:
        cand_graph.update_node_attrs(
            node_ids=cand_graph.node_ids(), attrs={"solution": False}
        )
    if "solution" not in cand_graph.edge_attr_keys():
        cand_graph.add_edge_attr_key("solution", default_value=False, dtype=pl.Boolean)
    else:
        cand_graph.update_edge_attrs(
            edge_ids=cand_graph.edge_ids(), attrs={"solution": False}
        )

    return cand_graph

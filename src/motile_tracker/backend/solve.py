from __future__ import annotations

import logging
import time
from collections.abc import Callable

import ilpy
import polars as pl
import tracksdata as td
from funtracks.data_model import Tracks
from motile import Solver, TrackGraph
from motile.constraints import MaxChildren, MaxParents
from motile.constraints.constraint import Constraint
from motile.costs import Appear, EdgeDistance, EdgeSelection, NodeSelection, Split
from motile.variables import EdgeSelected, NodeSelected
from tracksdata.constants import DEFAULT_ATTR_KEYS

from .solver_params import SolverParams, TilingParams

logger = logging.getLogger(__name__)

PIN_ATTR = "pinned"
# tracksdata attribute storage has no null representation (add_node_attr_key's
# default_value=None is inferred from dtype, not stored as null -- see
# infer_default_value_from_dtype), so PIN_ATTR uses an int sentinel instead of
# a bool: -1 means unconstrained, 0 means pinned-unselected, 1 means pinned-selected.
PIN_UNSET = -1
PIN_UNSELECTED = 0
PIN_SELECTED = 1

_SKIP_ATTRS = {DEFAULT_ATTR_KEYS.MASK, DEFAULT_ATTR_KEYS.BBOX}


def graphview_to_motile_dicts(
    cand_graph: td.graph.GraphView,
) -> tuple[dict[int, dict], dict[tuple[int, int], dict]]:
    """Unpack a tracksdata ``GraphView`` into plain node/edge dicts for ``motile.TrackGraph``.

    ``GraphView`` is always backed by an in-memory ``rustworkx.PyDiGraph`` (regardless
    of the root graph's backend), so this walks that graph directly instead of going
    through tracksdata's polars-based ``node_attrs()``/``edge_attrs()``, which is
    dramatically slower for large candidate graphs.

    Args:
        cand_graph: The candidate graph to unpack. Node and edge attribute dicts are
            reused by reference (not copied), matching how ``GraphView`` itself shares
            attribute storage with an in-memory root.

    Returns:
        A tuple ``(nodes, edges)`` matching the shape expected by
        ``motile.TrackGraph.add_node``/``add_edge``:

        - ``nodes``: mapping from node id to its attribute dict.
        - ``edges``: mapping from ``(source_id, target_id)`` to its attribute dict.
    """
    rx_graph = cand_graph.rx_graph
    node_ids = cand_graph.node_ids()

    nodes: dict[int, dict] = {}
    for local_idx, node_id in zip(rx_graph.node_indices(), node_ids, strict=True):
        attrs = rx_graph[local_idx]
        nodes[node_id] = {k: v for k, v in attrs.items() if k not in _SKIP_ATTRS}

    local_to_external = dict(zip(rx_graph.node_indices(), node_ids, strict=True))

    edges: dict[tuple[int, int], dict] = {}
    for src_local, tgt_local, attrs in rx_graph.edge_index_map().values():
        edge_id = (local_to_external[src_local], local_to_external[tgt_local])
        edges[edge_id] = {
            k: v
            for k, v in attrs.items()
            if k
            not in (
                DEFAULT_ATTR_KEYS.EDGE_ID,
                DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                DEFAULT_ATTR_KEYS.EDGE_TARGET,
            )
        }

    return nodes, edges

def _reset_solution(tracks):
    graph = tracks.graph_full
    all_node_ids = graph.node_ids()
    all_edge_ids = graph.edge_ids()
    if all_node_ids:
        graph.update_node_attrs(
            node_ids=all_node_ids,
            attrs={"solution": False, PIN_ATTR: PIN_UNSET},
        )
    if all_edge_ids:
        graph.update_edge_attrs(
            edge_ids=all_edge_ids,
            attrs={"solution": False, PIN_ATTR: PIN_UNSET},
        )

def _get_windows(
    tiling_params: TilingParams, total_time_points: int
) -> tuple[list[tuple[int, int]], bool]:
    """Compute the (start, end) sliding/chunked windows to solve.

    Args:
        tiling_params: Supplies window_size/overlap_size for chunked solving
            over the whole dataset.
        total_time_points: Total number of frames to cover, starting at 0.

    Returns:
        A tuple of (windows, is_windowed). windows is a list of (start, end)
        ranges (end exclusive) covering [0, total_time_points). is_windowed is
        False when window_size is None, in which case windows is a single
        range spanning everything.

    Raises:
        ValueError: If overlap_size is missing/invalid when window_size is set.
    """
    window_size = tiling_params.window_size
    if window_size is None:
        return [(0, total_time_points)], False

    overlap_size = tiling_params.overlap_size
    if overlap_size is None:
        raise ValueError("overlap_size is required when window_size is set")
    if overlap_size >= window_size:
        raise ValueError(
            f"overlap_size ({overlap_size}) must be less than window_size ({window_size})"
        )

    windows = []
    start = 0
    while start < total_time_points:
        end = min(start + window_size, total_time_points)
        windows.append((start, end))
        start += window_size - overlap_size
    return windows, True

def solve(
    tracks: Tracks,
    solver_params: SolverParams,
    tiling_params: TilingParams | None = None,
    on_solver_update: Callable | None = None,
) -> Tracks:
    """Get a tracking solution for the given full candidate tracks and parameters.

    Args:
        tracks: the tracks (with all candidate node and edges and scores)
            to run solving on
        solver_params (SolverParams): The solver parameters to use when
            initializing the solver
        tiling_params (TilingParams, optional): The chunked/tiled solving
            parameters (window_size, overlap_size). Defaults to None, which
            solves every frame in a single window.
        on_solver_update (Callable, optional): A function that is called
            whenever the motile solver emits an event. The function should take
            a dictionary of event data, and can be used to track progress of
            the solver. Defaults to None.

    Returns:
        tracks with solution and pinning attributes set on graph_full
    """
    if tiling_params is None:
        tiling_params = TilingParams()

    graph = tracks.graph_full

    time_points = graph.time_points()
    total_time_points = max(time_points) + 1 if time_points else 0
    windows, windowed = _get_windows(tiling_params, total_time_points)

    # ensure the pin schema exists before _reset_solution writes to it
    if PIN_ATTR not in graph.node_attr_keys():
        graph.add_node_attr_key(PIN_ATTR, default_value=PIN_UNSET, dtype=pl.Int8)
    if PIN_ATTR not in graph.edge_attr_keys():
        graph.add_edge_attr_key(PIN_ATTR, default_value=PIN_UNSET, dtype=pl.Int8)

    _reset_solution(tracks)

    for start, end in windows:
        if windowed:
            window_node_ids = graph.filter(
                (td.NodeAttr("t") >= start) & (td.NodeAttr("t") < end)
            ).node_ids()
            cand_graph = graph.filter(node_ids=window_node_ids).subgraph()
        else:
            cand_graph = graph.filter().subgraph()

        solver = construct_solver(cand_graph, solver_params)
        start_time = time.time()
        solution = solver.solve(verbose=False, on_event=on_solver_update)
        logger.info("Solution took %.2f seconds", time.time() - start_time)

        solution_tg = solver.get_selected_subgraph(solution=solution)
        selected_nodes = set(solution_tg.nodes.keys())
        selected_edges = set(solution_tg.edges.keys())  # (source_id, target_id) pairs

        # Pin every node/edge in this window to the decision just made, not just
        # the overlap region: TernaryPin fixes already-pinned nodes/edges to their
        # existing value, so a later overlapping window reproduces the same
        # decision for them regardless of whether we repin them here.
        window_node_ids = cand_graph.node_ids()
        node_is_selected = [n in selected_nodes for n in window_node_ids]
        graph.update_node_attrs(
            node_ids=window_node_ids,
            attrs={
                "solution": node_is_selected,
                PIN_ATTR: [
                    PIN_SELECTED if s else PIN_UNSELECTED for s in node_is_selected
                ],
            },
        )

        edge_pairs = cand_graph.edge_list()
        window_edge_ids = [cand_graph.edge_id(u, v) for u, v in edge_pairs]
        edge_is_selected = [(u, v) in selected_edges for u, v in edge_pairs]
        graph.update_edge_attrs(
            edge_ids=window_edge_ids,
            attrs={
                "solution": edge_is_selected,
                PIN_ATTR: [
                    PIN_SELECTED if s else PIN_UNSELECTED for s in edge_is_selected
                ],
            },
        )

    # PIN_ATTR is solving-internal bookkeeping; strip it before handing tracks back.
    if PIN_ATTR in graph.node_attr_keys():
        graph.remove_node_attr_key(PIN_ATTR)
    if PIN_ATTR in graph.edge_attr_keys():
        graph.remove_edge_attr_key(PIN_ATTR)

    # graph_solution is a LIVE filtered view whose membership was fixed at
    # Tracks.__init__ time: it propagates attribute value writes to nodes/edges
    # already in the view, but does not re-evaluate which nodes/edges satisfy
    # the solution==True filter after we just changed that attribute. Rebuild it
    # so it reflects the solution we just wrote to graph_full.
    tracks.graph_solution = graph.filter(
        td.NodeAttr("solution") == True,  # noqa: E712
        td.EdgeAttr("solution") == True,  # noqa: E712
    ).subgraph(mode=td.graph.ViewMode.LIVE)

    return tracks

class TernaryPin(Constraint):
    """Like motile's Pin, but treats PIN_UNSET as unconstrained.

    motile.constraints.Pin evaluates `{attribute} == True` for every node/edge
    and only skips ones where the attribute is entirely absent (NameError).
    Since tracksdata can't store nulls, our PIN_ATTR is always present once the
    schema key exists, so Pin would force-unselect every node/edge that hasn't
    actually been decided yet. This constraint instead only pins nodes/edges
    whose attribute value is PIN_SELECTED or PIN_UNSELECTED, leaving PIN_UNSET
    ones free for the solver to decide.
    """

    def __init__(self, attribute: str) -> None:
        self.attribute = attribute

    def instantiate(self, solver: Solver) -> list[ilpy.Constraint]:
        select = ilpy.Constraint()
        exclude = ilpy.Constraint()
        n_selected = 0

        for nodes_or_edges, variable_type in (
            (solver.graph.nodes, NodeSelected),
            (solver.graph.edges, EdgeSelected),
        ):
            indicator_variables = solver.get_variables(variable_type)
            for id_, node_or_edge in nodes_or_edges.items():
                pin_value = node_or_edge.get(self.attribute, PIN_UNSET)
                if pin_value == PIN_SELECTED:
                    select.set_coefficient(indicator_variables[id_], 1)
                    n_selected += 1
                elif pin_value == PIN_UNSELECTED:
                    exclude.set_coefficient(indicator_variables[id_], 1)
                # PIN_UNSET: leave unconstrained

        select.set_relation(ilpy.Relation.Equal)
        select.set_value(n_selected)

        exclude.set_relation(ilpy.Relation.Equal)
        exclude.set_value(0)

        return [select, exclude]


def construct_solver(
    cand_graph: td.graph.GraphView, solver_params: SolverParams
) -> Solver:
    """Construct a motile solver with the parameters specified in the solver
    params object.

    Args:
        cand_graph (td.graph.GraphView): The candidate graph to use in the solver
        solver_params (SolverParams): The costs and constraints to use in
            the solver

    Returns:
        Solver: A motile solver with the specified graph, costs, and
            constraints.
    """
    tg = TrackGraph(frame_attribute="t")

    # Unpack directly from the GraphView's underlying in-memory rustworkx graph,
    # avoiding the overhead of tracksdata's polars-based bulk attribute fetch.
    nodes, edges = graphview_to_motile_dicts(cand_graph)
    logging.info("Done creating motile track graph)")
    tg.nodes = nodes
    for edge_id, attrs in edges.items():
        tg.add_edge(edge_id, attrs)

    solver = Solver(tg)
    solver.add_constraint(MaxChildren(solver_params.max_children))
    solver.add_constraint(MaxParents(1))
    solver.add_constraint(TernaryPin(PIN_ATTR))

    if solver_params.edge_selection_cost is not None:
        solver.add_cost(
            EdgeSelection(
                constant=solver_params.edge_selection_cost,
            ),
            name="edge_const",
        )
    if solver_params.node_selection_cost is not None:
        solver.add_cost(
            NodeSelection(
                constant=solver_params.node_selection_cost,
            ),
            name="node_const",
        )
    if solver_params.appear_cost is not None:
        solver.add_cost(Appear(constant=solver_params.appear_cost))
    if solver_params.division_cost is not None:
        solver.add_cost(Split(constant=solver_params.division_cost))

    if solver_params.distance_cost is not None:
        solver.add_cost(
            EdgeDistance(
                position_attribute="pos",
                weight=solver_params.distance_cost,
            ),
            name="distance",
        )

    node_attr_keys = set(cand_graph.node_attr_keys())
    edge_attr_keys = set(cand_graph.edge_attr_keys())
    for attribute, weight in solver_params.attribute_weights.items():
        is_node_attr = attribute in node_attr_keys
        is_edge_attr = attribute in edge_attr_keys
        if is_node_attr and is_edge_attr:
            raise ValueError(
                f"Attribute '{attribute}' exists as both a node and an edge "
                "feature; cannot determine which cost to apply."
            )
        elif is_node_attr:
            solver.add_cost(
                NodeSelection(weight=weight, attribute=attribute), name=attribute
            )
        elif is_edge_attr:
            solver.add_cost(
                EdgeSelection(weight=weight, attribute=attribute), name=attribute
            )
        else:
            raise ValueError(
                f"Attribute '{attribute}' is not a node or edge feature on the "
                "candidate graph."
            )
    return solver


def get_solver_name() -> str:
    """Return the name of the ILP solver backend that will be used.

    Attempts Gurobi first; falls back to SCIP.
    """
    try:
        ilpy.solver_backends.create_solver_backend(ilpy.Preference.Gurobi)
        return "Gurobi"
    except RuntimeError:
        return "SCIP"

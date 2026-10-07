from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Node/edge feature keys that are never offered as selectable solving weights
# (position is handled by distance_cost, not a generic attribute weight).
_RESERVED_FEATURE_KEYS = frozenset({"t", "pos", "solution", "tracklet_id", "lineage_id"})


class CandidateGraphParams(BaseModel):
    """The set of parameters used to build a candidate graph from input data.
    Used to build the UI as well as store parameters for runs.
    """

    model_config = ConfigDict(validate_assignment=True)

    max_edge_distance: float = Field(
        50.0,
        title="Max Move Distance",
        description=r"""The maximum distance an object center can move between time frames.
Objects further than this cannot be matched, but making this value larger will increase solving time.""",
    )
    single_window_start: int | None = Field(
        None,
        title="Single Window Start",
        description=r"""If set along with single_window_size, only build a candidate graph for
a single window starting at this frame index. Useful for interactively testing parameters
on a small portion of the data before running on the full dataset.""",
        json_schema_extra={"ui_default": 0},
    )
    single_window_size: int | None = Field(
        None,
        title="Single Window Size",
        description=r"""Number of time frames in the single window started at
single_window_start.""",
        json_schema_extra={"ui_default": 50},
    )

    @field_validator("single_window_size")
    @classmethod
    def single_window_size_must_be_at_least_two(cls, v: int | None) -> int | None:
        if v is not None and v < 2:
            raise ValueError("single_window_size must be at least 2")
        return v


class TilingParams(BaseModel):
    """The set of parameters controlling chunked (tiled) solving: splitting
    the full time range into overlapping windows, solving each in turn, and
    pinning the overlap region between consecutive windows.
    """

    model_config = ConfigDict(validate_assignment=True)

    window_size: int | None = Field(
        None,
        title="Window Size",
        description=r"""Number of time frames to solve at once when using chunked solving.
If None, solve all frames at once. If set, the problem will be solved in windows
of this size, with overlapping regions pinned to maintain consistency.""",
        json_schema_extra={"ui_default": 50},
    )
    overlap_size: int | None = Field(
        None,
        title="Overlap Size",
        description=r"""Number of time frames to overlap between windows when using chunked solving.
Only used if window_size is set. The overlap region from the previous window will
be pinned when solving the next window. Must be less than window_size.""",
        json_schema_extra={"ui_default": 5},
    )

    @field_validator("window_size")
    @classmethod
    def window_size_must_be_at_least_two(cls, v: int | None) -> int | None:
        if v is not None and v < 2:
            raise ValueError("window_size must be at least 2")
        return v

    @field_validator("overlap_size")
    @classmethod
    def overlap_size_must_be_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("overlap_size must be at least 1")
        return v


class SolverParams(BaseModel):
    """The set of solver parameters supported in the motile tracker.
    Used to build the UI as well as store parameters for runs.
    """

    model_config = ConfigDict(validate_assignment=True)

    max_children: int = Field(
        2,
        title="Max Children",
        description="The maximum number of object in time t+1 that can be linked to an item in time t.\nIf no division, set to 1.",
    )
    edge_selection_cost: float | None = Field(
        -20.0,
        title="Edge Selection",
        description=r"""Cost for selecting an edge. The more negative the value, the more edges will be selected.""",
    )
    node_selection_cost: float | None = Field(
        None,
        title="Node Selection",
        description=r"""Cost for selecting a node. The more negative the value, the more nodes will be selected.""",
        json_schema_extra={"ui_default": -20.0},
    )
    appear_cost: float | None = Field(
        30,
        title="Appear",
        description=r"""Cost for starting a new track. A higher value means fewer and longer selected tracks.""",
    )
    division_cost: float | None = Field(
        20,
        title="Division",
        description=r"""Cost for a track dividing. A higher value means fewer divisions.
If this cost is higher than the appear cost, tracks will likely never divide.""",
    )
    distance_cost: float | None = Field(
        1,
        title="Distance",
        description=r"""Use the distance between objects as a feature for selecting edges.
The value is multiplied by the edge distance to create a cost for selecting that edge.""",
    )
    attribute_weights: dict[str, float] = Field(
        default_factory=dict,
        title="Attribute Weights",
        description=r"""Weights for arbitrary node/edge features computed on the tracks
(e.g. iou, or anything else added via the Features menu). Each entry adds an
EdgeSelection cost: the feature's value is multiplied by the weight to produce a cost
for selecting that edge. Recommended to be negative for features where a bigger value
is a better match (e.g. IoU), positive otherwise.""",
    )

    @classmethod
    def available_attribute_keys(cls, tracks) -> list[str]:
        """List the node/edge feature keys on `tracks` that can be used as
        attribute_weights entries.

        Args:
            tracks: A funtracks Tracks (or subclass) instance to inspect.

        Returns:
            list[str]: Feature keys present on tracks.graph_solution, excluding
                internal/reserved keys (time, position, solution, track/lineage
                ids) that are not meaningful as a generic solving weight.
        """
        graph = tracks.graph_solution
        keys = set(graph.node_attr_keys()) | set(graph.edge_attr_keys())
        return sorted(keys - _RESERVED_FEATURE_KEYS)

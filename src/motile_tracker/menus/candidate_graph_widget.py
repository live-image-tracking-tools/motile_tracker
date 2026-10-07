from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import dask.array as da
import napari.layers
import numpy as np
from napari_track_edit.data_views.dims_utils import TracksDims
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible
from tqdm import tqdm

from .params_editor import CandidateGraphParamsEditor

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


class CandidateGraphWidget(QCollapsible):
    """Widget for selecting an input layer, editing CandidateGraphParams, and
    requesting a candidate graph build.

    Does not build the graph itself: emits build_candidate_graph with the
    input data, params, and scale, for the caller (MotileWidget) to run on a
    background thread.
    """

    build_candidate_graph = Signal(object, object, object)  # input_data, params, scale

    def __init__(self, viewer: napari.Viewer):
        """Args:
        viewer (napari.Viewer): The napari viewer to get the input
            segmentation/points layer from.
        """
        super().__init__(title="Candidate Graph")
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.viewer = viewer
        self.candidate_graph_params_widget = CandidateGraphParamsEditor()
        self.layer_selection_box: QComboBox

        build_btn = QPushButton("Build Candidate Graph")
        build_btn.clicked.connect(self.emit_build_candidate_graph)
        build_btn.setToolTip(
            "Build the candidate graph without solving. Might take minutes or "
            "longer for larger samples."
        )

        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.addWidget(self._labels_layer_widget())
        content_layout.addWidget(self.candidate_graph_params_widget)
        content_layout.addWidget(build_btn)
        content.setLayout(content_layout)
        self.addWidget(content)
        self.expand(animate=False)

        self.update_layer_selection()

    def update_labels_layers(self) -> None:
        """Update the layer selection box with the input layers in the viewer"""
        prev_selection = self.layer_selection_box.currentText()
        self.layer_selection_box.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels | napari.layers.Points):
                self.layer_selection_box.addItem(layer.name)
        self.layer_selection_box.setCurrentText(prev_selection)

    def update_layer_selection(self) -> None:
        """Update the rest of the UI when the selected layer is updated"""
        # the frame constraint has to be recomputed when the selection changes
        self._update_max_frames()

    def _update_max_frames(self) -> None:
        """Update the max frame constraint from viewer dims."""

        # The viewer may carry extra leading dims the input layer does not have, so time
        # is not necessarily world axis 0. The input layer will serve to build Tracks, so
        # occupies the trailing axes the same way the tracks will.
        layer = self.get_input_layer()
        if layer is None:
            time_axis = 0
        else:
            time_axis = TracksDims(self.viewer.dims.ndim, layer.ndim).time_axis
        max_frame = self.viewer.dims.range[time_axis].stop
        self.candidate_graph_params_widget.set_max_frames(int(max_frame))

    def _labels_layer_widget(self) -> QWidget:
        """Create the widget to select the input layer. Uses magicgui,
        but explicitly connects to the viewer layers events to keep it synced.

        Returns:
            QWidget: A dropdown select with all the labels layers in layers
                and a refresh button to sync with napari.
        """
        layer_group = QWidget()
        layer_layout = QHBoxLayout()
        layer_layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Input Layer:")
        layer_layout.addWidget(label)
        label.setToolTip("Select the labels layer you want to use for tracking")

        self.layer_selection_box = QComboBox()
        self.update_labels_layers()
        layers_events = self.viewer.layers.events
        layers_events.inserted.connect(self.update_labels_layers)
        layers_events.removed.connect(self.update_labels_layers)
        layers_events.reordered.connect(self.update_labels_layers)
        self.layer_selection_box.currentTextChanged.connect(self.update_layer_selection)
        self.viewer.dims.events.range.connect(self._update_max_frames)

        size_policy = self.layer_selection_box.sizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.MinimumExpanding)
        self.layer_selection_box.setSizePolicy(size_policy)
        layer_layout.addWidget(self.layer_selection_box)

        layer_group.setLayout(layer_layout)
        return layer_group

    def get_input_layer(self) -> napari.layers.Layer | None:
        """Get the input segmentation or points in current selection in the
        layer dropdown.

        Returns:
            napari.layers.Layer | None: The points or labels layer with the name
                that is selected, or None if no layer is selected.
        """
        layer_name = self.layer_selection_box.currentText()
        if layer_name is None or layer_name not in self.viewer.layers:
            return None
        return self.viewer.layers[layer_name]

    def get_input_data(self) -> tuple[np.ndarray, list[float]] | None:
        """Get the input segmentation or points array, and scale, from the
        currently selected layer.

        Returns:
            tuple[np.ndarray, list[float]] | None: (input_data, scale), or
                None if no layer is selected.
        """
        input_layer = self.get_input_layer()
        if input_layer is None:
            return None
        if isinstance(input_layer, napari.layers.Labels):
            data = input_layer.data[0] if input_layer.multiscale else input_layer.data
            if isinstance(data, da.core.Array):
                input_data = self._convert_da_to_np_array(data)
            else:
                input_data = np.asarray(data)
            ndim = input_data.ndim
            if ndim > 4:
                raise ValueError(f"Expected segmentation to be at most 4D, found {ndim}")
            elif ndim < 3:
                raise ValueError(
                    f"Expected segmentation to be at least 3D, found {ndim}"
                )
        elif isinstance(input_layer, napari.layers.Points):
            input_data = input_layer.data
        else:
            return None
        return input_data, input_layer.scale

    def _convert_da_to_np_array(self, dask_array: da.core.Array) -> np.ndarray:
        """Convert from dask array to in-memory array.

        Args:
            dask_array (da.core.Array): a dask array

        Returns:
            np.ndarray: data as an in-memory numpy array
        """
        stack_list = []
        for i in tqdm(
            range(dask_array.shape[0]),
            desc="Converting dask array to in-memory array",
        ):
            stack_list.append(dask_array[i].compute())
        return np.stack(stack_list, axis=0)

    def emit_build_candidate_graph(self) -> None:
        """Emit build_candidate_graph with the current input data, params,
        and scale. No-ops (with a warning) if no input layer is selected.
        """
        input_data_and_scale = self.get_input_data()
        if input_data_and_scale is None:
            logger.warning("No input layer selected")
            return
        input_data, scale = input_data_and_scale
        params = self.candidate_graph_params_widget.candidate_graph_params.copy()
        self.build_candidate_graph.emit(input_data, params, scale)

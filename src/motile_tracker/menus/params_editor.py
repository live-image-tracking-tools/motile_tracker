from functools import partial
from types import NoneType
from typing import get_args, get_origin

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from motile_tracker.backend import CandidateGraphParams, SolverParams, TilingParams

from .param_values import EditableParamValue

# Either params object a param row can be bound to.
ParamsModel = CandidateGraphParams | SolverParams | TilingParams


def _get_base_type(annotation: type) -> type:
    """Extract the base numeric type from a type annotation.

    For example:
        int -> int
        float -> float
        int | None -> int
        float | None -> float
    """
    # Check if it's a Union type (e.g., int | None)
    if get_origin(annotation) is not None:
        args = get_args(annotation)
        # Filter out NoneType and return the first remaining type
        for arg in args:
            if arg is not NoneType:
                return arg
    return annotation


class EditableParam(QWidget):
    def __init__(
        self,
        param_name: str,
        params: ParamsModel,
        negative: bool = False,
    ):
        """A widget for editing a parameter. Can be updated from
        the backend by calling update_from_params with a new params object of
        the same type. If changed in the UI, will emit a send_value signal
        which can be used to keep a params object in sync.

        Args:
            param_name (str): The name of the parameter to view in this UI row.
                Must correspond to one of the attributes of params.
            params (ParamsModel): The params object to use to initialize the
                view. Provides the title to display and the initial value.
            negative (bool, optional): Whether to allow negative values for
                this parameter. Defaults to False.
        """
        super().__init__()
        self.param_name = param_name
        field = params.model_fields[param_name]
        self.dtype = _get_base_type(field.annotation)
        self.title = field.title
        self.negative = negative
        self.param_label = self._param_label_widget()
        self.param_label.setToolTip(field.description)
        self.param_value = EditableParamValue(self.dtype, self.negative)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.param_label)
        layout.addWidget(self.param_value)
        self.setLayout(layout)
        self.setMinimumHeight(32)

        self.update_from_params(params)

    def _param_label_widget(self) -> QLabel:
        return QLabel(self.title)

    def update_from_params(self, params: ParamsModel):
        param_val = params.__getattribute__(self.param_name)
        if param_val is None:
            raise ValueError("Got None for required field {self.param_name}")
        else:
            self.param_value.update_value(param_val)


class OptionalEditableParam(EditableParam):
    def __init__(
        self,
        param_name: str,
        params: ParamsModel,
        negative: bool = False,
    ):
        """A widget for holding optional editable parameters. Adds a checkbox
        to the label, which toggles None-ness of the value.

        Args:
            param_name (str): _description_
            params (ParamsModel): _description_
            negative (bool, optional): _description_. Defaults to False.
        """
        # Get ui_default before calling super().__init__ (which calls update_from_params)
        field = params.model_fields[param_name]
        extra = field.json_schema_extra or {}
        self.ui_default = extra.get("ui_default", 0)

        super().__init__(param_name, params, negative)
        self.param_label.toggled.connect(self.toggle_enable)

    def _param_label_widget(self) -> QCheckBox:
        qlabel = QCheckBox(self.title)
        qlabel.setMinimumHeight(32)
        return qlabel

    def update_from_params(self, params: ParamsModel):
        param_val = params.__getattribute__(self.param_name)
        if param_val is None:
            self.param_label.setChecked(False)
            self.param_value.setEnabled(False)
            # Show ui_default in the disabled spinbox
            self.param_value.update_value(self.ui_default)
        else:
            self.param_label.setChecked(True)
            self.param_value.setEnabled(True)
            self.param_value.update_value(param_val)

    def toggle_enable(self, checked: bool):
        self.param_value.setEnabled(checked)
        value = self.param_value.get_value() if checked else None
        # force the parameter to say that the value has changed when we toggle
        self.param_value.valueChanged.emit(value)

    def toggle_visible(self, visible: bool):
        self.setVisible(visible)
        if visible and self.param_label.isChecked():
            value = self.param_value.get_value()
        else:
            value = None
        self.param_value.valueChanged.emit(value)


class CandidateGraphParamsEditor(QWidget):
    """Widget for editing CandidateGraphParams.
    Spinboxes will be created for each parameter and linked such that editing
    the value in the spinbox will change the corresponding parameter.
    Checkboxes will also be created for each optional parameter (group) and
    linked such that unchecking the box will update the parameter value to
    None, and checking will update the parameter to the current spinbox value.
    To update for a backend change to CandidateGraphParams, emit the
    new_params signal, which the spinboxes and checkboxes will connect to and
    use to update the UI and thus the stored params.
    """

    new_params = Signal(CandidateGraphParams)

    def __init__(self):
        super().__init__()
        self.candidate_graph_params = CandidateGraphParams()
        self.param_categories = {
            "hyperparams": ["max_edge_distance"],
            "single_window": [
                "single_window_start",
                "single_window_size",
            ],
        }
        self.single_window_start_row: OptionalEditableParam
        self.single_window_size_row: OptionalEditableParam

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(
            self._params_group("Hyperparameters", "hyperparams", negative=False)
        )
        main_layout.addWidget(
            self._params_group("Single Window", "single_window", negative=False)
        )
        self.setLayout(main_layout)

        # Set up cross-field validation for single window
        self._setup_single_window_constraints()

    def _params_group(self, title: str, param_category: str, negative: bool) -> QWidget:
        widget = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setSpacing(0)
        for param_name in self.param_categories[param_category]:
            field = self.candidate_graph_params.model_fields[param_name]
            param_cls = (
                OptionalEditableParam
                if issubclass(NoneType, field.annotation)
                else EditableParam
            )
            param_row = param_cls(
                param_name, self.candidate_graph_params, negative=negative
            )
            param_row.param_value.valueChanged.connect(
                partial(self.candidate_graph_params.__setattr__, param_name)
            )
            self.new_params.connect(param_row.update_from_params)
            if param_name == "single_window_start":
                self.single_window_start_row = param_row
            elif param_name == "single_window_size":
                self.single_window_size_row = param_row
            layout.addWidget(param_row)
        widget.setLayout(layout)
        return widget

    def _setup_single_window_constraints(self) -> None:
        """Set up validation constraints for single window fields."""
        # Set single_window_size minimum to 2
        self.single_window_size_row.param_value.setMinimum(2)

        # When single_window_start checkbox toggles, enable/disable single_window_size
        self.single_window_start_row.param_label.toggled.connect(
            self.single_window_size_row.setEnabled
        )

        # Initialize state: disable single_window_size if single_window_start is unchecked
        if not self.single_window_start_row.param_label.isChecked():
            self.single_window_size_row.setEnabled(False)

    def set_max_frames(self, max_frame: int) -> None:
        """Set the maximum frame index for single_window_start.

        Args:
            max_frame: The maximum valid frame index (typically num_frames - 1).
        """
        # single_window_start can be at most max_frame - 1 (need at least 2 frames)
        max_start = max(0, max_frame - 1)
        self.single_window_start_row.param_value.setMaximum(max_start)
        # Clamp current value if needed
        if self.single_window_start_row.param_value.value() > max_start:
            self.single_window_start_row.param_value.setValue(max_start)


class SolverParamsEditor(QWidget):
    """Widget for editing SolverParams.
    Spinboxes will be created for each parameter in SolverParams and linked such that
    editing the value in the spinbox will change the corresponding parameter.
    Checkboxes will also  be created for each optional parameter (group) and linked such
    that unchecking the box will update the parameter value to None, and checking will
    update the parameter to the current spinbox value.
    To update for a backend change to SolverParams, emit the new_params signal,
    which the spinboxes and checkboxes will connect to and use to update the
    UI and thus the stored solver params.
    """

    new_params = Signal(SolverParams)

    def __init__(self):
        super().__init__()
        self.solver_params = SolverParams()
        self.param_categories = {
            "hyperparams": ["max_children"],
            "constant_costs": [
                "edge_selection_cost",
                "node_selection_cost",
                "appear_cost",
                "division_cost",
            ],
            "position_costs": [
                "distance_cost",
            ],
        }

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(
            self._params_group("Hyperparameters", "hyperparams", negative=False)
        )
        main_layout.addWidget(
            self._params_group("Constant Costs", "constant_costs", negative=True)
        )
        main_layout.addWidget(
            self._params_group("Position Cost", "position_costs", negative=True)
        )
        self.setLayout(main_layout)

    def _params_group(self, title: str, param_category: str, negative: bool) -> QWidget:
        widget = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setSpacing(0)
        # layout.addWidget(QLabel(title))
        for param_name in self.param_categories[param_category]:
            field = self.solver_params.model_fields[param_name]
            param_cls = (
                OptionalEditableParam
                if issubclass(NoneType, field.annotation)
                else EditableParam
            )
            param_row = param_cls(param_name, self.solver_params, negative=negative)
            param_row.param_value.valueChanged.connect(
                partial(self.solver_params.__setattr__, param_name)
            )
            self.new_params.connect(param_row.update_from_params)
            layout.addWidget(param_row)
        widget.setLayout(layout)
        return widget


class AttributeWeightRow(QWidget):
    """A single labeled weight row for one entry of SolverParams.attribute_weights."""

    valueChanged = Signal(str, object)  # attribute key, new weight (float or None)

    def __init__(self, attribute: str, weight: float | None):
        super().__init__()
        self.attribute = attribute
        self.param_label = QCheckBox(attribute)
        self.param_label.setMinimumHeight(32)
        self.param_value = EditableParamValue(float, negative=True)
        self.param_label.toggled.connect(self._on_toggled)
        self.param_value.valueChanged.connect(
            lambda v: self.valueChanged.emit(self.attribute, v)
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.param_label)
        layout.addWidget(self.param_value)
        self.setLayout(layout)
        self.setMinimumHeight(32)

        self.set_weight(weight)

    def set_weight(self, weight: float | None) -> None:
        """Set this row's checked state and value without emitting valueChanged."""
        self.param_value.blockSignals(True)
        self.param_label.blockSignals(True)
        if weight is None:
            self.param_label.setChecked(False)
            self.param_value.setEnabled(False)
            self.param_value.update_value(0.0)
        else:
            self.param_label.setChecked(True)
            self.param_value.setEnabled(True)
            self.param_value.update_value(weight)
        self.param_value.blockSignals(False)
        self.param_label.blockSignals(False)

    def _on_toggled(self, checked: bool) -> None:
        self.param_value.setEnabled(checked)
        value = self.param_value.get_value() if checked else None
        self.valueChanged.emit(self.attribute, value)


class AttributeWeightsEditor(QWidget):
    """Widget for editing SolverParams.attribute_weights: a dynamic set of
    weight rows, one per node/edge feature available on the currently
    selected tracks.

    Does not know about TracksViewer directly: the caller passes in the
    Tracks to inspect via refresh_from_tracks (or None, to just clear rows).
    """

    new_params = Signal(SolverParams)
    refresh_requested = Signal()

    def __init__(self):
        super().__init__()
        self.solver_params = SolverParams()
        self.rows: dict[str, AttributeWeightRow] = {}

        self.group = QGroupBox("Attribute Weights")
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setSpacing(0)
        self.group.setLayout(self.rows_layout)

        refresh_btn = QPushButton("Refresh based on current features")
        refresh_btn.setToolTip(
            "Re-scan the currently selected tracks for node/edge features and "
            "show a weight row for each one."
        )
        refresh_btn.clicked.connect(self._emit_refresh_request)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.group)
        main_layout.addWidget(refresh_btn)
        self.setLayout(main_layout)

        self.new_params.connect(self._on_new_params)

    def _emit_refresh_request(self) -> None:
        self.refresh_requested.emit()

    def refresh_from_tracks(self, tracks) -> None:
        """Rebuild the weight rows from the feature keys available on
        `tracks` (or clear all rows if `tracks` is None).

        Existing weights for keys that are still available are preserved;
        weights for keys no longer available are dropped.
        """
        keys = SolverParams.available_attribute_keys(tracks) if tracks is not None else []
        self._set_keys(keys)

    def _set_keys(self, keys: list[str]) -> None:
        kept_weights = {
            key: self.solver_params.attribute_weights[key]
            for key in keys
            if key in self.solver_params.attribute_weights
        }
        self.solver_params.attribute_weights = kept_weights

        for row in self.rows.values():
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        self.rows = {}

        for key in keys:
            row = AttributeWeightRow(key, kept_weights.get(key))
            row.valueChanged.connect(self._on_row_value_changed)
            self.rows_layout.addWidget(row)
            self.rows[key] = row

    def _on_row_value_changed(self, attribute: str, value: float | None) -> None:
        weights = dict(self.solver_params.attribute_weights)
        if value is None:
            weights.pop(attribute, None)
        else:
            weights[attribute] = value
        self.solver_params.attribute_weights = weights

    def _on_new_params(self, params: SolverParams) -> None:
        self.solver_params = params
        self._set_keys(list(self.rows.keys()))


class TilingParamsEditor(QWidget):
    """Widget for editing TilingParams: chunked/tiled solving over the full
    time range. Spinboxes/checkboxes are created and linked the same way as
    SolverParamsEditor.
    """

    new_params = Signal(TilingParams)

    def __init__(self):
        super().__init__()
        self.tiling_params = TilingParams()
        self.param_categories = {
            "chunked_solving": [
                "window_size",
                "overlap_size",
            ],
        }
        self.window_size_row: OptionalEditableParam
        self.overlap_size_row: OptionalEditableParam

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(
            self._params_group("Chunked Solving", "chunked_solving", negative=False)
        )
        self.setLayout(main_layout)

        # Set up cross-field validation for chunked solving
        self._setup_chunking_constraints()

    def _params_group(self, title: str, param_category: str, negative: bool) -> QWidget:
        widget = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setSpacing(0)
        for param_name in self.param_categories[param_category]:
            field = self.tiling_params.model_fields[param_name]
            param_cls = (
                OptionalEditableParam
                if issubclass(NoneType, field.annotation)
                else EditableParam
            )
            param_row = param_cls(param_name, self.tiling_params, negative=negative)
            param_row.param_value.valueChanged.connect(
                partial(self.tiling_params.__setattr__, param_name)
            )
            self.new_params.connect(param_row.update_from_params)
            if param_name == "window_size":
                self.window_size_row = param_row
            elif param_name == "overlap_size":
                self.overlap_size_row = param_row
            layout.addWidget(param_row)
        widget.setLayout(layout)
        return widget

    def _setup_chunking_constraints(self) -> None:
        """Set up validation constraints for chunked solving fields."""
        # Set window_size minimum to 2
        self.window_size_row.param_value.setMinimum(2)

        # Set overlap_size minimum to 1
        self.overlap_size_row.param_value.setMinimum(1)

        # When window_size value changes, update overlap_size max
        self.window_size_row.param_value.valueChanged.connect(
            self._update_overlap_constraints
        )

        # When window_size checkbox toggles, enable/disable overlap_size
        self.window_size_row.param_label.toggled.connect(self.overlap_size_row.setEnabled)

        # Initialize state: disable overlap_size if window_size is unchecked
        if not self.window_size_row.param_label.isChecked():
            self.overlap_size_row.setEnabled(False)

    def _update_overlap_constraints(self, window_size: int | None) -> None:
        """Update overlap_size spinbox maximum based on window_size."""
        if window_size is not None and window_size > 1:
            self.overlap_size_row.param_value.setMaximum(window_size - 1)
            # Clamp current value if needed
            if self.overlap_size_row.param_value.value() >= window_size:
                self.overlap_size_row.param_value.setValue(window_size - 1)

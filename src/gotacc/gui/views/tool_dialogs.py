from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
    QVBoxLayout,
    QWidget,
)

from gotacc.interfaces.policies import POLICY_REGISTRY

try:
    from .ui_dialog_algorithm_detail import Ui_AlgorithmDetailDialog
    from .ui_dialog_bounds_tools import Ui_BoundsToolsDialog
    from .ui_dialog_pv_library_selector import Ui_PVLibrarySelectorDialog
    from .ui_dialog_pv_monitor import Ui_PVMonitorDialog
except ImportError:  # pragma: no cover
    from ui_dialog_algorithm_detail import Ui_AlgorithmDetailDialog
    from ui_dialog_bounds_tools import Ui_BoundsToolsDialog
    from ui_dialog_pv_library_selector import Ui_PVLibrarySelectorDialog
    from ui_dialog_pv_monitor import Ui_PVMonitorDialog

try:
    from ..services.pv_library import PVLibraryItem
    from ..services.task_service import TaskService
except ImportError:  # pragma: no cover
    CURRENT_DIR = Path(__file__).resolve().parent
    GUI_ROOT = CURRENT_DIR.parent
    for path in (GUI_ROOT, GUI_ROOT / "services"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from pv_library import PVLibraryItem
    from task_service import TaskService


class MachineWriteConfirmationDialog(QDialog):
    ONLINE_START = "online_start"
    EXACT_VALUES = "exact_values"

    def __init__(
        self,
        task: dict,
        *,
        mode: str,
        action_title: str,
        values: dict[str, float] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if mode not in {self.ONLINE_START, self.EXACT_VALUES}:
            raise ValueError(f"Unsupported machine write confirmation mode: {mode!r}")

        self.mode = mode
        self.task = task
        self.task_cfg = TaskService.build_task_config(task)
        if self.task_cfg.backend.type != "epics":
            raise ValueError("Machine write confirmation requires an Online EPICS task.")

        self.setModal(True)
        self.setWindowTitle(action_title)
        self.resize(920, 480)

        root = QVBoxLayout(self)
        self.label_warning = QLabel(self)
        self.label_warning.setObjectName("machineWriteWarning")
        self.label_warning.setWordWrap(True)
        self.label_warning.setText(
            "This action can write accelerator setpoints. Verify the task, PV mapping, "
            "limits, and restoration settings before continuing."
        )
        root.addWidget(self.label_warning)

        summary_box = QGroupBox("Machine Write Authorization", self)
        summary_layout = QFormLayout(summary_box)
        machine = task.get("machine", {}) or {}
        restore_text = "Enabled" if bool(machine.get("restore_on_abort", True)) else "Disabled"
        readback_text = "Enabled" if bool(machine.get("readback_check", False)) else "Disabled"
        summary_rows = [
            ("Task", self.task_cfg.meta.name),
            ("Machine", self.task_cfg.meta.machine or "epics-machine"),
            ("Algorithm", self.task_cfg.optimizer.name),
            ("Evaluation budget", str(int(task.get("max_evaluations", 0) or 0))),
            ("Restore on abort", restore_text),
            ("Readback check", readback_text),
            (
                "Set / sample interval",
                f"{float(machine.get('set_interval', 1.0)):g}s / "
                f"{float(machine.get('sample_interval', 0.2)):g}s",
            ),
        ]
        for label, value in summary_rows:
            value_label = QLabel(str(value), summary_box)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            summary_layout.addRow(label, value_label)
        root.addWidget(summary_box)

        self.table = QTableWidget(self)
        self.table.setObjectName("machineWriteTable")
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        root.addWidget(self.table, 1)

        self._populate_rows(values or {})

        budget = int(task.get("max_evaluations", 0) or 0)
        self.label_notice = QLabel(self)
        self.label_notice.setWordWrap(True)
        if mode == self.ONLINE_START:
            self.label_notice.setText(
                f"The optimizer may perform up to {budget} automatic setpoint writes during this run."
            )
        else:
            self.label_notice.setText(
                "The values shown above will be written once after confirmation."
            )
        root.addWidget(self.label_notice)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.accept_button = self.button_box.button(QDialogButtonBox.Ok)
        self.accept_button.setText(
            "Start Online Run" if mode == self.ONLINE_START else action_title
        )
        self.accept_button.setProperty("primary", True)
        self.accept_button.style().unpolish(self.accept_button)
        self.accept_button.style().polish(self.accept_button)
        self.accept_button.setDefault(False)
        self.accept_button.setAutoDefault(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        for button in self.button_box.buttons():
            button.setAutoDefault(False)
            button.setDefault(False)

    def keyPressEvent(self, event) -> None:
        if event.key() in {Qt.Key_Enter, Qt.Key_Return}:
            event.ignore()
            return
        super().keyPressEvent(event)

    def _populate_rows(self, values: dict[str, float]) -> None:
        kwargs = self.task_cfg.backend.kwargs
        variable_names = list(kwargs.get("variable_names", []))
        setpoint_pvs = list(kwargs.get("knobs_pvnames", []))
        readback_pvs = list(kwargs.get("knob_readback_pvnames", []))
        bounds = list(self.task_cfg.backend.bounds)
        variables = TaskService._enabled_rows(self.task.get("variables", []))

        count = len(variable_names)
        if len(setpoint_pvs) != count or len(bounds) != count:
            raise ValueError("Online task variable, setpoint PV, and bounds counts do not match.")
        if readback_pvs and len(readback_pvs) != count:
            raise ValueError("Online task variable and readback PV counts do not match.")

        if self.mode == self.ONLINE_START:
            headers = ["Variable", "Setpoint PV", "Readback PV", "Lower", "Upper", "Initial"]
        else:
            headers = ["Variable", "Setpoint PV", "Readback PV", "Value"]
            missing = [name for name in variable_names if name not in values]
            if missing:
                raise ValueError(f"Writable values are missing variable(s): {', '.join(missing)}")

        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(count)
        self.table.setMinimumHeight(100)
        self.table.setMaximumHeight(min(330, 72 + 30 * max(1, count)))
        for row, name in enumerate(variable_names):
            readback = readback_pvs[row] if readback_pvs else "--"
            if self.mode == self.ONLINE_START:
                initial = variables[row].get("Initial", "") if row < len(variables) else ""
                row_values = [
                    name,
                    setpoint_pvs[row],
                    readback,
                    bounds[row][0],
                    bounds[row][1],
                    initial,
                ]
            else:
                row_values = [name, setpoint_pvs[row], readback, values[name]]
            for column, value in enumerate(row_values):
                item = QTableWidgetItem(str(value))
                if column == 0 or column >= 3:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)

        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        self.table.resizeColumnsToContents()


class PVLibrarySelectorDialog(QDialog):
    def __init__(
        self,
        entries: list[PVLibraryItem],
        *,
        title: str,
        intro_text: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.ui = Ui_PVLibrarySelectorDialog()
        self.ui.setupUi(self)
        self.setWindowTitle(title)
        self.ui.label_intro.setText(intro_text)

        self._all_entries = list(entries)
        self._visible_entries = list(entries)

        table = self.ui.tableWidget_library
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.MultiSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        for idx in range(table.columnCount() - 1):
            header.setSectionResizeMode(idx, header.Stretch)

        self.ui.lineEdit_filter.textChanged.connect(self._refresh_rows)
        self.ui.buttonBox.accepted.connect(self._accept_if_any)
        self.ui.buttonBox.rejected.connect(self.reject)

        self._refresh_rows()

    def _refresh_rows(self) -> None:
        query = self.ui.lineEdit_filter.text().strip().lower()

        def matches(entry: PVLibraryItem) -> bool:
            if not query:
                return True
            haystack = "\n".join(
                [
                    entry.name.lower(),
                    entry.pv_name.lower(),
                    entry.readback.lower(),
                    entry.group.lower(),
                    entry.note.lower(),
                ]
            )
            return query in haystack

        self._visible_entries = [entry for entry in self._all_entries if matches(entry)]
        table = self.ui.tableWidget_library
        table.setRowCount(len(self._visible_entries))
        for row, entry in enumerate(self._visible_entries):
            values = [entry.name, entry.pv_name, entry.readback, entry.group, entry.note]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                table.setItem(row, col, item)
        self.ui.label_summary.setText(
            f"Showing {len(self._visible_entries)} of {len(self._all_entries)} available PV rows."
        )

    def _accept_if_any(self) -> None:
        if not self.selected_entries():
            QMessageBox.information(self, self.windowTitle(), "Select at least one PV row first.")
            return
        self.accept()

    def selected_entries(self) -> list[PVLibraryItem]:
        selection_model = self.ui.tableWidget_library.selectionModel()
        if selection_model is None:
            return []
        rows = sorted({index.row() for index in selection_model.selectedRows()})
        return [self._visible_entries[row] for row in rows if 0 <= row < len(self._visible_entries)]


class PVMappingSelectorDialog(QDialog):
    ROLE_TITLES = {
        "knob": "Knobs",
        "objective": "Objectives",
        "constraint": "Constraints",
    }

    def __init__(
        self,
        *,
        knob_entries: list[PVLibraryItem],
        objective_entries: list[PVLibraryItem],
        constraint_entries: list[PVLibraryItem],
        current_keys: dict[str, set[str]] | None = None,
        source_label: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select PV Mapping")
        self.resize(980, 660)

        self._entries = {
            "knob": list(knob_entries),
            "objective": list(objective_entries),
            "constraint": list(constraint_entries),
        }
        self._current_keys = current_keys or {}
        self._tables: dict[str, QTableWidget] = {}

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Select PV rows for each role, then apply them into the Machine PV Mapping table. "
            "Leaving a role empty clears that role from the mapping.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        if source_label:
            source = QLabel(f"Library: {source_label}", self)
            source.setWordWrap(True)
            layout.addWidget(source)

        self.lineEdit_filter = QLineEdit(self)
        self.lineEdit_filter.setObjectName("lineEdit_pvMappingFilter")
        self.lineEdit_filter.setClearButtonEnabled(True)
        self.lineEdit_filter.setPlaceholderText(
            "Search by name, PV, readback, group or note..."
        )
        self.lineEdit_filter.setToolTip(
            "Filters Knobs, Objectives and Constraints without clearing selected rows."
        )
        self.lineEdit_filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.lineEdit_filter)

        tabs = QTabWidget(self)
        for role in ("knob", "objective", "constraint"):
            tabs.addTab(self._build_role_tab(role), self.ROLE_TITLES[role])
        layout.addWidget(tabs)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self._accept_with_confirmation)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

    def _build_role_tab(self, role: str) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        table = QTableWidget(tab)
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Name", "PV Name", "Readback", "Group", "Note"])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.MultiSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        self._tables[role] = table
        self._populate_table(role)
        return tab

    def _populate_table(self, role: str) -> None:
        table = self._tables[role]
        entries = self._entries[role]
        table.setRowCount(len(entries))
        current_keys = self._current_keys.get(role, set())
        for row, entry in enumerate(entries):
            values = [entry.name, entry.pv_name, entry.readback, entry.group, entry.note]
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(str(value)))
            if self._entry_matches_current(entry, current_keys):
                table.setRangeSelected(
                    QTableWidgetSelectionRange(row, 0, row, table.columnCount() - 1),
                    True,
                )
        table.resizeColumnsToContents()

    def _apply_filter(self, text: str) -> None:
        tokens = str(text).strip().casefold().split()
        for role, table in self._tables.items():
            for row, entry in enumerate(self._entries[role]):
                searchable = "\n".join(
                    (
                        entry.name,
                        entry.pv_name,
                        entry.readback,
                        entry.group,
                        entry.note,
                    )
                ).casefold()
                table.setRowHidden(row, not all(token in searchable for token in tokens))

    @staticmethod
    def _entry_matches_current(entry: PVLibraryItem, current_keys: set[str]) -> bool:
        return (
            str(entry.name).strip().lower() in current_keys
            or str(entry.pv_name).strip().lower() in current_keys
        )

    def selected_entries(self, role: str) -> list[PVLibraryItem]:
        table = self._tables.get(role)
        if table is None or table.selectionModel() is None:
            return []
        entries = self._entries.get(role, [])
        rows = sorted({index.row() for index in table.selectionModel().selectedRows()})
        return [entries[row] for row in rows if 0 <= row < len(entries)]

    def selected_entries_by_role(self) -> dict[str, list[PVLibraryItem]]:
        return {
            role: self.selected_entries(role)
            for role in ("knob", "objective", "constraint")
        }

    def _accept_with_confirmation(self) -> None:
        selected = self.selected_entries_by_role()
        if any(selected.values()):
            self.accept()
            return
        answer = QMessageBox.question(
            self,
            self.windowTitle(),
            "No PV rows are selected. Clear all PV Mapping roles?",
        )
        if answer == QMessageBox.Yes:
            self.accept()


class BoundsToolsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.ui = Ui_BoundsToolsDialog()
        self.ui.setupUi(self)
        self.setModal(True)
        self.ui.gridLayout_boundsTools.setColumnStretch(1, 1)
        self.ui.gridLayout_boundsTools.setColumnStretch(3, 1)
        table = self.ui.tableWidget_boundsPreview
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setMinimumHeight(150)
        self.ui.buttonBox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.ui.pushButton_applyBounds.setProperty("primary", True)
        close_button = self.ui.buttonBox.button(QDialogButtonBox.Close)
        for button in (
            self.ui.pushButton_previewBounds,
            self.ui.pushButton_applyBounds,
            close_button,
        ):
            button.setProperty("inlineAction", True)
            button.setFixedWidth(112)
            button.setFixedHeight(28)
        self.ui.buttonBox.rejected.connect(self.reject)


class AlgorithmDetailDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.ui = Ui_AlgorithmDetailDialog()
        self.ui.setupUi(self)
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)


class PVMonitorDialog(QDialog):
    def __init__(
        self,
        task_provider: Callable[[], dict],
        *,
        timeout_provider: Callable[[], float] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.ui = Ui_PVMonitorDialog()
        self.ui.setupUi(self)

        self._task_provider = task_provider
        self._timeout_provider = timeout_provider or (lambda: 1.0)
        self._rows: list[dict[str, str]] = []

        self.ui.buttonBox.rejected.connect(self.reject)
        self.ui.pushButton_refresh.clicked.connect(self.refresh_rows)
        self.ui.pushButton_readSelected.clicked.connect(self.read_selected)
        self.ui.pushButton_readAll.clicked.connect(self.read_all)

        self.refresh_rows()

    def _append_log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.ui.plainTextEdit_log.appendPlainText(f"[{ts}] {message}")

    def _set_status(self, text: str) -> None:
        self.ui.label_status.setText(text)

    def refresh_rows(self) -> None:
        task = self._task_provider()
        rows = TaskService.extract_machine_pvs(task)
        self._rows = rows

        table = self.ui.tableWidget_pvs
        table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            values = [
                row.get("role", ""),
                row.get("name", ""),
                row.get("pvname", ""),
                "--",
                "Idle",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col in {0, 1, 4}:
                    item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_idx, col, item)

        if not rows:
            self._set_status("No online EPICS PVs are configured in the current task.")
        else:
            self._set_status(f"Loaded {len(rows)} configured PVs from the current task.")
        self._append_log("PV list refreshed.")

    def _read_indices(self, indices: list[int]) -> None:
        if not indices:
            QMessageBox.information(self, "PV Monitor", "Select at least one PV row first.")
            return
        if not self._rows:
            QMessageBox.information(self, "PV Monitor", "No PVs are configured for the current task.")
            return

        try:
            from epics import caget
        except ImportError as exc:
            self._set_status("pyepics is not installed or not available in this environment.")
            QMessageBox.critical(self, "PV Monitor", str(exc))
            return

        timeout = float(self._timeout_provider())
        table = self.ui.tableWidget_pvs
        success = 0
        for idx in indices:
            row = self._rows[idx]
            pvname = row["pvname"]
            try:
                value = caget(pvname, timeout=timeout)
                status = "OK" if value is not None else "No Data"
                if value is not None:
                    success += 1
            except Exception as exc:  # pragma: no cover - runtime read protection
                value = str(exc)
                status = "Error"
            table.setItem(idx, 3, QTableWidgetItem(str(value)))
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(idx, 4, status_item)
            self._append_log(f"{pvname} -> {value} ({status})")

        self._set_status(f"Read {success}/{len(indices)} PVs successfully.")

    def read_selected(self) -> None:
        row = self.ui.tableWidget_pvs.currentRow()
        if row < 0:
            QMessageBox.information(self, "PV Monitor", "Select one PV row first.")
            return
        self._read_indices([row])

    def read_all(self) -> None:
        self._read_indices(list(range(len(self._rows))))


class SampleGuardRuleEditorDialog(QDialog):
    """Structured editor for declarative objective/constraint sample guards."""

    METRICS = ("mean_abs", "max_abs", "peak_to_peak", "mean", "std", "reduced")
    OPERATORS = ("gt", "ge", "lt", "le", "eq", "ne")

    def __init__(
        self,
        *,
        kind: str,
        target_names: list[str] | tuple[str, ...],
        policy_name: str = "sample_guard",
        kwargs: dict | None = None,
        preset_name: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if kind not in {"objective", "constraint"}:
            raise ValueError("Rule Editor kind must be objective or constraint")
        self.kind = kind
        self.target_names = [str(name).strip() for name in target_names if str(name).strip()]
        self._loading = False
        self.setWindowTitle(f"{kind.title()} Rule Editor")
        self.resize(760, 590)

        root = QVBoxLayout(self)
        intro = QLabel(
            "Build a reusable sample rule from fields. No Python expressions or raw JSON are required.",
            self,
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self.comboBox_preset = QComboBox(self)
        self.comboBox_preset.addItem("Custom Rule", "")
        for name in POLICY_REGISTRY.preset_names(kind, gui_only=True):
            preset = POLICY_REGISTRY.resolve_preset(kind, name)
            self.comboBox_preset.addItem(preset.display_name, preset.name)
        form.addRow("Preset", self.comboBox_preset)

        self.comboBox_target = QComboBox(self)
        self.comboBox_target.setEditable(True)
        self.comboBox_target.addItems(self.target_names)
        form.addRow("Target", self.comboBox_target)

        self.comboBox_match = QComboBox(self)
        self.comboBox_match.addItems(["any", "all"])
        form.addRow("Match conditions", self.comboBox_match)
        root.addLayout(form)

        condition_group = QGroupBox("Conditions", self)
        condition_layout = QVBoxLayout(condition_group)
        self.tableWidget_conditions = QTableWidget(0, 4, condition_group)
        self.tableWidget_conditions.setHorizontalHeaderLabels(
            ["Metric", "Operator", "Value", "Tolerance"]
        )
        self.tableWidget_conditions.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget_conditions.setSelectionBehavior(QAbstractItemView.SelectRows)
        condition_layout.addWidget(self.tableWidget_conditions)
        condition_buttons = QHBoxLayout()
        self.pushButton_addCondition = QPushButton("Add Condition", condition_group)
        self.pushButton_removeCondition = QPushButton("Remove Selected", condition_group)
        condition_buttons.addWidget(self.pushButton_addCondition)
        condition_buttons.addWidget(self.pushButton_removeCondition)
        condition_buttons.addStretch(1)
        condition_layout.addLayout(condition_buttons)
        root.addWidget(condition_group)

        action_group = QGroupBox("Action", self)
        action_form = QFormLayout(action_group)
        self.comboBox_action = QComboBox(action_group)
        self.comboBox_action.addItems(
            ["replace", "add_offset"] if kind == "objective" else ["replace", "violate_bound"]
        )
        action_form.addRow("Type", self.comboBox_action)
        self.doubleSpinBox_actionValue = self._number_box(action_group)
        action_form.addRow("Value", self.doubleSpinBox_actionValue)
        self.doubleSpinBox_deltaRatio = self._nonnegative_box(action_group, 0.1)
        self.doubleSpinBox_deltaMin = self._nonnegative_box(action_group, 1e-6)
        self.doubleSpinBox_scaleFloor = self._nonnegative_box(action_group, 1.0)
        action_form.addRow("Delta ratio", self.doubleSpinBox_deltaRatio)
        action_form.addRow("Minimum delta", self.doubleSpinBox_deltaMin)
        action_form.addRow("Scale floor", self.doubleSpinBox_scaleFloor)
        root.addWidget(action_group)

        self.label_summary = QLabel(self)
        self.label_summary.setWordWrap(True)
        root.addWidget(self.label_summary)
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        root.addWidget(self.buttonBox)

        self.comboBox_preset.currentIndexChanged.connect(self._on_preset_changed)
        self.comboBox_target.currentTextChanged.connect(self._on_rule_changed)
        self.comboBox_match.currentTextChanged.connect(self._on_rule_changed)
        self.comboBox_action.currentTextChanged.connect(self._on_action_changed)
        self.pushButton_addCondition.clicked.connect(self._add_custom_condition)
        self.pushButton_removeCondition.clicked.connect(self._remove_selected_conditions)
        self.buttonBox.accepted.connect(self._accept_if_valid)
        self.buttonBox.rejected.connect(self.reject)
        for box in (
            self.doubleSpinBox_actionValue,
            self.doubleSpinBox_deltaRatio,
            self.doubleSpinBox_deltaMin,
            self.doubleSpinBox_scaleFloor,
        ):
            box.valueChanged.connect(self._on_rule_changed)

        initial_preset = preset_name or self._legacy_preset_name(policy_name)
        initial_kwargs = dict(kwargs or {})
        if not initial_kwargs and initial_preset:
            initial_kwargs = POLICY_REGISTRY.expand_preset(kind, initial_preset)["kwargs"]
        if not initial_kwargs:
            initial_kwargs = POLICY_REGISTRY.resolve(kind, "sample_guard").defaults()
        self._load_rule(initial_kwargs, preset_name=initial_preset)

    @staticmethod
    def _number_box(parent) -> QDoubleSpinBox:
        box = QDoubleSpinBox(parent)
        box.setDecimals(12)
        box.setRange(-1e15, 1e15)
        box.setSingleStep(0.1)
        return box

    @classmethod
    def _nonnegative_box(cls, parent, value: float) -> QDoubleSpinBox:
        box = cls._number_box(parent)
        box.setRange(0.0, 1e15)
        box.setValue(value)
        return box

    def _legacy_preset_name(self, policy_name: str) -> str | None:
        name = str(policy_name or "").strip().lower()
        return name if name in POLICY_REGISTRY.preset_names(self.kind) else None

    def _set_condition_row(self, row: int, condition: dict) -> None:
        self.tableWidget_conditions.insertRow(row)
        metric = QComboBox(self.tableWidget_conditions)
        metric.addItems(list(self.METRICS))
        metric.setCurrentText(str(condition.get("metric", "mean_abs")))
        operator = QComboBox(self.tableWidget_conditions)
        operator.addItems(list(self.OPERATORS))
        operator.setCurrentText(str(condition.get("operator", "gt")))
        value = self._number_box(self.tableWidget_conditions)
        value.setValue(float(condition.get("value", 0.0)))
        atol = self._nonnegative_box(
            self.tableWidget_conditions, float(condition.get("atol", 0.0))
        )
        self.tableWidget_conditions.setCellWidget(row, 0, metric)
        self.tableWidget_conditions.setCellWidget(row, 1, operator)
        self.tableWidget_conditions.setCellWidget(row, 2, value)
        self.tableWidget_conditions.setCellWidget(row, 3, atol)
        metric.currentTextChanged.connect(self._on_rule_changed)
        operator.currentTextChanged.connect(self._on_rule_changed)
        value.valueChanged.connect(self._on_rule_changed)
        atol.valueChanged.connect(self._on_rule_changed)

    def _load_rule(self, kwargs: dict, *, preset_name: str | None = None) -> None:
        self._loading = True
        try:
            target = kwargs.get("target")
            target_col = int(kwargs.get("target_col", 0) or 0)
            if target is None and 0 <= target_col < len(self.target_names):
                target = self.target_names[target_col]
            self.comboBox_target.setCurrentText(str(target or ""))
            self.comboBox_match.setCurrentText(str(kwargs.get("match", "any")))
            self.tableWidget_conditions.setRowCount(0)
            for condition in kwargs.get("conditions", []):
                self._set_condition_row(self.tableWidget_conditions.rowCount(), dict(condition))
            action = dict(kwargs.get("action", {}))
            self.comboBox_action.setCurrentText(str(action.get("type", "replace")))
            self.doubleSpinBox_actionValue.setValue(float(action.get("value", 0.0)))
            self.doubleSpinBox_deltaRatio.setValue(float(action.get("delta_ratio", 0.1)))
            self.doubleSpinBox_deltaMin.setValue(float(action.get("delta_min", 1e-6)))
            self.doubleSpinBox_scaleFloor.setValue(float(action.get("scale_floor", 1.0)))
            index = self.comboBox_preset.findData(preset_name or "")
            self.comboBox_preset.setCurrentIndex(max(0, index))
        finally:
            self._loading = False
        self._update_action_fields()
        self._refresh_summary()

    def _on_preset_changed(self) -> None:
        if self._loading:
            return
        name = str(self.comboBox_preset.currentData() or "")
        if name:
            self._load_rule(POLICY_REGISTRY.expand_preset(self.kind, name)["kwargs"], preset_name=name)

    def _on_rule_changed(self, *_args) -> None:
        if self._loading:
            return
        self._loading = True
        self.comboBox_preset.setCurrentIndex(0)
        self._loading = False
        self._refresh_summary()

    def _on_action_changed(self, *_args) -> None:
        self._update_action_fields()
        self._on_rule_changed()

    def _update_action_fields(self) -> None:
        violate = self.comboBox_action.currentText() == "violate_bound"
        self.doubleSpinBox_actionValue.setVisible(not violate)
        value_label = self.doubleSpinBox_actionValue.parent().layout().labelForField(
            self.doubleSpinBox_actionValue
        )
        if value_label is not None:
            value_label.setVisible(not violate)
        for box in (
            self.doubleSpinBox_deltaRatio,
            self.doubleSpinBox_deltaMin,
            self.doubleSpinBox_scaleFloor,
        ):
            box.setVisible(violate)
            label = box.parent().layout().labelForField(box)
            if label is not None:
                label.setVisible(violate)

    def _add_custom_condition(self) -> None:
        self._set_condition_row(
            self.tableWidget_conditions.rowCount(),
            {"metric": "mean_abs", "operator": "gt", "value": 0.0},
        )
        self._on_rule_changed()

    def _remove_selected_conditions(self) -> None:
        rows = sorted(
            {index.row() for index in self.tableWidget_conditions.selectionModel().selectedRows()},
            reverse=True,
        )
        for row in rows:
            self.tableWidget_conditions.removeRow(row)
        self._on_rule_changed()

    def rule_state(self) -> dict:
        target = self.comboBox_target.currentText().strip()
        target_col = self.target_names.index(target) if target in self.target_names else 0
        conditions = []
        for row in range(self.tableWidget_conditions.rowCount()):
            condition = {
                "metric": self.tableWidget_conditions.cellWidget(row, 0).currentText(),
                "operator": self.tableWidget_conditions.cellWidget(row, 1).currentText(),
                "value": self.tableWidget_conditions.cellWidget(row, 2).value(),
            }
            atol = self.tableWidget_conditions.cellWidget(row, 3).value()
            if atol:
                condition["atol"] = atol
            conditions.append(condition)
        action_type = self.comboBox_action.currentText()
        if action_type == "violate_bound":
            action = {
                "type": action_type,
                "delta_ratio": self.doubleSpinBox_deltaRatio.value(),
                "delta_min": self.doubleSpinBox_deltaMin.value(),
                "scale_floor": self.doubleSpinBox_scaleFloor.value(),
            }
        else:
            action = {"type": action_type, "value": self.doubleSpinBox_actionValue.value()}
        return {
            "preset": str(self.comboBox_preset.currentData() or "custom"),
            "name": "sample_guard",
            "kwargs": {
                "target": target or None,
                "target_col": target_col,
                "conditions": conditions,
                "match": self.comboBox_match.currentText(),
                "action": action,
            },
        }

    def _refresh_summary(self) -> None:
        state = self.rule_state()
        count = len(state["kwargs"]["conditions"])
        target = state["kwargs"]["target"] or f"column {state['kwargs']['target_col']}"
        self.label_summary.setText(
            f"{target}: match {state['kwargs']['match']} of {count} condition(s), then "
            f"{state['kwargs']['action']['type']}."
        )

    def _accept_if_valid(self) -> None:
        try:
            state = self.rule_state()
            POLICY_REGISTRY.validate(self.kind, state["name"], state["kwargs"])
        except Exception as exc:
            QMessageBox.critical(self, "Rule Editor", str(exc))
            return
        self.accept()

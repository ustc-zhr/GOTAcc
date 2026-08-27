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


class PolicyTemplatePickerDialog(QDialog):
    """Preset-first policy entry point for routine machine operation."""

    def __init__(
        self,
        *,
        kind: str,
        target: str,
        pv_name: str,
        custom_presets: list[dict] | None = None,
        constraint_bound_ready: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.kind = str(kind).strip().lower()
        self.target = str(target).strip()
        self.pv_name = str(pv_name).strip()
        self.constraint_bound_ready = bool(constraint_bound_ready)
        self.setWindowTitle("Add Policy")
        self.resize(760, 430)

        root = QVBoxLayout(self)
        heading = QLabel(
            f"{self.kind.title()} · {self.target} · {self.pv_name or 'PV not assigned'}",
            self,
        )
        heading.setObjectName("policyTemplateTarget")
        heading.setWordWrap(True)
        root.addWidget(heading)
        intro = QLabel(
            "Choose a Policy Template to use its tested defaults. Choose Custom Policy "
            "only when the signal needs advanced conditions.",
            self,
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.tableWidget_templates = QTableWidget(0, 2, self)
        self.tableWidget_templates.setHorizontalHeaderLabels(
            ["Policy Template", "What it does"]
        )
        self.tableWidget_templates.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget_templates.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tableWidget_templates.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget_templates.verticalHeader().setVisible(False)
        header = self.tableWidget_templates.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.tableWidget_templates.setColumnWidth(0, 190)
        root.addWidget(self.tableWidget_templates, 1)

        templates: list[dict] = []
        for preset_id in POLICY_REGISTRY.preset_names(self.kind, gui_only=True):
            preset = POLICY_REGISTRY.resolve_preset(self.kind, preset_id)
            templates.append(
                {
                    "id": preset.name,
                    "name": preset.display_name,
                    "description": preset.description,
                    "policy": POLICY_REGISTRY.expand_preset(self.kind, preset.name),
                    "custom_rule": False,
                }
            )
        for preset in custom_presets or []:
            if str(preset.get("kind", "")).strip().lower() != self.kind:
                continue
            preset_id = str(preset.get("id", "")).strip()
            policy = preset.get("policy", {}) or {}
            if not preset_id or not isinstance(policy, dict):
                continue
            templates.append(
                {
                    "id": preset_id,
                    "name": str(preset.get("name", preset_id)),
                    "description": str(preset.get("description", "")).strip()
                    or "Use this machine-specific saved policy.",
                    "policy": policy,
                    "custom_rule": False,
                }
            )
        templates.append(
            {
                "id": "custom",
                "name": "Custom Policy",
                "description": (
                    "Open the Policy Editor to define conditions and an action."
                ),
                "policy": None,
                "custom_rule": True,
            }
        )
        for template in templates:
            row = self.tableWidget_templates.rowCount()
            self.tableWidget_templates.insertRow(row)
            name_item = QTableWidgetItem(str(template["name"]))
            name_item.setData(Qt.UserRole, template)
            self.tableWidget_templates.setItem(row, 0, name_item)
            self.tableWidget_templates.setItem(
                row, 1, QTableWidgetItem(str(template["description"]))
            )
            self.tableWidget_templates.setRowHeight(row, 48)

        self.label_setup = QLabel(self)
        self.label_setup.setProperty("tone", "warning")
        self.label_setup.setWordWrap(True)
        root.addWidget(self.label_setup)
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        self.buttonBox.button(QDialogButtonBox.Ok).setText("Use Policy")
        root.addWidget(self.buttonBox)
        self.buttonBox.accepted.connect(self._accept_selection)
        self.buttonBox.rejected.connect(self.reject)
        self.tableWidget_templates.currentCellChanged.connect(
            lambda *_args: self._refresh_setup_message()
        )
        self.tableWidget_templates.doubleClicked.connect(
            lambda *_args: self._accept_selection()
        )
        self._refresh_setup_message()

    def selected_template(self) -> dict | None:
        row = self.tableWidget_templates.currentRow()
        item = self.tableWidget_templates.item(row, 0) if row >= 0 else None
        value = item.data(Qt.UserRole) if item is not None else None
        return value if isinstance(value, dict) else None

    def _refresh_setup_message(self) -> None:
        template = self.selected_template()
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(template is not None)
        policy = template.get("policy") if template else None
        kwargs = policy.get("kwargs", {}) if isinstance(policy, dict) else {}
        action = kwargs.get("action", {}) if isinstance(kwargs, dict) else {}
        needs_bound = (
            self.kind == "constraint"
            and isinstance(action, dict)
            and action.get("type") == "violate_bound"
            and not self.constraint_bound_ready
        )
        self.label_setup.setVisible(needs_bound)
        self.label_setup.setText(
            f"Setup required: {self.target} needs a Lower or Upper bound in "
            "Task Builder after Sync To Task."
            if needs_bound
            else ""
        )

    def _accept_selection(self) -> None:
        if self.selected_template() is None:
            QMessageBox.information(self, "Add Policy", "Choose a Policy Template first.")
            return
        self.accept()


class SampleGuardRuleEditorDialog(QDialog):
    """Structured editor for declarative objective/constraint sample guards."""

    METRICS = ("mean_abs", "max_abs", "peak_to_peak", "mean", "std", "reduced")
    OPERATORS = ("gt", "ge", "lt", "le", "eq", "ne")
    METRIC_LABELS = {
        "mean_abs": "Mean absolute sample",
        "max_abs": "Maximum absolute sample",
        "peak_to_peak": "Signal variation (peak-to-peak)",
        "mean": "Mean sample",
        "std": "Sample standard deviation",
        "reduced": "Processed result",
    }
    OPERATOR_LABELS = {
        "gt": "Greater than",
        "ge": "Greater than or equal",
        "lt": "Less than",
        "le": "Less than or equal",
        "eq": "Equal to",
        "ne": "Not equal to",
    }
    MATCH_LABELS = {"any": "Any condition", "all": "All conditions"}
    ACTION_LABELS = {
        "replace": "Replace result",
        "add_offset": "Add offset",
        "violate_bound": "Mark constraint infeasible",
    }

    def __init__(
        self,
        *,
        kind: str,
        target_names: list[str] | tuple[str, ...],
        policy_name: str = "sample_guard",
        kwargs: dict | None = None,
        preset_name: str | None = None,
        custom_presets: list[dict] | None = None,
        locked_target: str | None = None,
        pv_name: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        if kind not in {"objective", "constraint"}:
            raise ValueError("Policy Editor kind must be objective or constraint")
        self.kind = kind
        self.target_names = [str(name).strip() for name in target_names if str(name).strip()]
        self.locked_target = str(locked_target or "").strip()
        self.pv_name = str(pv_name or "").strip()
        self.custom_presets = {
            str(preset.get("id", "")).strip().lower(): preset
            for preset in custom_presets or []
            if preset.get("kind") == kind and str(preset.get("id", "")).strip()
        }
        if self.locked_target and self.locked_target not in self.target_names:
            self.target_names.append(self.locked_target)
        self._loading = False
        self.setWindowTitle(f"{kind.title()} Policy Editor")
        self.resize(760, 590)

        root = QVBoxLayout(self)
        heading_parts = [f"Edit {kind.title()} Policy"]
        if self.locked_target:
            heading_parts.append(self.locked_target)
        if self.pv_name:
            heading_parts.append(self.pv_name)
        heading = QLabel(" · ".join(heading_parts), self)
        heading.setObjectName("policyEditorTarget")
        heading.setWordWrap(True)
        root.addWidget(heading)
        intro = QLabel(
            "Define when this policy applies and what it should do. No code or JSON is required.",
            self,
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self.comboBox_preset = QComboBox(self)
        self.comboBox_preset.addItem("Custom Policy", "")
        for name in POLICY_REGISTRY.preset_names(kind, gui_only=True):
            preset = POLICY_REGISTRY.resolve_preset(kind, name)
            self.comboBox_preset.addItem(preset.display_name, preset.name)
        if self.custom_presets:
            self.comboBox_preset.insertSeparator(self.comboBox_preset.count())
            for preset_id, preset in self.custom_presets.items():
                self.comboBox_preset.addItem(str(preset.get("name", preset_id)), preset_id)
        form.addRow("Policy Template", self.comboBox_preset)

        self.comboBox_target = QComboBox(self)
        self.comboBox_target.setEditable(True)
        self.comboBox_target.addItems(self.target_names)
        if self.locked_target:
            self.comboBox_target.setCurrentText(self.locked_target)
            self.comboBox_target.setEnabled(False)
            self.comboBox_target.setToolTip(
                "This target is bound to the selected PV Mapping row."
            )
        form.addRow("Target", self.comboBox_target)

        self.comboBox_match = QComboBox(self)
        for value, label in self.MATCH_LABELS.items():
            self.comboBox_match.addItem(label, value)
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
        action_values = (
            ["replace", "add_offset"]
            if kind == "objective"
            else ["replace", "violate_bound"]
        )
        for value in action_values:
            self.comboBox_action.addItem(self.ACTION_LABELS[value], value)
        action_form.addRow("Policy action", self.comboBox_action)
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
        self.label_validation = QLabel(self)
        self.label_validation.setProperty("tone", "danger")
        self.label_validation.setWordWrap(True)
        root.addWidget(self.label_validation)
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        self.buttonBox.button(QDialogButtonBox.Ok).setText("Save Policy")
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
        for value in self.METRICS:
            metric.addItem(self.METRIC_LABELS[value], value)
        metric.setCurrentIndex(
            max(0, metric.findData(str(condition.get("metric", "mean_abs"))))
        )
        operator = QComboBox(self.tableWidget_conditions)
        for value in self.OPERATORS:
            operator.addItem(self.OPERATOR_LABELS[value], value)
        operator.setCurrentIndex(
            max(0, operator.findData(str(condition.get("operator", "gt"))))
        )
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
            target = self.locked_target or kwargs.get("target")
            target_col = int(kwargs.get("target_col", 0) or 0)
            if target is None and 0 <= target_col < len(self.target_names):
                target = self.target_names[target_col]
            self.comboBox_target.setCurrentText(str(target or ""))
            self.comboBox_match.setCurrentIndex(
                max(
                    0,
                    self.comboBox_match.findData(str(kwargs.get("match", "any"))),
                )
            )
            self.tableWidget_conditions.setRowCount(0)
            for condition in kwargs.get("conditions", []):
                self._set_condition_row(self.tableWidget_conditions.rowCount(), dict(condition))
            action = dict(kwargs.get("action", {}))
            self.comboBox_action.setCurrentIndex(
                max(
                    0,
                    self.comboBox_action.findData(str(action.get("type", "replace"))),
                )
            )
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
        if name in self.custom_presets:
            policy = self.custom_presets[name].get("policy", {}) or {}
            self._load_rule(
                dict(policy.get("kwargs", {}) or {}),
                preset_name=name,
            )
        elif name:
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
        violate = self.comboBox_action.currentData() == "violate_bound"
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
        target = self.locked_target or self.comboBox_target.currentText().strip()
        target_col = self.target_names.index(target) if target in self.target_names else 0
        conditions = []
        for row in range(self.tableWidget_conditions.rowCount()):
            condition = {
                "metric": self.tableWidget_conditions.cellWidget(row, 0).currentData(),
                "operator": self.tableWidget_conditions.cellWidget(row, 1).currentData(),
                "value": self.tableWidget_conditions.cellWidget(row, 2).value(),
            }
            atol = self.tableWidget_conditions.cellWidget(row, 3).value()
            if atol:
                condition["atol"] = atol
            conditions.append(condition)
        action_type = str(self.comboBox_action.currentData())
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
                "match": str(self.comboBox_match.currentData()),
                "action": action,
            },
        }

    def _refresh_summary(self) -> None:
        state = self.rule_state()
        conditions = state["kwargs"]["conditions"]
        count = len(conditions)
        target = state["kwargs"]["target"] or f"column {state['kwargs']['target_col']}"
        action_state = state["kwargs"]["action"]
        action_type = action_state["type"]
        if action_type == "replace":
            action = f"replace the result with {action_state['value']:g}"
        elif action_type == "add_offset":
            action = f"add {action_state['value']:g} to the result"
        else:
            action = "mark the constraint as infeasible"
        if count == 1:
            condition = conditions[0]
            metric = self.METRIC_LABELS.get(condition["metric"], condition["metric"])
            operator = self.OPERATOR_LABELS.get(
                condition["operator"], condition["operator"]
            ).lower()
            match_text = f"{metric} is {operator} {condition['value']:g}"
        else:
            quantifier = "any" if state["kwargs"]["match"] == "any" else "all"
            match_text = f"{quantifier} of {count} conditions match"
        self.label_summary.setText(
            f"Policy behavior: {target} — when {match_text}, {action}."
        )
        self._refresh_validation(state)

    def _refresh_validation(self, state: dict | None = None) -> bool:
        try:
            current = state or self.rule_state()
            if not current["kwargs"]["conditions"]:
                raise ValueError("Add at least one condition before saving this policy.")
            POLICY_REGISTRY.validate(self.kind, current["name"], current["kwargs"])
        except Exception as exc:
            self.label_validation.setText(str(exc))
            self.label_validation.setVisible(True)
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
            return False
        self.label_validation.clear()
        self.label_validation.setVisible(False)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
        return True

    def _accept_if_valid(self) -> None:
        if not self._refresh_validation():
            return
        self.accept()


class MappingPolicyManagerDialog(QDialog):
    """Select an add/edit/remove action for policies bound to one mapping row."""

    def __init__(self, *, target: str, pv_name: str, policies: list[dict], parent=None) -> None:
        super().__init__(parent)
        self._request: tuple[str, int | None] | None = None
        self.setWindowTitle(f"Policies for {target}")
        self.resize(760, 380)

        root = QVBoxLayout(self)
        heading = QLabel(f"{target} — {pv_name or 'PV not assigned'}", self)
        heading.setWordWrap(True)
        root.addWidget(heading)
        hint = QLabel(
            "Policies are bound to this Machine PV Mapping signal and are included when the task is built.",
            self,
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.tableWidget_policies = QTableWidget(0, 3, self)
        self.tableWidget_policies.setHorizontalHeaderLabels(
            ["Enabled", "Policy Template", "Policy behavior"]
        )
        self.tableWidget_policies.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget_policies.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget_policies.setSelectionMode(QAbstractItemView.SingleSelection)
        for policy in policies:
            row = self.tableWidget_policies.rowCount()
            self.tableWidget_policies.insertRow(row)
            self.tableWidget_policies.setItem(
                row, 0, QTableWidgetItem("Yes" if policy.get("enabled") else "No")
            )
            self.tableWidget_policies.setItem(
                row, 1, QTableWidgetItem(str(policy.get("preset", "Custom Policy")))
            )
            self.tableWidget_policies.setItem(
                row, 2, QTableWidgetItem(str(policy.get("summary", "sample guard")))
            )
        if self.tableWidget_policies.rowCount():
            self.tableWidget_policies.selectRow(0)
        root.addWidget(self.tableWidget_policies)

        actions = QHBoxLayout()
        self.pushButton_add = QPushButton("Add Policy", self)
        self.pushButton_edit = QPushButton("Edit Policy", self)
        self.pushButton_remove = QPushButton("Remove Selected", self)
        self.pushButton_toggle = QPushButton("Enable / Disable", self)
        self.pushButton_savePreset = QPushButton("Save as Template", self)
        close_button = QPushButton("Close", self)
        actions.addWidget(self.pushButton_add)
        actions.addWidget(self.pushButton_edit)
        actions.addWidget(self.pushButton_remove)
        actions.addWidget(self.pushButton_toggle)
        actions.addWidget(self.pushButton_savePreset)
        actions.addStretch(1)
        actions.addWidget(close_button)
        root.addLayout(actions)

        has_policies = bool(policies)
        self.pushButton_edit.setEnabled(has_policies)
        self.pushButton_remove.setEnabled(has_policies)
        self.pushButton_toggle.setEnabled(has_policies)
        self.pushButton_savePreset.setEnabled(has_policies)
        self.pushButton_add.clicked.connect(lambda: self._finish("add"))
        self.pushButton_edit.clicked.connect(lambda: self._finish("edit"))
        self.pushButton_remove.clicked.connect(lambda: self._finish("remove"))
        self.pushButton_toggle.clicked.connect(lambda: self._finish("toggle"))
        self.pushButton_savePreset.clicked.connect(lambda: self._finish("save_preset"))
        self.tableWidget_policies.doubleClicked.connect(lambda *_: self._finish("edit"))
        close_button.clicked.connect(self.reject)

    def _finish(self, action: str) -> None:
        row = self.tableWidget_policies.currentRow()
        if action != "add" and row < 0:
            return
        self._request = (action, None if action == "add" else row)
        self.accept()

    def requested_action(self) -> tuple[str, int | None] | None:
        return self._request

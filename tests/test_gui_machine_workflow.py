import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from gotacc.gui.services.task_service import TaskService
from gotacc.gui.views.main_window import MainWindow
from gotacc.gui.views.tool_dialogs import BoundsToolsDialog


def _online_task(tmp_path):
    return {
        "task_name": "machine_sync_test",
        "description": "",
        "mode": "Online EPICS",
        "objective_type": "Single Objective",
        "algorithm": "BO",
        "max_evaluations": 5,
        "seed": 1,
        "workdir": str(tmp_path),
        "test_function": "",
        "variables": [
            {
                "Enable": "Y",
                "Name": "Q1",
                "Lower": "-1",
                "Upper": "1",
                "Initial": "0.1",
                "Group": "main",
            },
            {
                "Enable": "Y",
                "Name": "Q2",
                "Lower": "-2",
                "Upper": "2",
                "Initial": "0.2",
                "Group": "main",
            },
        ],
        "objectives": [
            {
                "Enable": "Y",
                "Name": "Transmission",
                "Direction": "maximize",
                "Weight": "1",
                "Samples": "1",
                "Math": "mean",
            }
        ],
        "constraints": [],
        "algorithm_params": [],
        "machine": {
            "ca_address": "",
            "restore_on_abort": True,
            "readback_check": True,
            "readback_tol": 1e-6,
            "set_interval": 0.5,
            "sample_interval": 0.1,
            "write_timeout": 2.0,
            "write_policy": "none",
            "objective_policies": [],
            "constraint_policies": [],
            "write_links": [],
            "mapping": [
                {"Role": "knob", "Name": "Q2", "PV Name": "TEST:Q2:SET", "Readback": "TEST:Q2:RB"},
                {"Role": "knob", "Name": "Q1", "PV Name": "TEST:Q1:SET", "Readback": "TEST:Q1:RB"},
                {"Role": "objective", "Name": "Transmission", "PV Name": "TEST:TRANS", "Readback": ""},
            ],
        },
    }


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)
    instance = MainWindow()
    yield instance
    instance.close()
    app.processEvents()


def test_mapping_master_detail_edits_selected_signal(tmp_path, window):
    task = _online_task(tmp_path)
    window._apply_task_payload(task, goto_builder=False)
    table = window.machine_ui.tableWidget_mapping
    table.setCurrentCell(2, 1)

    assert window.machine_ui.label_mappingDetailTitle.text() == "Objective · Transmission"
    assert window.machine_ui.lineEdit_mappingDetailPv.text() == "TEST:TRANS"
    assert window.machine_ui.pushButton_manageMappingPolicies.text() == "Add Policy"

    window.machine_ui.lineEdit_mappingDetailReadback.setText("TEST:TRANS:RB")
    window.machine_ui.lineEdit_mappingDetailReadback.editingFinished.emit()
    window.machine_ui.lineEdit_mappingDetailNote.setText("Primary transmission monitor")
    window.machine_ui.lineEdit_mappingDetailNote.editingFinished.emit()

    headers = window.task_builder_controller.table_headers(table)
    assert table.item(2, headers.index("Readback")).text() == "TEST:TRANS:RB"
    assert table.item(2, headers.index("Note")).text() == "Primary transmission monitor"
    serialized = window._current_task()["machine"]["mapping"][2]
    assert serialized["Readback"] == "TEST:TRANS:RB"
    assert serialized["Note"] == "Primary transmission monitor"
    assert "Policies" not in serialized


def test_mapping_sync_preserves_parameters_by_name_and_can_undo(tmp_path, window):
    task = _online_task(tmp_path)
    task["variables"].append(
        {
            "Enable": "N",
            "Name": "Legacy",
            "Lower": "-5",
            "Upper": "5",
            "Initial": "0",
            "Group": "main",
        }
    )
    task["machine"]["mapping"].insert(
        2,
        {"Role": "knob", "Name": "Q3", "PV Name": "TEST:Q3:SET", "Readback": "TEST:Q3:RB"},
    )
    window._apply_task_payload(task, goto_builder=False)
    window.go_to_page(window.PAGE_MACHINE)

    window.machine_controller.apply_selected_pv_library_entries()

    rows = TaskService.table_to_records(window.task_ui.tableWidget_variables)
    assert [row["Name"] for row in rows] == ["Q2", "Q1", "Q3"]
    assert rows[0]["Lower"] == "-2"
    assert rows[0]["Initial"] == "0.2"
    assert rows[1]["Lower"] == "-1"
    assert rows[1]["Initial"] == "0.1"
    assert rows[2]["Enable"] == "Y"
    assert rows[2]["Lower"] == ""
    assert rows[2]["Upper"] == ""
    assert rows[2]["Initial"] == ""
    assert "1 needs setup" in window.machine_ui.label_pvLibrarySummary.text()
    assert window.machine_ui.pushButton_undoMappingSync.isEnabled()
    assert window.ui.tabWidget_configure.currentIndex() == window.CONFIGURE_TAB_TASK_BUILDER
    assert window.task_ui.tabWidget_tables.currentIndex() == 0
    assert window.task_ui.tableWidget_variables.currentRow() == 2

    window.machine_controller.undo_last_mapping_sync()

    restored = TaskService.table_to_records(window.task_ui.tableWidget_variables)
    assert [row["Name"] for row in restored] == ["Q1", "Q2", "Legacy"]
    assert not window.machine_ui.pushButton_undoMappingSync.isEnabled()


def test_pv_check_covers_current_contract_and_becomes_stale(tmp_path, window, monkeypatch):
    task = _online_task(tmp_path)
    window._apply_task_payload(task, goto_builder=False)
    window.go_to_page(window.PAGE_MACHINE)
    window.machine_controller.apply_selected_pv_library_entries()
    assert window.ui.tabWidget_configure.currentIndex() == window.CONFIGURE_TAB_MACHINE
    reads = []

    def fake_caget(pvname, *, timeout):
        reads.append((pvname, timeout))
        return 1.0

    monkeypatch.setattr(window.machine_controller, "_prepare_epics_caget", lambda: fake_caget)

    assert window.machine_controller.check_machine_pv(show_dialog=False)
    assert {pv for pv, _timeout in reads} == {
        "TEST:Q1:SET",
        "TEST:Q1:RB",
        "TEST:Q2:SET",
        "TEST:Q2:RB",
        "TEST:TRANS",
    }
    current = window._current_task()
    assert window.machine_controller.ensure_machine_ready_for_online(current)
    assert window.machine_ui.label_statusValue.text() == "PV Check Passed"

    window.machine_ui.tableWidget_mapping.item(0, 2).setText("TEST:Q2:NEW")
    QApplication.processEvents()

    assert window.machine_ui.label_statusValue.text() == "Stale"
    assert window.state.last_test_read_status == "Stale"
    assert not window.state.machine_check_identity
    assert not window.machine_controller.ensure_machine_ready_for_online(window._current_task())


def test_online_validation_rejects_mapping_ambiguity(tmp_path):
    task = _online_task(tmp_path)
    task["machine"]["mapping"][1]["PV Name"] = "TEST:Q2:SET"
    task["variables"].append(dict(task["variables"][0]))

    ok, errors = TaskService.validate_task_data(task)

    assert not ok
    assert any("Duplicate enabled variable name" in error for error in errors)
    assert any("share Setpoint PV" in error for error in errors)


def test_bounds_tool_previews_exact_plan_before_apply(tmp_path, window, monkeypatch):
    window._apply_task_payload(_online_task(tmp_path), goto_builder=False)
    controller = window.task_builder_controller
    dialog = BoundsToolsDialog(window)
    controller._bounds_dialog = dialog
    source_reads = []
    try:
        ui = dialog.ui
        ui.comboBox_boundsSource.setCurrentText("Initial values")
        ui.comboBox_boundsMode.setCurrentText("± absolute delta")
        ui.doubleSpinBox_boundsPrimary.setValue(0.5)
        ui.checkBox_boundsUpdateInitial.setChecked(True)
        controller.update_bounds_tool_controls()
        monkeypatch.setattr(
            controller,
            "_resolve_bounds_source_values",
            lambda _task, rows: source_reads.append(len(rows)) or [1.0, 2.0],
        )

        controller.preview_bounds_tool()

        assert source_reads == [2]
        assert ui.tableWidget_boundsPreview.rowCount() == 2
        assert ui.tableWidget_boundsPreview.item(0, 0).text() == "Q1"
        assert ui.tableWidget_boundsPreview.item(0, 1).text() == "1"
        assert ui.tableWidget_boundsPreview.item(0, 2).text() == "0.5"
        assert ui.tableWidget_boundsPreview.item(0, 3).text() == "1.5"
        assert ui.tableWidget_boundsPreview.item(0, 4).text() == "1"
        assert ui.pushButton_applyBounds.isEnabled()
        assert ui.pushButton_applyBounds.property("primary") is True

        controller.apply_bounds_tool()

        assert source_reads == [2]
        variables = TaskService.table_to_records(window.task_ui.tableWidget_variables)
        assert variables[0]["Lower"] == "0.5"
        assert variables[0]["Upper"] == "1.5"
        assert variables[0]["Initial"] == "1"
        assert variables[1]["Lower"] == "1.5"
        assert variables[1]["Upper"] == "2.5"
        assert variables[1]["Initial"] == "2"
        assert not ui.pushButton_applyBounds.isEnabled()

        ui.doubleSpinBox_boundsPrimary.setValue(0.75)
        controller._on_bounds_tool_settings_changed()
        assert ui.tableWidget_boundsPreview.rowCount() == 0
        assert not controller._bounds_preview_plan

        ui.comboBox_boundsMode.setCurrentText("Fixed lower / upper")
        ui.checkBox_boundsUpdateInitial.setChecked(False)
        controller.update_bounds_tool_controls()
        ui.doubleSpinBox_boundsPrimary.setValue(-3.0)
        ui.doubleSpinBox_boundsSecondary.setValue(3.0)
        assert not ui.comboBox_boundsSource.isEnabled()
        monkeypatch.setattr(
            controller,
            "_resolve_bounds_source_values",
            lambda *_args: pytest.fail("Fixed bounds should not read a source"),
        )
        controller.preview_bounds_tool()
        assert ui.tableWidget_boundsPreview.item(0, 1).text() == "Not used"
        assert ui.tableWidget_boundsPreview.item(0, 2).text() == "-3"
        assert ui.tableWidget_boundsPreview.item(0, 3).text() == "3"
    finally:
        controller._bounds_dialog = None
        dialog.close()

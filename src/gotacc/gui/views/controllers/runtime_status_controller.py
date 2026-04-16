from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..main_window import MainWindow


class RuntimeStatusController:
    def __init__(self, window: "MainWindow") -> None:
        self.window = window
        self.view = window.view_adapter

    def update_runtime_labels(self) -> None:
        run = self.window.state.run
        hh = run.elapsed_seconds // 3600
        mm = (run.elapsed_seconds % 3600) // 60
        ss = run.elapsed_seconds % 60
        self.window.run_ui.label_elapsedValue.setText(f"{hh:02d}:{mm:02d}:{ss:02d}")
        self.window.run_ui.label_evalValue.setText(str(run.eval_count))
        self.window.run_ui.label_bestValue.setText(
            "--" if run.best_value is None else f"{run.best_value:.6f}"
        )
        self.window.run_ui.label_feasibilityValue.setText(f"{run.feasibility_ratio:.2f}")
        self.window.ui.label_statusBestValue.setText(
            "--" if run.best_value is None else f"{run.best_value:.6f}"
        )
        if hasattr(self.window.run_ui, "label_runSummary"):
            best_text = "--" if run.best_value is None else f"{run.best_value:.6f}"
            self.window.run_ui.label_runSummary.setText(
                f"{run.phase} · {run.eval_count} evaluation(s) · best {best_text} · feasibility {run.feasibility_ratio:.2f}"
            )
        self.sync_status_panels()

    def set_run_buttons_enabled(self, *, start: bool, pause: bool, resume: bool, stop: bool) -> None:
        self.window.ui.pushButton_startRun.setEnabled(start)
        self.window.ui.pushButton_pauseRun.setEnabled(pause)
        self.window.ui.pushButton_stopRun.setEnabled(stop)
        self.window.run_ui.pushButton_start.setEnabled(start)
        self.window.run_ui.pushButton_pause.setEnabled(pause)
        self.window.run_ui.pushButton_resume.setEnabled(resume)
        self.window.run_ui.pushButton_stop.setEnabled(stop)
        self.window.run_ui.pushButton_abortRestore.setEnabled(not start)
        if hasattr(self.window.run_ui, "pushButton_restoreInitial"):
            self.window.run_ui.pushButton_restoreInitial.setEnabled(
                bool(start and self.window.state.latest_initial_x)
            )
        self.window.run_ui.pushButton_setBest.setEnabled(self.window.state.run.best_value is not None)

    def set_run_phase(self, text: str) -> None:
        self.window.run_ui.label_phaseValue.setText(text)
        self.window.ui.label_cardStatusValue.setText(text)

    def append_run_history(self, status: str) -> None:
        task_name = self.window.task_ui.lineEdit_taskName.text().strip() or "untitled_task"
        mode = self.window.task_ui.comboBox_mode.currentText()
        algorithm = self.window.task_ui.comboBox_algorithm.currentText()
        row = self.window.ui.tableWidget_runHistory.rowCount()
        self.window.ui.tableWidget_runHistory.insertRow(row)
        values = [
            str(row + 1),
            task_name,
            mode,
            algorithm,
            status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]
        self.view.set_table_row(self.window.ui.tableWidget_runHistory, row, values)
        self.view.append_overview_activity(
            "Run",
            task=task_name,
            mode=mode,
            algorithm=algorithm,
            status=status,
        )

    def sync_status_panels(self) -> None:
        run = self.window.state.run
        self.window.ui.label_cardStatusValue.setText(run.phase)
        self.window.ui.label_statusBestValue.setText(
            "--" if run.best_value is None else f"{run.best_value:.6f}"
        )
        if hasattr(self.window.run_ui, "label_runSummary"):
            best_text = "--" if run.best_value is None else f"{run.best_value:.6f}"
            self.window.run_ui.label_runSummary.setText(
                f"{run.phase} · {run.eval_count} evaluation(s) · best {best_text} · feasibility {run.feasibility_ratio:.2f}"
            )
        self.window.run_ui.pushButton_setBest.setEnabled(run.best_value is not None)
        if hasattr(self.window.run_ui, "pushButton_restoreInitial"):
            self.window.run_ui.pushButton_restoreInitial.setEnabled(
                bool(run.phase not in {"Running", "Paused", "Stopping"} and self.window.state.latest_initial_x)
            )

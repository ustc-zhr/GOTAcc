from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem, QTreeWidgetItem, QVBoxLayout

if TYPE_CHECKING:  # pragma: no cover
    from ..main_window import MainWindow


class ResultsController:
    def __init__(self, window: "MainWindow", canvas_class) -> None:
        self.window = window
        self.view = window.view_adapter
        self.canvas_class = canvas_class

    def init_results_page(self) -> None:
        tree = self.window.ui.treeWidget_runList
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Run / Artifact", "Path / Value"])
        tree.header().setStretchLastSection(True)
        self.populate_results_tree()
        self.update_results_summary_table()

    def init_plot_canvases(self) -> None:
        self.window.obj_canvas = self.attach_plot_canvas(self.window.run_ui.frame_obj)
        self.window.pareto_canvas = self.attach_plot_canvas(self.window.run_ui.frame_pareto)
        self.window.run_var_canvas = self.attach_plot_canvas(self.window.run_ui.frame_variables)
        self.window.run_constraints_canvas = self.attach_plot_canvas(self.window.run_ui.frame_constraints)
        self.window.results_conv_canvas = self.attach_plot_canvas(self.window.ui.frame_plotConvergence)
        self.window.results_pareto_canvas = self.attach_plot_canvas(self.window.ui.frame_plotParetoFinal)
        variable_frame = getattr(self.window.ui, "frame_plotVariables", self.window.ui.frame_plotConvergence)
        self.window.var_canvas = self.attach_plot_canvas(variable_frame)
        self.reset_plot_data()
        self.redraw_plots()

    def attach_plot_canvas(self, frame):
        frame.setMinimumSize(180, 140)
        layout = frame.layout()
        if layout is None:
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(4, 4, 4, 4)
        else:
            layout.setContentsMargins(4, 4, 4, 4)
            while layout.count():
                item = layout.takeAt(0)
                child = item.widget()
                if child is not None:
                    child.deleteLater()
        canvas = self.canvas_class(frame)
        layout.addWidget(canvas)
        return canvas

    def reset_plot_data(self) -> None:
        state = self.window.state
        objective_type = self.window.task_ui.comboBox_objectiveType.currentText().strip().lower()
        algorithm = self.window.task_ui.comboBox_algorithm.currentText().strip().lower()
        if objective_type == "multi objective" or algorithm in {"mobo", "mopso", "nsga2"}:
            objective_dim = 2
        else:
            objective_dim = 1
        state.reset_plot_data(objective_dim)

    def redraw_plots(self) -> None:
        if not all(
            hasattr(self.window, name)
            for name in ("obj_canvas", "pareto_canvas", "results_conv_canvas", "results_pareto_canvas")
        ):
            return
        self.draw_objective_plot(self.window.obj_canvas, title="Objective History")
        self.draw_pareto_plot(self.window.pareto_canvas, title="Pareto / HV")
        self.draw_objective_plot(self.window.results_conv_canvas, title="Convergence")
        self.draw_pareto_plot(self.window.results_pareto_canvas, title="Final Pareto")
        if hasattr(self.window, "run_constraints_canvas"):
            self.window.run_constraints_canvas.clear_with_message(
                "Constraint History",
                "Constraint detail is summarized in the evaluation tables for the current GUI flow.",
            )
        self.draw_variable_trajectories()

    def draw_objective_plot(self, canvas, *, title: str) -> None:
        state = self.window.state
        if state.objective_dim != 1:
            if not state.hypervolume_history:
                canvas.clear_with_message(
                    title,
                    "Hypervolume history is shown here for multi-objective tasks once available.",
                )
                return
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            xs = list(range(1, len(state.hypervolume_history) + 1))
            ax.plot(xs, state.hypervolume_history, label="hypervolume")
            ax.set_title(title)
            ax.set_xlabel("Generation / Update")
            ax.set_ylabel("Hypervolume")
            ax.grid(True, alpha=0.3)
            ax.legend()
            canvas.draw_idle()
            return
        if not state.objective_history:
            canvas.clear_with_message(title, "No evaluations yet.")
            return

        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        xs = list(range(1, len(state.objective_history) + 1))
        ax.plot(xs, state.objective_history, label="objective")
        if state.best_history:
            ax.plot(xs, state.best_history, label="best-so-far")
        ax.set_title(title)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Objective")
        ax.grid(True, alpha=0.3)
        ax.legend()
        canvas.draw_idle()

    def draw_pareto_plot(self, canvas, *, title: str) -> None:
        state = self.window.state
        if state.objective_dim == 1:
            canvas.clear_with_message(title, "Pareto scatter is available for multi-objective tasks.")
            return
        points = state.pareto_points
        if "final" in title.lower() and state.pareto_front_points:
            points = state.pareto_front_points
        if not points:
            canvas.clear_with_message(title, "No multi-objective evaluations yet.")
            return

        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        xs = [p[0] for p in points if len(p) >= 2]
        ys = [p[1] for p in points if len(p) >= 2]
        ax.scatter(xs, ys)
        ax.set_title(title)
        ax.set_xlabel("f0")
        ax.set_ylabel("f1")
        ax.grid(True, alpha=0.3)
        canvas.draw_idle()

    def draw_variable_trajectories(self) -> None:
        targets = []
        if hasattr(self.window, "var_canvas"):
            targets.append(self.window.var_canvas)
        if hasattr(self.window, "run_var_canvas"):
            targets.append(self.window.run_var_canvas)
        if not targets:
            return
        if not self.window.state.eval_x_history:
            for canvas in targets:
                canvas.clear_with_message("Variable Trajectories", "No evaluation vectors yet.")
            return

        import numpy as np

        X = np.array(self.window.state.eval_x_history)
        for canvas in targets:
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)

            for i in range(X.shape[1]):
                ax.plot(X[:, i], label=f"x{i}")

            ax.set_title("Variable Trajectories")
            ax.set_xlabel("Evaluation")
            ax.set_ylabel("Value")
            ax.legend()
            canvas.draw_idle()

    def populate_history_table(self) -> None:
        table = getattr(self.window.ui, "tableWidget_history", None)
        if table is None:
            return

        table.setRowCount(len(self.window.state.eval_history))

        for i, (x, y, _) in enumerate(self.window.state.eval_history):
            table.setItem(i, 0, QTableWidgetItem(str(i)))
            table.setItem(i, 1, QTableWidgetItem(str(x)))
            table.setItem(i, 2, QTableWidgetItem(str(y)))

    def on_history_row_clicked(self, row) -> None:
        if row >= len(self.window.state.eval_history):
            return

        x, y, c = self.window.state.eval_history[row]

        inspector = self.window.ui.tableWidget_solutionInspector
        if not self.view.qobj_alive(inspector):
            return

        self.view.set_table_row(inspector, 0, ["Run", self.window.task_ui.lineEdit_taskName.text()])
        self.view.set_table_row(inspector, 1, ["Point", str(x)])
        self.view.set_table_row(inspector, 2, ["Objective", str(y)])
        self.view.set_table_row(inspector, 3, ["Constraints", str(c)])

    def append_recent_eval(self, payload: dict) -> None:
        eval_id = str(payload.get("eval_id", ""))
        timestamp = str(payload.get("timestamp", ""))
        status = str(payload.get("status", ""))
        x_values = payload.get("x_values", {})
        x_summary = ", ".join(f"{k}={v:.3f}" for k, v in list(x_values.items())[:3])
        if payload.get("objective_summary"):
            y_summary = str(payload.get("objective_summary"))
        elif payload.get("objective_value") is not None:
            y_summary = f"y0={float(payload.get('objective_value', 0.0)):.6f}"
        else:
            y_summary = "--"
        c_summary = str(payload.get("constraint_summary", ""))

        for table in self.view.living_tables(
            self.window.run_ui.tableWidget_recent,
            getattr(self.window.ui, "tableWidget_recentEvaluations", None),
        ):
            row = table.rowCount()
            table.insertRow(row)
            self.view.set_table_row(table, row, [eval_id, timestamp, status, x_summary, y_summary, c_summary])

    def summarize_x_values(self, x_values: dict | None) -> str:
        if not x_values:
            return "--"
        return ", ".join(f"{k}={float(v):.6g}" for k, v in x_values.items())

    def populate_results_tree(self) -> None:
        state = self.window.state
        tree = self.window.ui.treeWidget_runList
        tree.clear()

        run_task = (state.latest_task_snapshot or {}).get(
            "task_name",
            self.window.task_ui.lineEdit_taskName.text().strip() or "untitled_task",
        )
        run_state = (
            state.latest_finish_payload.get("state", state.run.phase)
            if state.latest_finish_payload
            else state.run.phase
        )
        run_item = QTreeWidgetItem([f"Run: {run_task}", run_state])
        run_item.setData(0, Qt.UserRole, {"kind": "run"})
        tree.addTopLevelItem(run_item)

        summary = QTreeWidgetItem(["Summary", ""])
        summary.setData(0, Qt.UserRole, {"kind": "summary"})
        run_item.addChild(summary)
        summary.addChild(
            QTreeWidgetItem(
                ["Best Value", "--" if state.run.best_value is None else f"{state.run.best_value:.6f}"]
            )
        )
        summary.addChild(QTreeWidgetItem(["Best Point", self.summarize_x_values(state.latest_best_x)]))
        summary.addChild(QTreeWidgetItem(["Objective Dim", str(state.objective_dim)]))
        summary.addChild(QTreeWidgetItem(["Evaluations", str(state.run.eval_count)]))
        if state.objective_dim > 1:
            summary.addChild(
                QTreeWidgetItem(["Pareto Points", str(len(state.pareto_front_points) or len(state.pareto_points))])
            )
            summary.addChild(QTreeWidgetItem(["HV Samples", str(len(state.hypervolume_history))]))

        artifacts = QTreeWidgetItem(["Artifacts", ""])
        artifacts.setData(0, Qt.UserRole, {"kind": "artifacts"})
        run_item.addChild(artifacts)

        if state.latest_history_path:
            item = QTreeWidgetItem(["History File", state.latest_history_path])
            item.setData(0, Qt.UserRole, {"kind": "path", "path": state.latest_history_path})
            artifacts.addChild(item)
        if state.latest_plot_path:
            item = QTreeWidgetItem(["Convergence Plot", state.latest_plot_path])
            item.setData(0, Qt.UserRole, {"kind": "path", "path": state.latest_plot_path})
            artifacts.addChild(item)
        if state.latest_result_output_dir:
            item = QTreeWidgetItem(["Output Directory", state.latest_result_output_dir])
            item.setData(0, Qt.UserRole, {"kind": "path", "path": state.latest_result_output_dir})
            artifacts.addChild(item)

        latest_eval = QTreeWidgetItem(["Latest Evaluation", ""])
        latest_eval.setData(0, Qt.UserRole, {"kind": "latest_eval"})
        run_item.addChild(latest_eval)
        if state.latest_eval_payload:
            latest_eval.addChild(
                QTreeWidgetItem(
                    ["Point", self.summarize_x_values(state.latest_eval_payload.get("x_values", {}))]
                )
            )
            latest_eval.addChild(
                QTreeWidgetItem(
                    [
                        "Objective",
                        str(
                            state.latest_eval_payload.get("objective_summary")
                            or state.latest_eval_payload.get("objective_value")
                            or "--"
                        ),
                    ]
                )
            )
            latest_eval.addChild(
                QTreeWidgetItem(
                    ["Constraints", str(state.latest_eval_payload.get("constraint_summary", "--"))]
                )
            )

        tree.expandAll()
        if tree.topLevelItemCount() > 0 and tree.currentItem() is None:
            tree.setCurrentItem(run_item)

    def update_results_summary_table(self, selected_item=None) -> None:
        state = self.window.state
        table = self.window.ui.tableWidget_solutionInspector
        rows = []
        task_name = (state.latest_task_snapshot or {}).get(
            "task_name",
            self.window.task_ui.lineEdit_taskName.text().strip() or "untitled_task",
        )
        rows.append(("Task", task_name))
        rows.append(
            (
                "Status",
                state.latest_finish_payload.get("state", state.run.phase)
                if state.latest_finish_payload
                else state.run.phase,
            )
        )
        rows.append(("Best Value", "--" if state.run.best_value is None else f"{state.run.best_value:.6f}"))
        rows.append(("Best Point", self.summarize_x_values(state.latest_best_x)))
        rows.append(("History Path", state.latest_history_path or "--"))
        rows.append(("Output Directory", state.latest_result_output_dir or "--"))
        if state.objective_dim > 1:
            rows.append(("Pareto Points", str(len(state.pareto_front_points) or len(state.pareto_points))))
            rows.append(("HV Samples", str(len(state.hypervolume_history))))

        if selected_item is not None:
            data = selected_item.data(0, Qt.UserRole) or {}
            rows.append(("Selected Item", selected_item.text(0)))
            rows.append(("Selected Value", selected_item.text(1)))
            if isinstance(data, dict) and data.get("kind") == "path":
                rows.append(("Action", "Double-click to open"))

        table.setRowCount(0)
        for field, value in rows:
            row = table.rowCount()
            table.insertRow(row)
            self.view.set_table_row(table, row, [field, value])

    def on_results_tree_selection_changed(self) -> None:
        items = self.window.ui.treeWidget_runList.selectedItems()
        self.update_results_summary_table(items[0] if items else None)

    def open_selected_result_item(self, item, _column: int) -> None:
        data = item.data(0, Qt.UserRole) or {}
        path = data.get("path") if isinstance(data, dict) else None
        if not path:
            return
        file_path = Path(path)
        target = file_path if file_path.exists() else file_path.parent
        if not target.exists():
            QMessageBox.information(self.window, "Open Artifact", f"Path does not exist yet:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def update_results_after_start(self, task: dict) -> None:
        state = self.window.state
        state.latest_task_snapshot = dict(task)
        state.latest_eval_payload.clear()
        state.latest_finish_payload.clear()
        state.latest_best_x.clear()
        state.latest_history_path = ""
        state.latest_plot_path = ""
        state.pareto_front_points.clear()
        state.hypervolume_history.clear()
        state.latest_result_output_dir = str(Path(task.get("workdir", Path.cwd())).resolve())
        self.populate_results_tree()
        self.update_results_summary_table()

    def update_results_after_evaluation(self, payload: dict) -> None:
        state = self.window.state
        state.latest_eval_payload = dict(payload)
        if payload.get("best_changed"):
            state.latest_best_x = dict(payload.get("x_values", {}))
        self.populate_results_tree()
        self.update_results_summary_table()

    def update_results_after_finish(self, payload: dict) -> None:
        state = self.window.state
        state.latest_finish_payload = dict(payload)
        best_x = payload.get("best_x")
        if isinstance(best_x, dict) and best_x:
            state.latest_best_x = dict(best_x)
        pareto_y = payload.get("pareto_y")
        if isinstance(pareto_y, list):
            state.pareto_front_points = [
                (float(row[0]), float(row[1]))
                for row in pareto_y
                if isinstance(row, (list, tuple)) and len(row) >= 2
            ]
        hv_history = payload.get("hypervolume_history")
        if isinstance(hv_history, list):
            state.hypervolume_history = [float(v) for v in hv_history]
        state.latest_history_path = str(payload.get("history_path") or "")
        state.latest_plot_path = str(payload.get("plot_path") or "")
        output_dir = ""
        if state.latest_history_path:
            output_dir = str(Path(state.latest_history_path).resolve().parent)
        elif state.latest_plot_path:
            output_dir = str(Path(state.latest_plot_path).resolve().parent)
        elif state.latest_task_snapshot:
            output_dir = str(Path(state.latest_task_snapshot.get("workdir", Path.cwd())).resolve())
        state.latest_result_output_dir = output_dir
        self.populate_results_tree()
        self.update_results_summary_table()

# main_app.py
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout

from lane_map_viewer_gui import LaneMapViewer
from scenario_rule_gui import ScenarioRuleGUI


class MainApp(QMainWindow):
    """Only coordination layer: routes authoritative semantic data
    between the map viewer and the scenario GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRACE")

        self.viewer = LaneMapViewer()
        self.scenario_gui = ScenarioRuleGUI()
        self.viewer.scenario_gui = self.scenario_gui
        self._active_scenario_mode = None  # "manual" | "json"

        # semantic routing (map ctx -> scenario, enriched ctx -> viewer)
        self.viewer.mask_loaded.connect(self._sync_mask_meta)
        self.viewer.meta_loaded.connect(self._sync_mask_meta)
        self.viewer.map_context_ready.connect(self._on_context_update)
        self.scenario_gui.engine_context_ready.connect(
            self.viewer.receive_engine_context
        )

        # timing & road network
        self.viewer.duration_changed.connect(
            self.scenario_gui.set_duration_seconds
        )
        self.viewer.xodr_changed.connect(self.scenario_gui.set_xodr_path)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(self.viewer, stretch=2)
        layout.addWidget(self.scenario_gui, stretch=1)
        self.setCentralWidget(container)

    def _on_context_update(self, ctx):
        self._active_scenario_mode = ctx.get("mode")
        print("CONTEXT SENT")
        self.scenario_gui.ingest_map_context(ctx)
        # new map context invalidates previously generated rules
        self.scenario_gui._rules = []
        self.scenario_gui.btn_explain.setEnabled(False)

    def _sync_mask_meta(self, *_):
        if self.viewer.view.mask is not None and self.viewer.view.meta is not None:
            self.scenario_gui.set_mask_meta(
                self.viewer.view.mask, self.viewer.view.meta
            )


def main():
    import sys
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

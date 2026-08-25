import json

import numpy as np
import pytest

from gotacc.interfaces.epics import BPMGuardConstraintPolicy, FelEnergyGuardPolicy
from gotacc.interfaces.policies import POLICY_REGISTRY


def test_fel_and_bpm_presets_match_legacy_policy_behavior():
    backend = type(
        "Backend",
        (),
        {
            "objective_names": ["fel_energy"],
            "obj_pvnames": ["FEL:ENERGY"],
            "constraint_names": ["orbit_x"],
            "constraint_pvnames": ["BPM:01:X"],
            "constraint_bounds": [(-1.0, 1.0)],
        },
    )()

    fel_spec = POLICY_REGISTRY.expand_preset("objective", "fel_energy_guard")
    fel_rule = POLICY_REGISTRY.build("objective", fel_spec["name"], fel_spec["kwargs"])
    total = np.asarray([[2e6], [3e6], [4e6]])
    reduced = np.asarray([3e6])
    np.testing.assert_allclose(
        fel_rule.post_reduce(reduced, total, backend),
        FelEnergyGuardPolicy().post_reduce(reduced, total, backend),
    )

    bpm_spec = POLICY_REGISTRY.expand_preset("constraint", "bpm_guard")
    bpm_rule = POLICY_REGISTRY.build("constraint", bpm_spec["name"], bpm_spec["kwargs"])
    zeros = np.zeros((3, 1))
    np.testing.assert_allclose(
        bpm_rule.post_reduce(np.asarray([0.0]), zeros, backend),
        BPMGuardConstraintPolicy().post_reduce(np.asarray([0.0]), zeros, backend),
    )


def test_structured_rule_editor_loads_presets_without_raw_json(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication, QPlainTextEdit
    from gotacc.gui.views.tool_dialogs import SampleGuardRuleEditorDialog

    app = QApplication.instance() or QApplication([])
    spec = POLICY_REGISTRY.expand_preset("objective", "fel_energy_guard")
    dialog = SampleGuardRuleEditorDialog(
        kind="objective",
        target_names=["fel_energy", "beam_current"],
        policy_name=spec["name"],
        kwargs=spec["kwargs"],
        preset_name="fel_energy_guard",
    )
    try:
        assert dialog.findChildren(QPlainTextEdit) == []
        assert dialog.comboBox_preset.currentData() == "fel_energy_guard"
        assert dialog.tableWidget_conditions.rowCount() == 2
        state = dialog.rule_state()
        assert state["name"] == "sample_guard"
        assert state["kwargs"]["target"] == "fel_energy"
        assert state["kwargs"]["match"] == "any"
        assert state["kwargs"]["action"] == {"type": "replace", "value": 0.0}
        json.dumps(state["kwargs"])
    finally:
        dialog.close()
        app.processEvents()

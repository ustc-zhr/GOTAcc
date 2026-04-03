import importlib


MODULES = [
    "GOTAcc.src.gotacc",
    "GOTAcc.src.gotacc.cli.run_bo",
    "GOTAcc.src.gotacc.algorithms.single_objective.bo",
    "GOTAcc.src.gotacc.algorithms.single_objective.turbo",
    "GOTAcc.src.gotacc.algorithms.single_objective.rcds",
    "GOTAcc.src.gotacc.algorithms.multi_objective.mobo",
    "GOTAcc.src.gotacc.algorithms.multi_objective.mggpo",
    "GOTAcc.src.gotacc.algorithms.multi_objective.mopso",
    "GOTAcc.src.gotacc.algorithms.multi_objective.nsga2",
]


def test_import_core_modules():
    for module_name in MODULES:
        importlib.import_module(module_name)


test_import_core_modules()
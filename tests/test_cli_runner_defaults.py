from types import SimpleNamespace

import pytest

from gotacc.configs.schema import RuntimeConfig
from gotacc.runners.run_cli import apply_runtime_overrides, build_parser


@pytest.mark.parametrize(
    ("argv", "initial", "expected"),
    [
        (["--config", "unused.py"], False, False),
        (["--config", "unused.py", "--set-best"], False, True),
        (["--config", "unused.py", "--no-set-best"], True, False),
    ],
)
def test_cli_best_writeback_requires_or_honors_explicit_choice(
    argv,
    initial,
    expected,
):
    cfg = SimpleNamespace(runtime=RuntimeConfig(set_best=initial))
    args = build_parser().parse_args(argv)

    apply_runtime_overrides(cfg, args)

    assert cfg.runtime.set_best is expected

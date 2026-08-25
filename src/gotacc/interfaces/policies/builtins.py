from __future__ import annotations

from typing import Any, Mapping

from gotacc.interfaces.epics import (
    BPMGuardConstraintPolicy,
    EqualWritePolicy,
    FelEnergyGuardPolicy,
    ZeroGuardPolicy,
)

from .registry import PolicyDefinition, PolicyRegistry


def _build_equal_write(kwargs: Mapping[str, Any]) -> EqualWritePolicy:
    return EqualWritePolicy(extra_links=kwargs["pvlinks"])


def _build_fel_energy_guard(kwargs: Mapping[str, Any]) -> FelEnergyGuardPolicy:
    return FelEnergyGuardPolicy(
        target_col=kwargs["target_col"],
        large_threshold=kwargs["large_threshold"],
        change_threshold=kwargs["change_threshold"],
    )


def _build_zero_guard(kwargs: Mapping[str, Any]) -> ZeroGuardPolicy:
    return ZeroGuardPolicy(
        target_col=kwargs["target_col"],
        zero_atol=kwargs["zero_atol"],
        offset=kwargs["offset"],
    )


def _build_bpm_guard(kwargs: Mapping[str, Any]) -> BPMGuardConstraintPolicy:
    return BPMGuardConstraintPolicy(
        target_col=kwargs["target_col"],
        zero_atol=kwargs["zero_atol"],
        delta_ratio=kwargs["delta_ratio"],
        delta_min=kwargs["delta_min"],
        scale_floor=kwargs["scale_floor"],
    )


POLICY_REGISTRY = PolicyRegistry()

POLICY_REGISTRY.register(
    PolicyDefinition(
        name="equal",
        kind="write",
        aliases=("xiaosesan_symmetry",),
        default_kwargs={"pvlinks": None},
        factory=_build_equal_write,
        description="Write selected knob values to additional linked PVs.",
    )
)

POLICY_REGISTRY.register(
    PolicyDefinition(
        name="fel_energy_guard",
        kind="objective",
        default_kwargs={
            "target_col": 0,
            "large_threshold": 1e6,
            "change_threshold": 1e-6,
        },
        factory=_build_fel_energy_guard,
        description="Replace abnormal or nearly constant FEL energy samples.",
        is_default=True,
    )
)

POLICY_REGISTRY.register(
    PolicyDefinition(
        name="zero_guard",
        kind="objective",
        aliases=("xiaosesan_zero_guard",),
        default_kwargs={
            "target_col": 1,
            "zero_atol": 1e-12,
            "offset": 100.0,
        },
        factory=_build_zero_guard,
        description="Add an offset when a reduced objective is effectively zero.",
    )
)

POLICY_REGISTRY.register(
    PolicyDefinition(
        name="bpm_guard",
        kind="constraint",
        aliases=("bpm_zero_guard",),
        default_kwargs={
            "target_col": 0,
            "zero_atol": 1e-9,
            "delta_ratio": 0.1,
            "delta_min": 1e-6,
            "scale_floor": 1.0,
        },
        factory=_build_bpm_guard,
        description="Treat all-zero BPM constraint samples as infeasible.",
        is_default=True,
    )
)

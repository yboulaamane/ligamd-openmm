"""
test_stage_boundaries.py

Regression tests for the nstlim/gamd_production argument bug.

Bug (fork regression vs MiaoLab20/gamd-openmm): gamdSimulation.py passed
config.integrator.number_of_steps.gamd_production as the nstlim argument to
GamdIntegratorFactory.get_integrator(), but nstlim must be the *total*
simulation length (ntcmd + nteb + gamd_production).

Passing gamd_production sets stage_5_end = gamd_production inside
GamdStageIntegrator.__init__, but the Runner steps to
ntcmd + nteb + gamd_production steps total.  Every step beyond
gamd_production falls outside all five stage conditionals, so boost
parameters stop updating and dynamics silently degenerate — no exception
is raised and reporters keep writing near-identical frames.

Scaled reproduction: halving all step counts moves the freeze to exactly
half the production count, confirming the bug tracks gamd_production
rather than any absolute threshold.
"""

import ast
import os

import pytest
import openmm.unit as unit

from gamd.langevin.dual_boost_integrators import LowerBoundIntegrator


# Small but structurally valid step counts.
# ntcmd and nteb must each be >= ntave and a multiple of ntave.
NTCMDPREP = 200
NTCMD = 1000
NTEBPREP = 200
NTEB = 1000
NTAVE = 50


def _make_integrator(gamd_production):
    """Build LowerBoundIntegrator with the correct nstlim = total steps."""
    total = NTCMD + NTEB + gamd_production
    return LowerBoundIntegrator(
        group=0,
        ntcmdprep=NTCMDPREP,
        ntcmd=NTCMD,
        ntebprep=NTEBPREP,
        nteb=NTEB,
        nstlim=total,
        ntave=NTAVE,
    )


def _make_integrator_buggy(gamd_production):
    """Build LowerBoundIntegrator with the buggy nstlim = gamd_production only."""
    return LowerBoundIntegrator(
        group=0,
        ntcmdprep=NTCMDPREP,
        ntcmd=NTCMD,
        ntebprep=NTEBPREP,
        nteb=NTEB,
        nstlim=gamd_production,  # bug: should be ntcmd + nteb + gamd_production
        ntave=NTAVE,
    )


@pytest.mark.parametrize("gamd_production", [500, 5_000, 250_000_000])
class TestStageBoundaries:

    def test_stage5_end_equals_nstlim(self, gamd_production):
        """(a) stage_5_end must equal nstlim when built with total_simulation_length."""
        integrator = _make_integrator(gamd_production)
        assert integrator.stage_5_end == integrator.nstlim

    def test_buggy_nstlim_leaves_ntcmd_plus_nteb_steps_uncovered(
        self, gamd_production
    ):
        """
        (b) Passing gamd_production as nstlim leaves exactly ntcmd + nteb
        steps beyond stage_5_end — the freeze window observed in production runs.
        """
        integrator = _make_integrator_buggy(gamd_production)
        true_total = NTCMD + NTEB + gamd_production
        # stage_5_end was set to gamd_production (the wrong nstlim)
        uncovered = true_total - integrator.stage_5_end
        assert uncovered == NTCMD + NTEB

    def test_stages_tile_run_contiguously(self, gamd_production):
        """
        (c) The five stage spans tile [1, nstlim] with no gaps or overlaps.

        Adjacency: each stage must start immediately after the previous ends.
        Coverage: sum of all stage lengths must equal nstlim.
        """
        integrator = _make_integrator(gamd_production)

        # Adjacent boundaries must be consecutive
        assert integrator.stage_2_start == integrator.stage_1_end + 1
        assert integrator.stage_3_start == integrator.stage_2_end + 1
        assert integrator.stage_4_start == integrator.stage_3_end + 1
        assert integrator.stage_5_start == integrator.stage_4_end + 1

        # Stage 5 must reach the very last step
        assert integrator.stage_5_end == integrator.nstlim

        # Stage 1 activates on steps 1..stage_1_end (stepCount is incremented
        # before the stage checks, so stage_1_start=0 is never seen).
        stage1_len = integrator.stage_1_end  # steps 1 to stage_1_end
        stage2_len = integrator.stage_2_end - integrator.stage_2_start + 1
        stage3_len = integrator.stage_3_end - integrator.stage_3_start + 1
        stage4_len = integrator.stage_4_end - integrator.stage_4_start + 1
        stage5_len = integrator.stage_5_end - integrator.stage_5_start + 1

        covered = stage1_len + stage2_len + stage3_len + stage4_len + stage5_len
        assert covered == integrator.nstlim


# ---------------------------------------------------------------------------
# Call-site guard: static analysis of gamdSimulation.py
# ---------------------------------------------------------------------------

_SIM_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "gamdSimulation.py")
)


def test_get_integrator_call_uses_total_simulation_length():
    """
    (e) Guard the actual call site: get_integrator must receive
    total_simulation_length as its nstlim positional argument (index 8),
    never gamd_production.

    Argument order: boost_type, system, temperature, dt,
                    ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave, ...
    """
    with open(_SIM_PATH) as fh:
        source = fh.read()

    tree = ast.parse(source)

    found_call = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "get_integrator"):
            continue
        found_call = True

        assert len(node.args) >= 9, (
            "get_integrator call has fewer than 9 positional args — "
            "call-site structure changed; update this test."
        )

        nstlim_arg = node.args[8]  # 9th positional arg = nstlim

        assert isinstance(nstlim_arg, ast.Attribute), (
            f"nstlim argument (index 8) should be an attribute access, "
            f"got {type(nstlim_arg).__name__!r}"
        )
        assert nstlim_arg.attr == "total_simulation_length", (
            f"nstlim argument must be 'total_simulation_length', "
            f"got {nstlim_arg.attr!r}.  "
            f"Passing 'gamd_production' instead silently freezes boost "
            f"parameter updates for the last ntcmd+nteb steps of every run."
        )

    assert found_call, (
        "Could not find a get_integrator() call in gamdSimulation.py — "
        "update this test if the call was moved or renamed."
    )

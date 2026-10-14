"""
A boosted force group whose energy never varies (Vmax == Vmin) must not
produce NaN. Previously ForceScalingFactor = 1 - k0*(E-V)/(Vmax-Vmin) was 0/0,
and the arithmetic reset to 1.0 kept the NaN (0*NaN = NaN), which then
poisoned the velocities of every atom. The dual LiGaMD integrators hit this on
any system where group 2 (dihedrals) is empty.
"""

import numpy as np
import openmm
import openmm.unit as unit
import pytest

from gamd.langevin.ligand_boost_integrators import (
    LowerBoundDualLigandIntegrator, UpperBoundDualLigandIntegrator)


def _system_with_constant_group_2():
    system = openmm.System()
    for _ in range(3):
        system.addParticle(39.9)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.3, 1000.0)
    bonds.addBond(1, 2, 0.3, 1000.0)
    bonds.setForceGroup(1)
    system.addForce(bonds)
    empty = openmm.CustomExternalForce("0")
    empty.addParticle(0, [])
    empty.setForceGroup(2)
    system.addForce(empty)
    positions = [openmm.Vec3(0, 0, 0), openmm.Vec3(0.32, 0, 0), openmm.Vec3(0.6, 0.05, 0)]
    return system, positions


@pytest.mark.parametrize("integrator_class",
                         [LowerBoundDualLigandIntegrator, UpperBoundDualLigandIntegrator])
def test_constant_group_energy_does_not_produce_nan(integrator_class):
    system, positions = _system_with_constant_group_2()
    integrator = integrator_class(dt=1.0 * unit.femtoseconds, ntcmdprep=100, ntcmd=200,
                                  ntebprep=100, nteb=200, nstlim=600, ntave=50)
    integrator.setRandomNumberSeed(1)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1)

    integrator.step(600)

    state = context.getState(getPositions=True, getVelocities=True)
    assert np.isfinite(state.getPositions(asNumpy=True)._value).all()
    assert np.isfinite(state.getVelocities(asNumpy=True)._value).all()
    for i in range(integrator.getNumGlobalVariables()):
        name = integrator.getGlobalVariableName(i)
        assert np.isfinite(integrator.getGlobalVariableByName(name)), name
    assert integrator.getGlobalVariableByName("ForceScalingFactor_Dihedral") == 1.0

from types import SimpleNamespace

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import pytest
from mdtraj.formats import DCDTrajectoryFile

from gamd.runners import Runner
from gamd.utils import SubsetDCDReporter

SOLUTE_ATOMS = [0, 1]
N_ATOMS = 5
BOX_NM = 3.0


def make_simulation():
    """Two ligand atoms and one water, free of forces so they drift steadily."""
    topology = app.Topology()
    topology.setPeriodicBoxVectors(np.eye(3) * BOX_NM * unit.nanometer)
    chain = topology.addChain()
    ligand = topology.addResidue("LIG", chain)
    water = topology.addResidue("WAT", chain)
    for residue, count in ((ligand, 2), (water, 3)):
        for index in range(count):
            topology.addAtom(f"A{index}", app.Element.getBySymbol("C"), residue)

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*(np.eye(3) * BOX_NM))
    for _ in range(N_ATOMS):
        system.addParticle(12.0)

    simulation = app.Simulation(
        topology, system, openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"))
    positions = np.arange(N_ATOMS * 3, dtype=float).reshape(N_ATOMS, 3) * 0.1
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(np.ones((N_ATOMS, 3)))
    return simulation


def expected_solute_angstrom(simulation):
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    return positions[SOLUTE_ATOMS]


def read_dcd(path):
    with DCDTrajectoryFile(str(path)) as dcd:
        return dcd.read()[0]


def test_writes_only_the_chosen_atoms(tmp_path):
    path = tmp_path / "output.dcd"
    simulation = make_simulation()
    simulation.reporters.append(SubsetDCDReporter(str(path), 5, SOLUTE_ATOMS))
    simulation.step(10)
    expected = expected_solute_angstrom(simulation)
    del simulation.reporters[:]

    xyz = read_dcd(path)
    assert xyz.shape == (2, len(SOLUTE_ATOMS), 3)
    assert xyz[-1] == pytest.approx(expected, abs=1e-3)


def test_append_continues_the_same_file(tmp_path):
    path = tmp_path / "output.dcd"
    simulation = make_simulation()
    first = SubsetDCDReporter(str(path), 5, SOLUTE_ATOMS)
    simulation.reporters.append(first)
    simulation.step(10)
    simulation.reporters.remove(first)
    first._out.close()

    simulation.reporters.append(
        SubsetDCDReporter(str(path), 5, SOLUTE_ATOMS, append=True))
    simulation.step(10)
    expected = expected_solute_angstrom(simulation)
    del simulation.reporters[:]

    xyz = read_dcd(path)
    assert xyz.shape == (4, len(SOLUTE_ATOMS), 3)
    assert xyz[-1] == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("env_value, expected_atoms", [("1", 2), (None, N_ATOMS)])
def test_runner_follows_the_environment_variable(
        tmp_path, monkeypatch, env_value, expected_atoms):
    if env_value is None:
        monkeypatch.delenv("GAMD_SOLUTE_ONLY", raising=False)
    else:
        monkeypatch.setenv("GAMD_SOLUTE_ONLY", env_value)

    simulation = make_simulation()
    runner = Runner.__new__(Runner)
    runner.gamd_simulation = SimpleNamespace(
        simulation=simulation, traj_reporter=app.DCDReporter)
    runner.config = SimpleNamespace(outputs=SimpleNamespace(
        directory=str(tmp_path),
        reporting=SimpleNamespace(
            coordinates_file_type="dcd", coordinates_interval=5)))

    runner.register_trajectory_reporter(restart=False)
    simulation.step(5)
    del simulation.reporters[:]

    assert read_dcd(tmp_path / "output.dcd").shape == (1, expected_atoms, 3)

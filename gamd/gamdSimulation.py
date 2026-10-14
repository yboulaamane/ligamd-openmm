import os
import mdtraj
import parmed
import openmm as openmm
import openmm.app as openmm_app
import openmm.unit as unit
from openmm import XmlSerializer
from openmm.app import PDBFile
import copy
import math

from gamd import parser
from gamd.integrator_factory import *

def load_pdb_positions_and_box_vectors(pdb_coords_filename, need_box):
    try:
        positions = openmm_app.PDBxFile(pdb_coords_filename)
    except:
        positions = openmm_app.PDBFile(pdb_coords_filename)
    pdb_parmed = parmed.load_file(pdb_coords_filename)
    if need_box:
        assert pdb_parmed.box_vectors is not None, "No box vectors found in {}.".format(pdb_coords_filename)
    return positions, pdb_parmed.box_vectors


def separate_ligand_interactions(system, topology, timask1):
    """
    Expose the ligand's nonbonded interaction energy E_L (ligand atoms with
    every atom, defined by timask1) as Force Group 1 so it can be boosted.

    The main NonbondedForce is left untouched, so the unboosted potential is
    exactly the original one (including PME reciprocal space). E_L is added in
    group 1 and the same term with opposite sign is added in group 0, which
    the integrator never boosts; the two cancel in the total energy.
    """
    print(f"LiGaMD: Isolating ligand atoms using mask: {timask1}")

    md_top = mdtraj.Topology.from_openmm(topology)
    ligand_indices = md_top.select(timask1)
    if len(ligand_indices) == 0:
        raise ValueError(f"LiGaMD Error: No atoms found for mask '{timask1}'")
    ligand_set = set(int(i) for i in ligand_indices)
    all_indices = set(range(system.getNumParticles()))

    nb_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nb_force = force
            break
    if nb_force is None:
        raise Exception("LiGaMD Error: System does not have a NonbondedForce.")

    # E_L uses the direct-space Coulomb kernel of the main force's method, so the
    # boosted energy tracks what NonbondedForce computes. A CustomNonbondedForce
    # cannot do reciprocal space; that part stays in the main force, unboosted.
    method = nb_force.getNonbondedMethod()
    if method in (openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald, openmm.NonbondedForce.LJPME):
        alpha, _, _, _ = nb_force.getPMEParameters()
        alpha = alpha.value_in_unit(unit.nanometer**-1)
        if alpha == 0.0:
            # alpha left at "auto": derive it the same way OpenMM does
            tol = nb_force.getEwaldErrorTolerance()
            alpha = math.sqrt(-math.log(2.0 * tol)) / nb_force.getCutoffDistance().value_in_unit(unit.nanometer)
        coulomb = f"erfc({alpha}*r)/r"
    elif method in (openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.CutoffNonPeriodic):
        # reaction field, matching what NonbondedForce uses for these methods
        rc = nb_force.getCutoffDistance().value_in_unit(unit.nanometer)
        eps = nb_force.getReactionFieldDielectric()
        krf = (eps - 1) / ((2 * eps + 1) * rc**3)
        crf = 3 * eps / ((2 * eps + 1) * rc)
        coulomb = f"(1/r + {krf}*r^2 - {crf})"
    else:
        coulomb = "1/r"

    if method in (openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald,
                  openmm.NonbondedForce.LJPME, openmm.NonbondedForce.CutoffPeriodic):
        custom_method = openmm.CustomNonbondedForce.CutoffPeriodic
    elif method == openmm.NonbondedForce.NoCutoff:
        custom_method = openmm.CustomNonbondedForce.NoCutoff
    else:
        custom_method = openmm.CustomNonbondedForce.CutoffNonPeriodic

    particle_params = [nb_force.getParticleParameters(i) for i in range(system.getNumParticles())]
    # 1-2/1-3/1-4 pairs are excluded here and stay in the main force's exceptions.
    exclusions = [nb_force.getExceptionParameters(i)[:2] for i in range(nb_force.getNumExceptions())]

    def make_ligand_force(sign, group):
        force = openmm.CustomNonbondedForce(f"""
            {sign}*(4*epsilon*((sigma/r)^12-(sigma/r)^6) + 138.935456*charge1*charge2*{coulomb});
            sigma=0.5*(sigma1+sigma2);
            epsilon=sqrt(epsilon1*epsilon2);
        """)
        force.addPerParticleParameter("charge")
        force.addPerParticleParameter("sigma")
        force.addPerParticleParameter("epsilon")
        for charge, sigma, epsilon in particle_params:
            force.addParticle([charge, sigma, epsilon])
        for p1, p2 in exclusions:
            force.addExclusion(p1, p2)
        force.addInteractionGroup(ligand_set, all_indices)
        force.setNonbondedMethod(custom_method)
        force.setCutoffDistance(nb_force.getCutoffDistance())
        if nb_force.getUseSwitchingFunction():
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(nb_force.getSwitchingDistance())
        force.setForceGroup(group)
        system.addForce(force)

    make_ligand_force(1, group=1)
    make_ligand_force(-1, group=0)

    print(f"LiGaMD: Exposed nonbonded energy of {len(ligand_indices)} ligand atoms as Force Group 1.")
    return system


class GamdSimulation:
    def __init__(self):
        self.system = None
        self.integrator = None
        self.simulation = None
        self.traj_reporter = None
        self.first_boost_group = None
        self.second_boost_group = None
        self.first_boost_type = None
        self.second_boost_type = None
        self.topology = None
        self.positions = None
        self.box_vectors = None
        self.platform = "CUDA"
        self.device_index = 0


class GamdSimulationFactory:
    def __init__(self):
        return

    def createGamdSimulation(self, config, platform_name, device_index):
        # ... (Same initialization logic as original until System Creation) ...
        # Copying standard init logic for brevity, insert full original code here if needed
        # Assuming the standard factory logic for reading files is here:
        
        need_box = True
        # [Standard Nonbonded Setup Code from original file]
        if config.system.nonbonded_method == "pme":
            nonbondedMethod = openmm_app.PME
        elif config.system.nonbonded_method == "nocutoff":
            nonbondedMethod = openmm_app.NoCutoff
            need_box = False
        elif config.system.nonbonded_method == "cutoffnonperiodic":
            nonbondedMethod = openmm_app.CutoffNonPeriodic
        elif config.system.nonbonded_method == "cutoffperiodic":
            nonbondedMethod = openmm_app.CutoffPeriodic
        elif config.system.nonbonded_method == "ewald":
            nonbondedMethod = openmm_app.Ewald
        else:
            raise Exception("nonbonded method not found")

        if config.system.constraints == "none" or config.system.constraints is None:
            constraints = None
        elif config.system.constraints == "hbonds":
            constraints = openmm_app.HBonds
        elif config.system.constraints == "allbonds":
            constraints = openmm_app.AllBonds
        elif config.system.constraints == "hangles":
            constraints = openmm_app.HAngles
        else:
            raise Exception("constraints not found")

        gamdSimulation = GamdSimulation()
        gamdSimulation.topology = None
        gamdSimulation.positions = None
        gamdSimulation.box_vectors = None

        # --- LOAD FILES (Standard) ---
        if config.input_files.amber is not None:
            prmtop = openmm_app.AmberPrmtopFile(config.input_files.amber.topology)
            topology = prmtop
            if config.input_files.amber.coordinates_filetype in ["inpcrd", "rst7"]:
                positions = openmm_app.AmberInpcrdFile(config.input_files.amber.coordinates)
                box_vectors = positions.boxVectors
            elif config.input_files.amber.coordinates_filetype == "pdb":
                pdb_coords_filename = config.input_files.amber.coordinates
                positions, box_vectors = load_pdb_positions_and_box_vectors(pdb_coords_filename, need_box)
            else:
                raise Exception("Invalid input type")
            gamdSimulation.system = prmtop.createSystem(
                nonbondedMethod=nonbondedMethod,
                nonbondedCutoff=config.system.nonbonded_cutoff,
                constraints=constraints)
            gamdSimulation.topology = topology.topology
            gamdSimulation.positions = positions.positions
            gamdSimulation.box_vectors = box_vectors
            
        elif hasattr(config.input_files, "openmm"):
            # OpenMM XML support
            system_file = config.input_files.openmm.system
            state_file = config.input_files.openmm.state
            with open(system_file, 'r') as sf:
                gamdSimulation.system = XmlSerializer.deserialize(sf.read())
            with open(state_file, 'r') as stf:
                state = XmlSerializer.deserialize(stf.read())
            gamdSimulation.topology = PDBFile(config.input_files.openmm.topology).topology
            gamdSimulation.positions = state.getPositions()
            gamdSimulation.box_vectors = state.getPeriodicBoxVectors()

        elif config.input_files.charmm is not None:
            # Charmm support (keeping original logic)
            psf = openmm_app.CharmmPsfFile(config.input_files.charmm.topology)
            if config.input_files.charmm.coordinates_filetype == "crd":
                positions = openmm_app.CharmmCrdFile(config.input_files.charmm.coordinates)
            elif config.input_files.charmm.coordinates_filetype == "pdb":
                positions = openmm_app.PDBFile(config.input_files.charmm.coordinates)
            if config.input_files.charmm.is_config_box_vector_defined:
                psf.setBox(*config.input_files.charmm.box_vectors)
            params = openmm_app.CharmmParameterSet(*config.input_files.charmm.parameters)
            topology = psf
            gamdSimulation.system = psf.createSystem(
                params=params,
                nonbondedMethod=nonbondedMethod,
                nonbondedCutoff=config.system.nonbonded_cutoff,
                switchDistance=config.system.switch_distance,
                ewaldErrorTolerance = config.system.ewald_error_tolerance,
                constraints=constraints)
            gamdSimulation.topology = topology.topology
            gamdSimulation.positions = positions.positions
            if hasattr(positions, 'boxVectors'):
                gamdSimulation.box_vectors = positions.boxVectors
            else:
                gamdSimulation.box_vectors = None

        # --- LIGAMD FORCE SPLITTING ---
        # If LiGaMD is requested, we must modify the system structure BEFORE creating the integrator
        boost_type = config.integrator.boost_type
        if boost_type in ["ligamd", "ligamd-dual", "lower-ligand", "upper-ligand", "lower-dual-ligand", "upper-dual-ligand", "upper-ligamd", "upper-ligamd-dual"]:
            if not hasattr(config.integrator, "timask1") or config.integrator.timask1 is None:
                raise Exception("LiGaMD Error: 'timask1' must be defined in the XML integrator section.")
            
            gamdSimulation.system = separate_ligand_interactions(
                gamdSimulation.system,
                gamdSimulation.topology,
                config.integrator.timask1
            )

        # --- INTEGRATOR CREATION ---
        if config.integrator.algorithm == "langevin":
            gamdIntegratorFactory = GamdIntegratorFactory()
            result = gamdIntegratorFactory.get_integrator(
                config.integrator.boost_type, gamdSimulation.system, config.temperature,
                config.integrator.dt,
                config.integrator.number_of_steps.conventional_md_prep,
                config.integrator.number_of_steps.conventional_md,
                config.integrator.number_of_steps.gamd_equilibration_prep,
                config.integrator.number_of_steps.gamd_equilibration,
                config.integrator.number_of_steps.total_simulation_length,
                config.integrator.number_of_steps.averaging_window_interval,
                sigma0p=config.integrator.sigma0.primary,
                sigma0d=config.integrator.sigma0.secondary)
            
            [gamdSimulation.first_boost_group,
             gamdSimulation.second_boost_group,
             integrator, gamdSimulation.first_boost_type,
             gamdSimulation.second_boost_type] = result
            
            integrator.setRandomNumberSeed(config.integrator.random_seed)
            integrator.setFriction(config.integrator.friction_coefficient)
            gamdSimulation.integrator = integrator

        else:
            raise Exception("Algorithm not implemented: {}".format(config.integrator.algorithm))

        # --- BAROSTAT & PLATFORM ---
        if config.barostat is not None:
            barostat = openmm.MonteCarloBarostat(
                config.barostat.pressure,
                config.temperature,
                config.barostat.frequency)
            gamdSimulation.system.addForce(barostat)

        properties = {}
        user_platform_name = platform_name.lower()
        if user_platform_name == "cuda":
            platform = openmm.Platform.getPlatformByName('CUDA')
            properties['CudaPrecision'] = 'mixed'
            properties['DeviceIndex'] = device_index
            gamdSimulation.simulation = openmm_app.Simulation(
                gamdSimulation.topology, gamdSimulation.system,
                gamdSimulation.integrator, platform, properties)
        elif user_platform_name == "opencl":
            platform = openmm.Platform.getPlatformByName('OpenCL')
            properties['DeviceIndex'] = device_index
            gamdSimulation.simulation = openmm_app.Simulation(
                gamdSimulation.topology, gamdSimulation.system,
                gamdSimulation.integrator, platform, properties)
        else:
            platform = openmm.Platform.getPlatformByName(platform_name)
            gamdSimulation.simulation = openmm_app.Simulation(
                gamdSimulation.topology, gamdSimulation.system,
                gamdSimulation.integrator, platform)

        # Set positions
        gamdSimulation.simulation.context.setPositions(gamdSimulation.positions)
        if gamdSimulation.box_vectors is not None:
            gamdSimulation.simulation.context.setPeriodicBoxVectors(*gamdSimulation.box_vectors)
        if config.run_minimization:
            gamdSimulation.simulation.minimizeEnergy()
        gamdSimulation.simulation.context.setVelocitiesToTemperature(config.temperature)

        # Reporters
        if config.outputs.reporting.coordinates_file_type == "dcd":
            gamdSimulation.traj_reporter = openmm_app.DCDReporter
        elif config.outputs.reporting.coordinates_file_type == "pdb":
            gamdSimulation.traj_reporter = openmm_app.PDBReporter
        elif config.outputs.reporting.coordinates_file_type == "h5":
            gamdSimulation.traj_reporter = mdtraj.reporters.HDF5Reporter

        return gamdSimulation

"""
integrator_factory.py: Implements the GaMD integration method.

Portions copyright (c) 2021 University of Kansas
Authors: Matthew Copeland, Yinglong Miao
Contributors: Lane Votapka
"""

import openmm.unit as unit

from gamd.langevin.dihedral_boost_integrators import LowerBoundIntegrator as DihedralBoostLowerBoundIntegrator
from gamd.langevin.dihedral_boost_integrators import UpperBoundIntegrator as DihedralBoostUpperBoundIntegrator
from gamd.langevin.dual_boost_integrators import LowerBoundIntegrator as DualBoostLowerBoundIntegrator
from gamd.langevin.dual_boost_integrators import UpperBoundIntegrator as DualBoostUpperBoundIntegrator
from gamd.langevin.dual_non_bonded_dihedral_boost_integrators import \
    LowerBoundIntegrator as DualNonBondedDihedralLowerIntegrator
from gamd.langevin.dual_non_bonded_dihedral_boost_integrators import \
    UpperBoundIntegrator as DualNonBondedDihedralUpperIntegrator
from gamd.langevin.non_bonded_boost_integrators import LowerBoundIntegrator as NonBondedLowerBoundIntegrator
from gamd.langevin.non_bonded_boost_integrators import UpperBoundIntegrator as NonBondedUpperBoundIntegrator
from gamd.langevin.total_boost_integrators import LowerBoundIntegrator as TotalBoostLowerBoundIntegrator
from gamd.langevin.total_boost_integrators import UpperBoundIntegrator as TotalBoostUpperBoundIntegrator

# --- NEW LiGaMD IMPORTS ---
from gamd.langevin.ligand_boost_integrators import (
    LowerBoundLigandIntegrator, LowerBoundDualLigandIntegrator,
    UpperBoundLigandIntegrator, UpperBoundDualLigandIntegrator)
# --------------------------

from gamd.stage_integrator import BoostType


def print_force_group_information(system):
    for force in system.getForces():
        print("Force:  ", force)
        print("Force Group:  ", force.getForceGroup())


def set_all_forces_to_group(system, group):
    for force in system.getForces():
        force.setForceGroup(group)


def set_dihedral_group(system):
    return set_single_group(2, ['PeriodicTorsionForce', 'CMAPTorsionForce'], system)


def set_non_bonded_group(system):
    return set_single_group(1, ['NonbondedForce','CustomNonbondedForce'], system)


def set_single_group(group, name_list, system):
    for force in system.getForces():
        if force.__class__.__name__ in name_list:
            force.setForceGroup(group)
    return group


def create_gamd_cmd_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave):
    group = set_dihedral_group(system)
    integrator = DihedralBoostLowerBoundIntegrator(group, dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd,
                                                   ntebprep=ntebprep, nteb=nteb, nstlim=nstlim,
                                                   ntave=ntave, temperature=temperature,
                                                   sigma0=0.0 * unit.kilocalories_per_mole)
    result = ["", group, integrator]
    return result


def create_lower_total_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave,
                                        sigma0=6.0 * unit.kilocalories_per_mole):
    group = set_dihedral_group(system)
    integrator = TotalBoostLowerBoundIntegrator(dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd,
                                                ntebprep=ntebprep, nteb=nteb, nstlim=nstlim,
                                                ntave=ntave, sigma0=sigma0, temperature=temperature)
    result = ["", group, integrator]
    return result


def create_upper_total_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave,
                                        sigma0=6.0 * unit.kilocalories_per_mole):
    group = set_dihedral_group(system)
    integrator = TotalBoostUpperBoundIntegrator(dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd,
                                                ntebprep=ntebprep, nteb=nteb, nstlim=nstlim,
                                                ntave=ntave, sigma0=sigma0,
                                                temperature=temperature)
    result = ["", group, integrator]
    return result


def create_lower_dihedral_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave,
                                           sigma0=6.0 * unit.kilocalories_per_mole):
    group = set_dihedral_group(system)
    integrator = DihedralBoostLowerBoundIntegrator(group, dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd, ntebprep=ntebprep,
                                                   nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0=sigma0,
                                                   temperature=temperature)
    result = ["", group, integrator]
    return result


def create_upper_dihedral_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave,
                                           sigma0=6.0 * unit.kilocalories_per_mole):
    group = set_dihedral_group(system)
    integrator = DihedralBoostUpperBoundIntegrator(group, dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd, ntebprep=ntebprep,
                                                   nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0=sigma0,
                                                   temperature=temperature)
    result = ["", group, integrator]
    return result


def create_lower_dual_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave,
                                       sigma0p=6.0 * unit.kilocalories_per_mole,
                                       sigma0d=6.0 * unit.kilocalories_per_mole):
    group = set_dihedral_group(system)
    integrator = DualBoostLowerBoundIntegrator(group, dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd, ntebprep=ntebprep,
                                               nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0p=sigma0p,
                                               sigma0d=sigma0d, temperature=temperature)
    result = ["", group, integrator]
    return result


def create_upper_dual_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave,
                                       sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole):
    group = set_dihedral_group(system)
    integrator = DualBoostUpperBoundIntegrator(group, dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd, ntebprep=ntebprep,
                                               nteb=nteb, nstlim=nstlim, ntave=ntave,
                                               sigma0d=sigma0d,
                                               sigma0p=sigma0p, temperature=temperature)
    result = ["", group, integrator]
    return result


def create_lower_non_bonded_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave,
                                             sigma0=6.0 * unit.kilocalories_per_mole):
    group = set_non_bonded_group(system)
    integrator = NonBondedLowerBoundIntegrator(group, dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd, ntebprep=ntebprep,
                                               nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0=sigma0,
                                               temperature=temperature)
    result = ["", group, integrator]
    return result


def create_upper_non_bonded_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave,
                                             sigma0=6.0 * unit.kilocalories_per_mole):
    group = set_non_bonded_group(system)
    integrator = NonBondedUpperBoundIntegrator(group, dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd, ntebprep=ntebprep,
                                               nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0=sigma0,
                                               temperature=temperature)
    result = ["", group, integrator]
    return result


def create_lower_dual_non_bonded_dihederal_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                            ntebprep, nteb, nstlim, ntave,
                                                            sigma0p=6.0 * unit.kilocalories_per_mole,
                                                            sigma0d=6.0 * unit.kilocalories_per_mole):
    nonbonded_group = set_non_bonded_group(system)
    dihedral_group = set_dihedral_group(system)
    integrator = DualNonBondedDihedralLowerIntegrator(nonbonded_group, dihedral_group, dt=dt, ntcmdprep=ntcmdprep,
                                                      ntcmd=ntcmd, ntebprep=ntebprep,
                                                      nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0p=sigma0p,
                                                      sigma0d=sigma0d,
                                                      temperature=temperature)
    result = [nonbonded_group, dihedral_group, integrator]
    return result


def create_upper_dual_non_bonded_dihederal_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                            ntebprep, nteb, nstlim, ntave,
                                                            sigma0p=6.0 * unit.kilocalories_per_mole,
                                                            sigma0d=6.0 * unit.kilocalories_per_mole):
    nonbonded_group = set_non_bonded_group(system)
    dihedral_group = set_dihedral_group(system)
    integrator = DualNonBondedDihedralUpperIntegrator(nonbonded_group, dihedral_group, dt=dt, ntcmdprep=ntcmdprep,
                                                      ntcmd=ntcmd, ntebprep=ntebprep,
                                                      nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0p=sigma0p,
                                                      sigma0d=sigma0d,
                                                      temperature=temperature)
    result = [nonbonded_group, dihedral_group, integrator]
    return result


# --- NEW FACTORY FUNCTIONS FOR LiGaMD ---
def create_lower_ligand_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave, sigma0):
    # System already prepared by gamdSimulation to have Ligand in Group 1
    # We do NOT reset force groups here because gamdSimulation.py handled the splitting.
    integrator = LowerBoundLigandIntegrator(dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd, ntebprep=ntebprep,
                                            nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0=sigma0,
                                            temperature=temperature)
    # The boost is on Group 1 (Ligand)
    result = [1, "", integrator] 
    return result

def create_lower_dual_ligand_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave, sigma0p, sigma0d):
    
    # NEW LINE: Isolate Protein Dihedrals into Force Group 2
    set_dihedral_group(system)
    
    integrator = LowerBoundDualLigandIntegrator(dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd, ntebprep=ntebprep,
                                                nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0p=sigma0p, sigma0d=sigma0d,
                                                temperature=temperature)
    # Return Group 1 (Ligand) and Group 2 (Dihedral)
    result = [1, 2, integrator]
    return result
    
def create_upper_ligand_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave, sigma0p):
    integrator = UpperBoundLigandIntegrator(dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd, ntebprep=ntebprep,
                                            nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0=sigma0p,
                                            temperature=temperature)
    return [1, "", integrator]

def create_upper_dual_ligand_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave, sigma0p, sigma0d):
    # Isolate Protein Dihedrals into Force Group 2 safely
    set_dihedral_group(system)
    integrator = UpperBoundDualLigandIntegrator(dt=dt, ntcmdprep=ntcmdprep, ntcmd=ntcmd, ntebprep=ntebprep,
                                                nteb=nteb, nstlim=nstlim, ntave=ntave, sigma0p=sigma0p, sigma0d=sigma0d,
                                                temperature=temperature)
    return [1, 2, integrator]
# ----------------------------------------


class GamdIntegratorFactory:

    def __init__(self):
        pass

    @staticmethod
    def get_integrator(boost_type_str, system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim, ntave,
                       sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole):
        
        # --- IMPORTANT: Do NOT reset force groups if using LiGaMD ---
        # gamdSimulation.py has carefully split the system into Group 1 (Ligand) and Group 0 (Env).
        # We only reset to 0 if it is a standard simulation.
        if "ligand" not in boost_type_str and "ligamd" not in boost_type_str:
            set_all_forces_to_group(system, 0)
        
        result = []
        first_boost_type = BoostType.TOTAL
        second_boost_type = BoostType.DIHEDRAL
        
        if boost_type_str == "gamd-cmd-base":
            result = create_gamd_cmd_integrator(system, temperature, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim,
                                                ntave)
        elif boost_type_str == "lower-total":
            result = create_lower_total_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                         ntebprep, nteb, nstlim, ntave, sigma0p)
        elif boost_type_str == "upper-total":
            result = create_upper_total_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                         ntebprep, nteb, nstlim, ntave, sigma0p)
        elif boost_type_str == "lower-dihedral":
            result = create_lower_dihedral_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                            ntebprep, nteb, nstlim, ntave, sigma0p)
        elif boost_type_str == "upper-dihedral":
            result = create_upper_dihedral_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                            ntebprep, nteb, nstlim, ntave, sigma0p)
        elif boost_type_str == "lower-dual":
            result = create_lower_dual_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                        ntebprep, nteb, nstlim, ntave, sigma0p, sigma0d)
        elif boost_type_str == "upper-dual":
            result = create_upper_dual_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                        ntebprep, nteb, nstlim, ntave, sigma0p, sigma0d)
        elif boost_type_str == "lower-nonbonded":
            result = create_lower_non_bonded_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                              ntebprep, nteb, nstlim, ntave, sigma0p)
            second_boost_type = BoostType.NON_BONDED
        elif boost_type_str == "upper-nonbonded":
            result = create_upper_non_bonded_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                              ntebprep, nteb, nstlim, ntave, sigma0p)
            second_boost_type = BoostType.NON_BONDED
        elif boost_type_str == "lower-dual-nonbonded-dihedral":
            result = create_lower_dual_non_bonded_dihederal_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                                             ntebprep, nteb, nstlim, ntave, sigma0p,
                                                                             sigma0d)
            first_boost_type = BoostType.NON_BONDED
            second_boost_type = BoostType.DIHEDRAL
        elif boost_type_str == "upper-dual-nonbonded-dihedral":
            result = create_upper_dual_non_bonded_dihederal_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                                             ntebprep, nteb, nstlim, ntave, sigma0p,
                                                                             sigma0d)
            first_boost_type = BoostType.NON_BONDED
            second_boost_type = BoostType.DIHEDRAL

        # --- NEW LOGIC FOR LIGAMD ---
        elif boost_type_str in ["ligamd", "lower-ligand"]:
            result = create_lower_ligand_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                          ntebprep, nteb, nstlim, ntave, sigma0p)
            first_boost_type = BoostType.NON_BONDED # Repurposed as "Ligand"
            second_boost_type = BoostType.TOTAL # Unused placeholder

        elif boost_type_str in ["ligamd-dual", "lower-dual-ligand"]:
            result = create_lower_dual_ligand_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                               ntebprep, nteb, nstlim, ntave, sigma0p, sigma0d)
            first_boost_type = BoostType.NON_BONDED  # Ligand
            second_boost_type = BoostType.DIHEDRAL   # Protein

        elif boost_type_str in ["upper-ligamd", "upper-ligand"]:
            result = create_upper_ligand_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                          ntebprep, nteb, nstlim, ntave, sigma0p)
            first_boost_type = BoostType.NON_BONDED
            second_boost_type = BoostType.TOTAL

        elif boost_type_str in ["upper-ligamd-dual", "upper-dual-ligand"]:
            result = create_upper_dual_ligand_boost_integrator(system, temperature, dt, ntcmdprep, ntcmd,
                                                               ntebprep, nteb, nstlim, ntave, sigma0p, sigma0d)
            first_boost_type = BoostType.NON_BONDED
            second_boost_type = BoostType.DIHEDRAL
        # ----------------------------

        else:
            raise ValueError("Invalid boost_type_str passed to GamdIntegratorFactory.getIntegrator.")

        result.append(first_boost_type)
        result.append(second_boost_type)
        return result

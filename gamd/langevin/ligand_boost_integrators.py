from abc import ABC
import openmm.unit as unit
from gamd.langevin.base_integrator import GroupBoostIntegrator
from ..stage_integrator import BoostMethod
from ..stage_integrator import BoostType

class LigandBoostIntegrator(GroupBoostIntegrator, ABC):
    def __init__(self, group, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim,
                 ntave, sigma0, collision_rate, temperature, restart_filename):
        # FIX: Map "Ligand" to "NonBonded" to match the BoostType.NON_BONDED Enum value
        # This ensures variables are named "ForceScalingFactor_NonBonded" as expected by the kernel.
        group_dict = {group: "NonBonded"}
        
        super(LigandBoostIntegrator, self).__init__(group_dict,
                                                    BoostType.NON_BONDED, 
                                                    BoostMethod.GROUPS,
                                                    dt, ntcmdprep, ntcmd,
                                                    ntebprep, nteb, nstlim,
                                                    ntave, collision_rate,
                                                    temperature,
                                                    restart_filename)
        # Variable name must match the Enum suffix ("NonBonded")
        self.addGlobalVariable("sigma0_NonBonded", sigma0)

class DualLigandBoostIntegrator(GroupBoostIntegrator, ABC):
    def __init__(self, ligand_group, system_group, dt, ntcmdprep, ntcmd, ntebprep, nteb, nstlim,
                 ntave, sigma0_ligand, sigma0_system, collision_rate,
                 temperature, restart_filename):
        
        # FIX: Mapping Ligand -> NonBonded and System -> Dihedral
        # This allows us to use the existing "DUAL_NON_BONDED_DIHEDRAL" kernel infrastructure.
        # "Dihedral" here is just a label for the 2nd boost group (which is actually the whole system).
        group_dict = {ligand_group: "NonBonded", system_group: "Dihedral"}
        
        super(DualLigandBoostIntegrator, self).__init__(group_dict,
                                                        BoostType.DUAL_NON_BONDED_DIHEDRAL, 
                                                        BoostMethod.GROUPS,
                                                        dt, ntcmdprep, ntcmd,
                                                        ntebprep, nteb, nstlim,
                                                        ntave, collision_rate,
                                                        temperature,
                                                        restart_filename)
        
        self.addGlobalVariable("sigma0_NonBonded", sigma0_ligand)
        self.addGlobalVariable("sigma0_Dihedral", sigma0_system)

# --- Concrete Classes ---

class LowerBoundLigandIntegrator(LigandBoostIntegrator):
    def __init__(self, dt=2.0 * unit.femtoseconds, ntcmdprep=200000, ntcmd=1000000, 
                 ntebprep=200000, nteb=1000000, nstlim=3000000, ntave=50000, 
                 sigma0=6.0 * unit.kilocalories_per_mole,
                 collision_rate=1.0 / unit.picoseconds,
                 temperature=298.15 * unit.kelvin, restart_filename=None):
        super(LowerBoundLigandIntegrator, self).__init__(1, dt, ntcmdprep, ntcmd, ntebprep,
                                                         nteb, nstlim, ntave, sigma0,
                                                         collision_rate, temperature, restart_filename)
    def _calculate_threshold_energy_and_effective_harmonic_constant(self, compute_type):
        super()._lower_bound_calculate_threshold_energy_and_effective_harmonic_constant(compute_type)

class LowerBoundDualLigandIntegrator(DualLigandBoostIntegrator):
    def __init__(self, dt=2.0 * unit.femtoseconds, ntcmdprep=200000, ntcmd=1000000, 
                 ntebprep=200000, nteb=1000000, nstlim=3000000, ntave=50000, 
                 sigma0p=6.0 * unit.kilocalories_per_mole, # Ligand
                 sigma0d=6.0 * unit.kilocalories_per_mole, # System
                 collision_rate=1.0 / unit.picoseconds,
                 temperature=298.15 * unit.kelvin, restart_filename=None):
        # Group 1 = Ligand (labeled NonBonded), Group 0 = System (labeled Dihedral)
        super(LowerBoundDualLigandIntegrator, self).__init__(1, 2, dt, ntcmdprep, ntcmd, ntebprep,
                                                             nteb, nstlim, ntave, sigma0p, sigma0d,
                                                             collision_rate, temperature, restart_filename)
    def _calculate_threshold_energy_and_effective_harmonic_constant(self, compute_type):
        super()._lower_bound_calculate_threshold_energy_and_effective_harmonic_constant(compute_type)
        
        
class UpperBoundLigandIntegrator(LigandBoostIntegrator):
    def __init__(self, dt=2.0 * unit.femtoseconds, ntcmdprep=200000, ntcmd=1000000, 
                 ntebprep=200000, nteb=1000000, nstlim=3000000, ntave=50000, 
                 sigma0=6.0 * unit.kilocalories_per_mole,
                 collision_rate=1.0 / unit.picoseconds,
                 temperature=298.15 * unit.kelvin, restart_filename=None):
        super(UpperBoundLigandIntegrator, self).__init__(1, dt, ntcmdprep, ntcmd, ntebprep,
                                                         nteb, nstlim, ntave, sigma0,
                                                         collision_rate, temperature, restart_filename)
    def _calculate_threshold_energy_and_effective_harmonic_constant(self, compute_type):
        # NOTE: Using the upper_bound method here
        super()._upper_bound_calculate_threshold_energy_and_effective_harmonic_constant(compute_type)


class UpperBoundDualLigandIntegrator(DualLigandBoostIntegrator):
    def __init__(self, dt=2.0 * unit.femtoseconds, ntcmdprep=200000, ntcmd=1000000, 
                 ntebprep=200000, nteb=1000000, nstlim=3000000, ntave=50000, 
                 sigma0p=6.0 * unit.kilocalories_per_mole, # Ligand
                 sigma0d=6.0 * unit.kilocalories_per_mole, # Protein Dihedrals
                 collision_rate=1.0 / unit.picoseconds,
                 temperature=298.15 * unit.kelvin, restart_filename=None):
        # Group 1 = Ligand (NonBonded), Group 2 = Protein Dihedrals
        super(UpperBoundDualLigandIntegrator, self).__init__(1, 2, dt, ntcmdprep, ntcmd, ntebprep,
                                                             nteb, nstlim, ntave, sigma0p, sigma0d,
                                                             collision_rate, temperature, restart_filename)
    def _calculate_threshold_energy_and_effective_harmonic_constant(self, compute_type):
        # NOTE: Using the upper_bound method here
        super()._upper_bound_calculate_threshold_energy_and_effective_harmonic_constant(compute_type)

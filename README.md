# LiGaMD-OpenMM: Ligand-Gaussian Accelerated Molecular Dynamics

> **Notice**  
> This repository is a modified fork of the original `gamd-openmm` developed by the Miao Lab.  
> It has been specifically engineered to support **LiGaMD (Ligand-Gaussian Accelerated Molecular Dynamics)**, enabling isolated boosting of ligand interactions and targeted dual-boost systems for enhanced sampling of ligand binding and unbinding pathways.

---

## Overview

Gaussian Accelerated Molecular Dynamics (GaMD) is a biomolecular enhanced sampling method that works by adding a harmonic boost potential to smooth the system potential energy surface. By constructing a boost potential that follows a Gaussian distribution, accurate reweighting of GaMD simulations is achieved using cumulant expansion to the second order.

GaMD has been demonstrated on three biomolecular model systems:

- Alanine dipeptide  
- A set of three RNA tetraloops  
- The ligand rbt203 binding to HIV-1 Tar RNA  

Without the need to set predefined reaction coordinates, GaMD enables unconstrained enhanced sampling of biomolecules. Furthermore, the free energy profiles obtained from reweighting allow identification of distinct low-energy states and quantitative characterization of protein folding and ligand binding pathways.

---

## LiGaMD Extension

With the addition of **LiGaMD**, this module allows researchers to:

- Selectively apply boost potentials to the **non-bonded interactions of a bound ligand**
- Pair ligand boosting with secondary boosts (e.g., protein dihedrals)
- Open binding pockets without risking solvent instability
- Enhance sampling of ligand binding and unbinding events

---

## Installation

### 1. Install Anaconda (Python 3.x)

Download and install Anaconda (Python 3.x).

### 2. Install OpenMM

Follow instructions in the **OpenMM User Guide – Section 2.2 Installing OpenMM**.

### 3. Install AmberTools (for post-MD analysis)

```bash
conda install -c conda-forge ambertools
```

### 4. Install PyReweighting

Clone the PyReweighting repository:

```bash
git clone https://github.com/MiaoLab20/PyReweighting.git
```

> **Note:**  
> If developing the GaMD module and using test scripts, ensure the PyReweighting directory is added to your `PATH`.

### 5. Clone and Install LiGaMD-OpenMM

```bash
git clone https://github.com/yboulaamane/ligamd-openmm.git
cd ligamd-openmm
pip install -e .
```

You can either:

- Copy `gamdRunner` into your user `bin` directory, or  
- Add the `ligamd-openmm` directory to your `PATH`

---

## Testing (Optional)

To run tests:

```bash
pytest gamd/tests
```

---

## Running Simulations

Run GaMD/LiGaMD using your configuration file:

```bash
gamdRunner xml configuration-file.xml
```

An example repository is available:

### gamd-openmm-examples

This contains:

- Example data files  
- Example configuration files  
- Validation instructions  
- Command-line usage guidance  

> **Note:**  
> `gamdRunner` currently supports running conventional MD, equilibration, and production stages as part of a single execution.

---

## Running LiGaMD

### 1. Choose a boost type

Set `<boost-type>` in the `<integrator>` block to one of:

| `<boost-type>` | Alias | What is boosted |
|---|---|---|
| `ligamd` | `lower-ligand` | Ligand non-bonded energy, lower-bound threshold |
| `upper-ligamd` | `upper-ligand` | Ligand non-bonded energy, upper-bound threshold |
| `ligamd-dual` | `lower-dual-ligand` | Ligand non-bonded energy + all dihedrals, lower bound |
| `upper-ligamd-dual` | `upper-dual-ligand` | Ligand non-bonded energy + all dihedrals, upper bound |

The dual modes are usually the better choice for unbinding. The dihedral boost lets the protein open exit channels, and the ligand boost loosens the ligand's contacts.

### 2. Select the ligand with `<timask1>`

`<timask1>` is required for every LiGaMD boost type. It is an MDTraj atom-selection string, **not** an Amber mask:

```xml
<timask1>resname LIG</timask1>          <!-- not :LIG -->
```

The ligand's non-bonded interaction energy with every atom in the system (LJ + direct-space electrostatics) becomes the boosted energy. The unboosted potential is exactly that of the original force field, including PME reciprocal space.

### 3. Set `sigma0`

- `<primary>`: upper limit of the ligand boost's standard deviation (kcal/mol).
- `<secondary>`: upper limit of the dihedral boost's standard deviation (kcal/mol). Dual modes only; ignored otherwise.

Larger values give stronger acceleration but noisier reweighting.

### 4. Example configuration

```xml
<?xml version="1.0" ?>
<gamd>
    <temperature>310.0</temperature>                  <!-- K -->

    <system>
        <nonbonded-method>PME</nonbonded-method>
        <nonbonded-cutoff>1.0</nonbonded-cutoff>      <!-- nm -->
        <constraints>HBonds</constraints>
    </system>

    <barostat>
        <pressure>1.0</pressure>                      <!-- bar -->
        <frequency>25</frequency>
    </barostat>

    <run-minimization>True</run-minimization>

    <integrator>
        <algorithm>langevin</algorithm>
        <boost-type>ligamd-dual</boost-type>
        <timask1>resname LIG</timask1>
        <sigma0>
            <primary>6.0</primary>                    <!-- ligand, kcal/mol -->
            <secondary>6.0</secondary>                <!-- dihedrals, kcal/mol -->
        </sigma0>
        <random-seed>0</random-seed>
        <dt>0.002</dt>                                <!-- ps -->
        <friction-coefficient>1.0</friction-coefficient>  <!-- 1/ps -->
        <number-of-steps>
            <conventional-md-prep>200000</conventional-md-prep>
            <conventional-md>1000000</conventional-md>
            <gamd-equilibration-prep>200000</gamd-equilibration-prep>
            <gamd-equilibration>1000000</gamd-equilibration>
            <gamd-production>50000000</gamd-production>
            <averaging-window-interval>50000</averaging-window-interval>
        </number-of-steps>
    </integrator>

    <input-files>
        <amber>
            <topology>complex.prmtop</topology>
            <coordinates type="rst7">complex.rst7</coordinates>
        </amber>
    </input-files>

    <outputs>
        <directory>output/</directory>
        <overwrite-output>True</overwrite-output>
        <reporting>
            <energy><interval>5000</interval></energy>
            <coordinates><file-type>DCD</file-type><interval>5000</interval></coordinates>
            <statistics><interval>5000</interval></statistics>
        </reporting>
    </outputs>
</gamd>
```

**Step counts:**
- The total run is `conventional-md + gamd-equilibration + gamd-production` steps.
- The two `-prep` counts are the first part of their stage, not extra steps. They are the equilibration period before boost statistics start being collected.
- `conventional-md` and `gamd-equilibration` must be multiples of `averaging-window-interval`.

### 5. Run

```bash
gamdRunner xml ligamd.xml                    # CUDA, GPU 0
gamdRunner xml ligamd.xml -p CUDA -d 1       # another GPU
gamdRunner xml ligamd.xml -r                 # restart from the last checkpoint
```

Run several independent replicas, each with a different `<random-seed>` and its own output directory (`-o`). That usually samples unbinding better than one long trajectory.

### 6. Outputs

Main files in the output directory:

| File | Contents |
|---|---|
| `output.dcd` | Trajectory |
| `gamd.log` | Per-frame boost potentials; input for PyReweighting |
| `state-data.log` | Energies, temperature, volume |
| `gamd-running.csv` | Boost statistics over time (Vmax, Vmin, Vavg, sigmaV, k0, k, threshold) per group |
| `gamd-restart.dat` | Boost statistics (Vmax, Vmin, Vavg, sigmaV) at the end of the run |

Reweight free-energy profiles from `gamd.log` with [PyReweighting](https://github.com/MiaoLab20/PyReweighting).

---

## Important Options

To see all available command-line options:

```bash
gamdRunner -h
```

---

## Current Status

The following GaMD boost types are implemented (upper and lower bound versions):

- `dihedral`
- `total`
- `dual total/dihedral`
- `non-bonded`
- `dual non-bonded/dihedral`
- `ligamd` (Ligand-specific non-bonded boost)
- `ligamd-dual` (Ligand non-bonded + Protein dihedral boost)

---

## Questions

For usage questions:

- Use the `gamd-discuss` mailing list  
- Clearly mention you are using the **OpenMM version of GaMD**

Use GitHub Issues only for:

- Code/documentation problems  
- Feature requests  

---

## Authors and Contributors

### LiGaMD Integration

- Yassir Boulaamane  

### Original GaMD-OpenMM Authors (Miao Lab)

The following contributors participated directly in coding and validation (alphabetical order):

- Matthew Copeland  
- Hung Do  
- Keya Joshi  
- Yinglong Miao  
- Lane Votapka  
- Jinan Wang  

Thanks to everyone who has contributed feedback, bug reports, and improvements.

---

## Citing

If you use this LiGaMD-OpenMM implementation, please cite both the original GaMD-OpenMM paper and the LiGaMD methodology:

**Copeland, M.M., Do, H.N., Votapka, L., Joshi, K., Wang, J., Amaro, R., and Miao, Y. (2022)**  
*Gaussian accelerated molecular dynamics in OpenMM.*  
Journal of Physical Chemistry B, 126(31): 5810–5820.

**Miao, Y., et al. (2020)**  
*Ligand Gaussian accelerated molecular dynamics (LiGaMD): Characterization of ligand binding thermodynamics and kinetics.*  
Journal of Chemical Theory and Computation, 16(9): 5526–5547.

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

## 4. Install PyReweighting

Clone the PyReweighting repository:

```bash
git clone https://github.com/MiaoLab20/PyReweighting.git
```

> **Note:**  
> If developing the GaMD module and using test scripts, ensure the PyReweighting directory is added to your `PATH`.

## 4. Install PyReweighting

Clone the PyReweighting repository:

```bash
git clone https://github.com/MiaoLab20/PyReweighting.git
```

> **Note:**  
> If developing the GaMD module and using test scripts, ensure the PyReweighting directory is added to your `PATH`.

---

## 5. Clone and Install LiGaMD-OpenMM

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
python setup.py test
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

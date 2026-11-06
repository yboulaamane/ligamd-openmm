Installation
============

LiGaMD-OpenMM runs on Linux with conda and an NVIDIA GPU with driver 525 or
newer. From a terminal::

  git clone https://github.com/yboulaamane/ligamd-openmm.git
  cd ligamd-openmm
  conda env create -f environment.yml
  conda activate ligamd
  pip install -e . --no-deps

This installs OpenMM, the package and the ``gamdRunner`` command. Keep
``--no-deps``, because without it pip replaces conda's NumPy and OpenMM stops
importing.

To check the install, run ``pytest gamd/tests``. Then, on a machine with a GPU,
run ``python -m openmm.testInstallation`` and confirm that CUDA is listed.

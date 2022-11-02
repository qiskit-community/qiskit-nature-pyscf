# Qiskit Nature PySCF

[![License](https://img.shields.io/github/license/qiskit-community/qiskit-nature-pyscf.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)<!--- long-description-skip-begin -->[![Build Status](https://github.com/qiskit-community/qiskit-nature-pyscf/workflows/Nature%20PySCF%20Unit%20Tests/badge.svg?branch=main)](https://github.com/qiskit-community/qiskit-nature-pyscf/actions?query=workflow%3A"Nature%20PySCF%20Unit%20Tests"+branch%3Amain+event%3Apush)[![](https://img.shields.io/github/release/qiskit-community/qiskit-nature-pyscf.svg?style=popout-square)](https://github.com/qiskit-community/qiskit-nature-pyscf/releases)[![](https://img.shields.io/pypi/dm/qiskit-nature-pyscf.svg?style=popout-square)](https://pypi.org/project/qiskit-nature-pyscf/)[![Coverage Status](https://coveralls.io/repos/github/qiskit-community/qiskit-nature-pyscf/badge.svg?branch=main)](https://coveralls.io/github/qiskit-community/qiskit-nature-pyscf?branch=main)<!--- long-description-skip-end -->

**Qiskit Nature PySCF** is a third-party integration plugin of Qiskit Nature + PySCF.

## Installation

We encourage installing Qiskit Nature via the pip tool (a python package manager).

```bash
pip install qiskit-nature-pyscf
```

**pip** will handle all dependencies automatically and you will always install the latest
(and well-tested) version.

If you want to work on the very latest work-in-progress versions, either to try features ahead of
their official release or if you want to contribute to Qiskit Nature PySCF, then you can install from source.


## Usage

This plugin couples the APIs of PySCF and Qiskit Nature, enabling a user of PySCF to leverage
Quantum-based algorithms implemented in Qiskit to be used in-place of their classical counterparts.

### Active Space Calculations

One very common approach is to use a Quantum algorithm to find the ground state in an active space
calculation. To this extent, this plugin provides the `QiskitSolver` class, which you can inject
directly into your `CASCI` or `CASSCF` simulation objects of PySCF.

Below we show a simple example of how to do this.

```python
from pyscf import gto, scf, mcscf

from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import GroundStateEigensolver, VQEUCCFactory
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter

from qiskit_nature_pyscf import QiskitSolver

mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g")

h_f = scf.RHF(mol).run()

norb, nelec = 2, 2

cas = mcscf.CASCI(h_f, norb, nelec)

converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)

vqe = VQEUCCFactory(Estimator(), UCCSD(), SLSQP())

algorithm = GroundStateEigensolver(converter, vqe)

cas.fcisolver = QiskitSolver(algorithm)

cas.run()
```

More detailed documentation can be found at
[Documentation](https://qiskit-community.github.io/qiskit-nature-pyscf/). For more detailed 
explanations we recommend to check out the documentation of
[PySCF](https://pyscf.org/) and [Qiskit Nature](https://qiskit.org/documentation/nature/).

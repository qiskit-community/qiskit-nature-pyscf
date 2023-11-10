# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the PySCFGroundStateSolver."""

from test import QiskitNaturePySCFTestCase

from pyscf import mcscf, fci
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from qiskit_nature_pyscf import PySCFGroundStateSolver


class TestPySCFGroundStateSolver(QiskitNaturePySCFTestCase):
    """Tests for the PySCFGroundStateSolver."""

    def test_direct_spin1_fci(self):
        """Test with the ``fci.direct_spin1.FCISolver``."""
        driver = PySCFDriver(
            atom="H 0.0 0.0 0.0; H 0.0 0.0 1.5",
            basis="sto3g",
        )
        problem = driver.run()

        solver = PySCFGroundStateSolver(fci.direct_spin1.FCI())

        result = solver.solve(problem)

        casci = mcscf.CASCI(driver._calc, 2, 2)
        casci.run()

        self.assertAlmostEqual(casci.e_tot, result.total_energies[0])

    def test_direct_uhf_fci(self):
        """Test with the ``fci.direct_uhf.FCISolver``."""
        driver = PySCFDriver(
            atom="O 0.0 0.0 0.0; O 0.0 0.0 1.5",
            basis="sto3g",
            spin=2,
            method=MethodType.UHF,
        )
        problem = driver.run()

        transformer = ActiveSpaceTransformer(4, 4)

        problem = transformer.transform(problem)

        solver = PySCFGroundStateSolver(fci.direct_uhf.FCI())

        result = solver.solve(problem)

        casci = mcscf.UCASCI(driver._calc, 4, 4)
        casci.run()

        self.assertAlmostEqual(casci.e_tot, result.total_energies[0])

    def test_nroots_support(self):
        """Test support for more than ground-state calculations."""
        driver = PySCFDriver(
            atom="H 0.0 0.0 0.0; H 0.0 0.0 1.5",
            basis="sto3g",
        )
        problem = driver.run()

        fci_solver = fci.direct_spin1.FCI()
        fci_solver.nroots = 2
        solver = PySCFGroundStateSolver(fci_solver)

        result = solver.solve(problem)
        print(result)

        casci = mcscf.CASCI(driver._calc, 2, 2)
        casci.fcisolver.nroots = 2
        casci.run()

        self.assertAlmostEqual(casci.e_tot[0], result.total_energies[0])
        self.assertAlmostEqual(casci.e_tot[1], result.total_energies[1])
        self.assertEqual(len(result.num_particles), 2)
        self.assertEqual(len(result.magnetization), 2)
        self.assertEqual(len(result.total_angular_momentum), 2)

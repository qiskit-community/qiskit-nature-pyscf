# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the QiskitSolver."""

from test import QiskitNaturePySCFTestCase

from pyscf import gto, scf, mcscf

from qiskit_nature.second_q.algorithms import GroundStateEigensolver, NumPyMinimumEigensolverFactory
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature_pyscf.qiskit_solver import QiskitSolver


class TestQiskitSolver(QiskitNaturePySCFTestCase):
    """Tests for the QiskitSolver."""

    def test_rhf_casci_h2(self):
        """Test the RHF and CASCI method on H2."""
        mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="631g*")

        h_f = scf.RHF(mol).run()

        norb, nelec = 2, 2

        cas_ref = mcscf.CASCI(h_f, norb, nelec).run()

        qcas = mcscf.CASCI(h_f, norb, nelec)

        ground_state_solver = GroundStateEigensolver(
            QubitConverter(JordanWignerMapper()),
            NumPyMinimumEigensolverFactory(),
        )

        qcas.fcisolver = QiskitSolver(ground_state_solver)

        qcas.run()

        self.assertAlmostEqual(qcas.e_tot, cas_ref.e_tot)

    def test_uhf_ucasci_h2(self):
        """Test the UHF and UCASCI method on H2."""
        mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="631g*")

        h_f = scf.UHF(mol).run()

        norb, nelec = 2, 2

        cas_ref = mcscf.UCASCI(h_f, norb, nelec).run()

        qcas = mcscf.UCASCI(h_f, norb, nelec)

        ground_state_solver = GroundStateEigensolver(
            QubitConverter(JordanWignerMapper()),
            NumPyMinimumEigensolverFactory(),
        )

        qcas.fcisolver = QiskitSolver(ground_state_solver)

        qcas.run()

        self.assertAlmostEqual(qcas.e_tot, cas_ref.e_tot)

    def test_rhf_casscf_h2(self):
        """Test the RHF and CASSCF method on H2."""
        mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="631g*")

        h_f = scf.RHF(mol).run()

        norb, nelec = 2, 2

        cas_ref = mcscf.CASSCF(h_f, norb, nelec).run()

        qcas = mcscf.CASSCF(h_f, norb, nelec)

        ground_state_solver = GroundStateEigensolver(
            QubitConverter(JordanWignerMapper()),
            NumPyMinimumEigensolverFactory(),
        )

        qcas.fcisolver = QiskitSolver(ground_state_solver)

        qcas.run()

        self.assertAlmostEqual(qcas.e_tot, cas_ref.e_tot)

    def test_uhf_ucasscf_h2(self):
        """Test the UHF and UCASSCF method on H2."""
        mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="631g*")

        h_f = scf.UHF(mol).run()

        norb, nelec = 2, 2

        cas_ref = mcscf.UCASSCF(h_f, norb, nelec).run()

        qcas = mcscf.UCASSCF(h_f, norb, nelec)

        ground_state_solver = GroundStateEigensolver(
            QubitConverter(JordanWignerMapper()),
            NumPyMinimumEigensolverFactory(),
        )

        qcas.fcisolver = QiskitSolver(ground_state_solver)

        qcas.run()

        self.assertAlmostEqual(qcas.e_tot, cas_ref.e_tot)

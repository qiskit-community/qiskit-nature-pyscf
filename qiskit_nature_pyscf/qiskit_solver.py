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

"""A Qiskit-based FCI solver for PySCF."""

from __future__ import annotations

import numpy as np

from pyscf import ao2mo

from qiskit_nature.second_q.algorithms import GroundStateSolver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators.tensor_ordering import IndexType, to_chemist_ordering
from qiskit_nature.second_q.problems import (
    ElectronicBasis,
    ElectronicStructureProblem,
    ElectronicStructureResult,
)
from qiskit_nature.second_q.properties import (
    AngularMomentum,
    ElectronicDensity,
    Magnetization,
    ParticleNumber,
)


class QiskitSolver:
    """A Qiskit-based FCI solver for PySCF.

    This class bridges the APIs of PySCF and Qiskit Nature. You can use an instance of it to
    overwrite the ``fcisolver`` attribute of a :class:`pyscf.mcscf.casci.CASCI` object, in order to
    use a Quantum algorithm as provided by Qiskit for the CI solution step.
    Depending on the algorithm and its configuration, this may **not** result in the true FCI
    result, so care must be taken when using this class.

    Here is an example use case:

    .. code-block:: python

        from pyscf import gto, scf, mcscf

        from qiskit.algorithms.optimizers import SLSQP
        from qiskit.primitives import Estimator
        from qiskit_nature.second_q.algorithms import GroundStateEigensolver, VQEUCCFactory
        from qiskit_nature.second_q.circuit.library import UCCSD
        from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter

        from qiskit_nature_pyscf import QiskitSolver

        mol = gto.M(atom="Li 0 0 0; H 0 0 1.51", basis="631g*")

        h_f = scf.RHF(mol).run()

        norb, nelec = 2, 2

        cas = mcscf.CASCI(h_f, norb, nelec)

        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)

        vqe = VQEUCCFactory(Estimator(), UCCSD(), SLSQP())

        algorithm = GroundStateEigensolver(converter, vqe)

        cas.fcisolver = QiskitSolver(algorithm)

        cas.run()

    For more details please to the documentation of [PySCF](https://pyscf.org/) and
    [Qiskit Nature](https://qiskit.org/documentation/nature/).

    An instance of this class has the following attributes:

    Attributes:
        solver: a ground-state solver provided by Qiskit Nature. This must be supplied by the user
            prior to running the algorithm.
        density: an ElectronicDensity provided by Qiskit Nature. This is only available after the
            first iteration of the algorithm.
        result: an ElectronicStructureResult provided by Qiskit Nature. This is only available after
            the first iteration of the algorithm.
    """

    def __init__(self, solver: GroundStateSolver):
        """
        Args:
            solver: a ground-state solver provided by Qiskit Nature.
        """
        self.solver = solver
        self.density: ElectronicDensity = None
        self.result: ElectronicStructureResult = None

    # pylint: disable=invalid-name,missing-type-doc
    def kernel(
        self,
        h1: np.ndarray | tuple[np.ndarray, np.ndarray],
        h2: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
        norb: int,
        nelec: int | tuple[int, int],
        ci0=None,
        ecore: float = 0,
        **kwargs,
    ) -> tuple[float, QiskitSolver]:
        # pylint: disable=unused-argument
        """Finds the ground-state of the provided Hamiltonian.

        Args:
            h1: the one-body integrals. If this is a tuple, it indicates the alpha- and beta-spin
                coefficients, respectively.
            h2: the two-body integrals in chemists index ordering. If this is a tuple, it indicates the
                alpha-alpha-, alpha-beta-, and beta-beta-spin integrals, respectively.
            norb: the number of (active) orbitals.
            nelec: the number of (active) electrons. If this is an integer, it is the total number of
                electrons; if it is a tuple, it indicates the alpha- and beta-spin particles,
                respectively.
            ci0: the initial CI vector. (Currently ignored!)
            ecore: the core (inactive) energy.
            kwargs: any additional keyword arguments. (Currently ignored!)

        Returns:
            The pair of total evaluated energy (including ``ecore``) and a reference to this
            ``QiskitSolver`` itself.
        """
        if isinstance(h1, tuple):
            h1_a, h1_b = h1
        else:
            h1_a, h1_b = h1, None
        if isinstance(h2, tuple):
            h2_aa, h2_ab, h2_bb = h2
        else:
            h2_aa, h2_ab, h2_bb = h2, None, None

        hamiltonian = ElectronicEnergy.from_raw_integrals(
            h1_a,
            ao2mo.restore(1, h2_aa, norb),
            h1_b=h1_b,
            h2_bb=h2_bb,
            h2_ba=h2_ab.T if h2_ab is not None else None,
        )
        hamiltonian.constants["inactive energy"] = ecore

        problem = ElectronicStructureProblem(hamiltonian)
        problem.basis = ElectronicBasis.MO
        problem.num_spatial_orbitals = norb
        if not isinstance(nelec, tuple):
            nbeta = nelec // 2
            nalpha = nelec - nbeta
            nelec = (nalpha, nbeta)
        problem.num_particles = nelec

        if self.density is None:
            self.density = ElectronicDensity.from_orbital_occupation(
                problem.orbital_occupations,
                problem.orbital_occupations_b,
            )

        problem.properties.particle_number = ParticleNumber(norb)
        problem.properties.magnetization = Magnetization(norb)
        problem.properties.angular_momentum = AngularMomentum(norb)
        problem.properties.electronic_density = self.density

        self.result = self.solver.solve(problem)
        self.density = self.result.electronic_density

        e_tot = self.result.total_energies[0]
        return e_tot, self

    approx_kernel = kernel

    def make_rdm1(
        self, fake_ci_vec: QiskitSolver, norb: int, nelec: int | tuple[int, int]
    ) -> np.ndarray:
        # pylint: disable=unused-argument
        """Constructs the spin-traced 1-RDM.

        Args:
            fake_ci_vec: the reference to the ``QiskitSolver``.
            norb: the number of (active) orbitals. (currently ignored!)
            nelec: the number of (active) electrons. (currently ignored!)

        Returns:
            The spin-traced 1-RDM.
        """
        return np.asarray(fake_ci_vec.density.trace_spin()["+-"])

    def make_rdm1s(
        self, fake_ci_vec: QiskitSolver, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        # pylint: disable=unused-argument
        """Constructs the alpha- and beta-spin 1-RDMs.

        Args:
            fake_ci_vec: the reference to the ``QiskitSolver``.
            norb: the number of (active) orbitals. (currently ignored!)
            nelec: the number of (active) electrons. (currently ignored!)

        Returns:
            The alpha- and beta-spin 1-RDMs.
        """
        return np.asarray(fake_ci_vec.density.alpha["+-"]), np.asarray(
            fake_ci_vec.density.beta["+-"]
        )

    def make_rdm12(
        self, fake_ci_vec: QiskitSolver, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        # pylint: disable=unused-argument
        """Constructs the spin-traced 1- and 2-RDMs.

        Args:
            fake_ci_vec: the reference to the ``QiskitSolver``.
            norb: the number of (active) orbitals. (currently ignored!)
            nelec: the number of (active) electrons. (currently ignored!)

        Returns:
            The spin-traced 1- and 2-RDMs. The 2-RDM is returned in chemist index ordering.
        """
        traced = fake_ci_vec.density.trace_spin()
        return (
            np.asarray(traced["+-"]),
            to_chemist_ordering(traced["++--"], index_order=IndexType.PHYSICIST),
        )

    def make_rdm12s(
        self, fake_ci_vec: QiskitSolver, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        # pylint: disable=unused-argument
        """Constructs the alpha- and beta-spin 1- and 2-RDMs.

        Args:
            fake_ci_vec: the reference to the ``QiskitSolver``.
            norb: the number of (active) orbitals. (currently ignored!)
            nelec: the number of (active) electrons. (currently ignored!)

        Returns:
            A pair of tuples; the first tuple being the alpha- and beta-spin 1-RDMs; the second
            tuple being the alpha-alpha-, alpha-beta- and beta-beta-spin 2-RDMs in chemist index
            ordering.
        """
        dm2_aa = to_chemist_ordering(
            fake_ci_vec.density.alpha["++--"], index_order=IndexType.PHYSICIST
        )
        dm2_bb = to_chemist_ordering(
            fake_ci_vec.density.beta["++--"], index_order=IndexType.PHYSICIST
        )
        dm2_ba = to_chemist_ordering(
            fake_ci_vec.density.beta_alpha["++--"], index_order=IndexType.PHYSICIST
        )
        return (self.make_rdm1s(fake_ci_vec, norb, nelec), (dm2_aa, dm2_ba.T, dm2_bb))

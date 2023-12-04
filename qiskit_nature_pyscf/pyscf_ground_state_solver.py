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

"""A PySCF-based GroundStateSolver for Qiskit Nature."""

from __future__ import annotations

import logging

import numpy as np
from pyscf import fci
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.algorithms import GroundStateSolver
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.operators.tensor_ordering import find_index_order, to_chemist_ordering
from qiskit_nature.second_q.problems import (
    BaseProblem,
    ElectronicStructureProblem,
    ElectronicStructureResult,
)
from qiskit_nature.second_q.properties import ElectronicDensity

LOGGER = logging.getLogger(__name__)


class PySCFGroundStateSolver(GroundStateSolver):
    """A PySCF-based GroundStateSolver for Qiskit Nature.

    This class provides a :class:`~qiskit_nature.second_q.algorithms.GroundStateSolver` for Qiskit
    Nature, leveraging the ``fci`` module of PySCF. This is utility does not enable any Quantum
    algorithms to be used (since it replaces them in the Qiskit workflow) but instead provides a
    useful utility for debugging classical computational workflows based on Qiskit Nature.
    More importantly, it provides a more efficient implementation of what Qiskit otherwise achieves
    using a :class:`~qiskit_algorithms.NumPyMinimumEigensolver` in combination with a
    ``filter_criterion``. For non-singlet spin ground states the setup using Qiskit components is a
    lot more involved, whereas this class provides an easy-to-use alternative.

    Here is an example use cas:

    .. code-block:: python

       from pyscf import fci

       from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
       from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

       from qiskit_nature_pyscf import PySCFGroundStateSolver

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
       print(result)

    For more details please to the documentation of [PySCF](https://pyscf.org/) and
    [Qiskit Nature](https://qiskit.org/ecosystem/nature/).
    """

    def __init__(self, solver: fci.direct_spin1.FCISolver) -> None:
        """
        Args:
            solver: a FCI solver provided by PySCF.
        """
        super().__init__(None)
        self._solver = solver

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> ElectronicStructureResult:
        """Finds the ground-state of the provided problem.

        This method is written to match the
        :class:`~qiskit_nature.second_q.algorithms.GroundStateSolver` API of Qiskit Nature but it
        does not support the evaluation of ``aux_operators`` and will ignore them.
        It is also only capable of solving an
        :class:`~qiskit_nature.second_q.problems.ElectronicStructureProblem` and will raise an error
        if another problem object is provided.

        Args:
            problem: the problem instance to be solved.
            aux_operators: **ignored**.

        Raises:
            TypeError: if a problem other than an
                :class:`~qiskit_nature.second_q.problems.ElectronicStructureProblem` is provided.

        Returns:
            The interpreted result object in Qiskit Nature format.
        """
        if aux_operators is not None:
            LOGGER.warning(
                "This solver does not support auxiliary operators. They will be ignored."
            )

        if not isinstance(problem, ElectronicStructureProblem):
            raise TypeError(
                "This solver only supports an ElectronicStructureProblem but you provided a "
                f"problem of type '{type(problem)}'."
            )

        e_ints = problem.hamiltonian.electronic_integrals

        restricted_spin = e_ints.beta_alpha.is_empty()

        one_body: np.ndarray
        two_body: np.ndarray

        if restricted_spin:
            one_body = np.asarray(problem.hamiltonian.electronic_integrals.alpha["+-"])
            two_body = np.asarray(
                to_chemist_ordering(
                    problem.hamiltonian.electronic_integrals.alpha["++--"],
                )
            )
        else:
            one_body = np.asarray(
                [
                    problem.hamiltonian.electronic_integrals.alpha["+-"],
                    problem.hamiltonian.electronic_integrals.beta["+-"],
                ]
            )
            index_order = find_index_order(problem.hamiltonian.electronic_integrals.alpha["++--"])
            two_body = np.asarray(
                [
                    to_chemist_ordering(
                        problem.hamiltonian.electronic_integrals.alpha["++--"],
                        index_order=index_order,
                    ),
                    to_chemist_ordering(
                        problem.hamiltonian.electronic_integrals.alpha_beta["++--"],
                        index_order=index_order,
                    ),
                    to_chemist_ordering(
                        problem.hamiltonian.electronic_integrals.beta["++--"],
                        index_order=index_order,
                    ),
                ]
            )

        energy, ci_vec = self.solver.kernel(
            h1e=one_body,
            eri=two_body,
            norb=problem.num_spatial_orbitals,
            nelec=problem.num_particles,
        )

        if not isinstance(energy, np.ndarray):
            # computed a single root, but in the following we pretent to always have multiple ones
            energy = [energy]
            ci_vec = [ci_vec]

        raw_density = self.solver.make_rdm1s(
            ci_vec[0], norb=problem.num_spatial_orbitals, nelec=problem.num_particles
        )
        density = ElectronicDensity.from_raw_integrals(raw_density[0], h1_b=raw_density[1])

        overlap_ab: np.ndarray | None = None
        if problem.properties.angular_momentum is not None:
            overlap_ab = problem.properties.angular_momentum.overlap

        magnetization: list[float] = []
        angular_momentum: list[float] = []
        for vec in ci_vec:
            densities = self.solver.make_rdm12s(
                vec, norb=problem.num_spatial_orbitals, nelec=problem.num_particles
            )
            spin_square, spin_z = self._compute_spin(densities, overlap_ab)
            magnetization.append(spin_z)
            angular_momentum.append(spin_square)

        result = ElectronicStructureResult()
        result.computed_energies = np.asarray(energy)
        result.hartree_fock_energy = problem.reference_energy
        result.extracted_transformer_energies = dict(problem.hamiltonian.constants.items())
        result.nuclear_repulsion_energy = result.extracted_transformer_energies.pop(
            "nuclear_repulsion_energy", None
        )
        result.num_particles = [sum(problem.num_particles)] * len(energy)
        result.magnetization = magnetization
        result.total_angular_momentum = angular_momentum
        # NOTE: the ElectronicStructureResult does not yet support multiple densities. Thus, we only
        # have the ground-state density here, regardless of how many roots were computed.
        result.electronic_density = density

        return result

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> tuple[SparsePauliOp, dict[str, SparsePauliOp] | None]:
        """This solver does not deal with qubit operators and this method raises a RuntimeError."""
        raise RuntimeError(
            "This solver is a purely classical one and as such solves an ElectronicStructureProblem"
            " on the second-quantization level. Thus, it has no concept of qubit operators and does"
            " not support this method."
        )

    def supports_aux_operators(self) -> bool:
        """Returns whether the eigensolver supports auxiliary operators."""
        return False

    @property
    def solver(self) -> fci.direct_spin1.FCISolver:
        """Returns the solver."""
        return self._solver

    @staticmethod
    def _compute_spin(
        densities: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
        overlap_ab: np.ndarray | None,
    ):
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = densities

        identity = np.eye(dm1a.shape[0])
        if overlap_ab is None:
            overlap_ab = identity

        overlap_ba = overlap_ab.T

        # NOTE: we know for a fact, that overlap_aa and overlap_bb will always equal the identity
        # when dealing with a single wavefunction
        ssz = (
            np.einsum("ijkl,ij,kl->", dm2aa, identity, identity)
            - np.einsum("ijkl,ij,kl->", dm2ab, identity, identity)
            + np.einsum("ijkl,ij,kl->", dm2bb, identity, identity)
            - np.einsum("ijkl,ij,kl->", dm2ab, identity, identity)
            + np.einsum("ji,ij->", dm1a, identity)
            + np.einsum("ji,ij->", dm1b, identity)
        ) * 0.25

        dm2abba = -dm2ab.transpose(0, 3, 2, 1)  # alpha^+ beta^+ alpha beta
        dm2baab = -dm2ab.transpose(2, 1, 0, 3)  # beta^+ alpha^+ beta alpha
        ssxy = (
            np.einsum("ijkl,ij,kl->", dm2abba, overlap_ab, overlap_ba)
            + np.einsum("ijkl,ij,kl->", dm2baab, overlap_ba, overlap_ab)
            # NOTE: the following two lines are different from PySCF because we may deal with
            # non-unitary overlap_ab matrices
            + np.einsum("ji,ij->", dm1a, overlap_ab @ overlap_ba)
            + np.einsum("ji,ij->", dm1b, overlap_ba @ overlap_ab)
        ) * 0.5

        spin_square = ssxy + ssz

        return spin_square, np.sqrt(ssz)

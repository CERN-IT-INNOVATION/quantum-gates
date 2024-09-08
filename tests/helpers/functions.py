"""
Helper functions for the unit tests.
"""

import numpy as np
import random

from src.quantum_gates.utilities import DeviceParameters
from tests.helpers.gates import X, SX,Z, H, CNOT, identity, single_qubit_gate_list


def generate_random_matrix_products(nqubits: int, steps: int, prob_cnot: float=0.5, many_identites=False):
    """ Generates list of matrix products for nqubits. In this version, we just use X, Z, H, 1 (single qubit) and
        CNOT (two qubit) matrices in the matrix products. CNOT gates are selected with a probability (prop_cnot) in
        cases there is still budget for two qubit gates.
    """

    gate_list = [X, SX, Z, H] + [identity for i in range(40)] if many_identites else single_qubit_gate_list + identity

    # Result
    mp_list = []

    # Generate
    for step in range(steps):
        mp = []
        budget = nqubits
        while budget > 0:
            # Case: Only one qubit left
            if budget == 1:
                mp.append(random.choice(gate_list))
                budget -= 1

            # Case: More than one qubit left -> We can apply two qubit gates.
            else:
                # Select CNOT with probability prop_cnot.
                if random.random() < prob_cnot:
                    mp.append(CNOT)
                    mp.append(1)
                    budget -= 2
                else:
                    mp.append(random.choice(gate_list))
                    budget -= 1
        assert budget == 0
        mp_list.append(mp)
    return mp_list


def vector_almost_equal(m1, m2, nqubit: int, abstol: float=1e-9):
    """ Check if m1 and m2 are close.
    """
    diff = m1 - m2
    return all((abs(diff[i]) < abstol for i in range(2**nqubit)))


def matrix_almost_equal(m1, m2, nqubit: int, abstol: float=1e-9):
    """ Check if m1 and m2 are close. """
    diff = m1 - m2
    return all((abs(diff[i,j]) < abstol) for i in range(2**nqubit) for j in range(2**nqubit))

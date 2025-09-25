"""
Helper functions and variable definitions for the unit tests.
"""

import numpy as np


""" Gates """

identity = np.eye(2)
X = np.array(
    [[0,1],
     [1,0]]
)

SX = np.sqrt(1/2) * np.array(
    [[1.0, -1.0j],
     [-1.0j, 1.0]]
)

Z = np.array(
    [[1,0],
     [0,-1]]
)

H = np.array(
    [[1,1],
     [1,-1]]
) / np.sqrt(2)

single_qubit_gate_list = [identity, X, SX, Z, H]

CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]]
)

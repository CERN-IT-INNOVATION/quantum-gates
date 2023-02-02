import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info.operators import Operator


def hadamard_reverse_qft_circ(n_qubits: int):
    """ Generates the Qiskit circuit applying Hadamard gates on all the qubits and the inverse Quantum Fourier
        transform.
    """

    qft = QuantumCircuit(n_qubits, n_qubits)

    def swap_registers(circ: QuantumCircuit, n_qubits: int):
        for qubit in range(n_qubits//2):
            circ.swap(qubit, n_qubits-qubit-1)
        return circ

    def qft_rotations(circ: QuantumCircuit, n_qubits: int):
        if n_qubits == 0:
            return qft
        n_qubits -= 1
        qft.h(n_qubits)
        for i in range(n_qubits):
            qft.cp(np.pi/2**(n_qubits - i), i, n_qubits)
        qft_rotations(circ, n_qubits)

    qft_rotations(qft, n_qubits)
    swap_registers(qft, n_qubits)

    for j in range(0, n_qubits):
        qft.h(j)

    qft = qft.inverse()
    qft.barrier(range(n_qubits))
    qft.measure(range(n_qubits), range(n_qubits))
    return qft


def ghz_circ(n_qubits: int, backend):
    """ Generates the GHZ circuit for n qubits. The circuit first applies a Hadamard on the first qubit, and then
        iteratively applies CNOT gates with qubit i as control and i+1 as target, i = 0, ..., n_qubits - 2.
    """

    ghz = QuantumCircuit(n_qubits, n_qubits)

    ghz.h(0)
    for j in range(1, n_qubits):
        ghz.cx(0, j)
    ghz.barrier(range(n_qubits))
    ghz.measure(range(n_qubits), range(n_qubits))

    return ghz


def qft_circ(n_qubits: int):
    """ Generates the Quantum Fourier Transform circuit.
    """

    qft = QuantumCircuit(n_qubits, n_qubits)

    def qft_rotations(circ, n_qubits):
        if n_qubits == 0:
            return qft
        n_qubits -= 1
        qft.h(n_qubits)
        for i in range(n_qubits):
            qft.cp(np.pi/2**(n_qubits - i), i, n_qubits)

        qft_rotations(circ, n_qubits)

    qft_rotations(qft, n_qubits)
    qft.barrier(range(n_qubits))
    qft.measure(range(n_qubits), range(n_qubits))

    return qft


def qaoa_circ(G, theta: float):
    """ Generates a Quantum Approximate Optimization Algorithm circuit.
    """

    # Parameters
    gamma = Parameter('gamma')
    beta = Parameter('beta')

    n = len(G.nodes())  # Number of qubits
    p = len(theta)//2   # Number of parameters

    beta_range = theta[:p]
    gamma_range = theta[p:]

    # Cost unitary
    circ_gamma = QuantumCircuit(n)
    for pair in list(G.edges()):
        circ_gamma.rzz(-gamma, pair[0], pair[1])

    # Mixer unitary
    circ_beta = QuantumCircuit(n)
    circ_beta.rx(2 * beta, range(n))

    circuits_g = [circ_gamma.bind_parameters({gamma: gamma_val}) for gamma_val in gamma_range]
    circuits_b = [circ_beta.bind_parameters({beta: beta_val}) for beta_val in beta_range]

    Ug = [Operator(i) for i in circuits_g]

    Ub = [Operator(i) for i in circuits_b]

    # QAOA circuit
    qc = QuantumCircuit(n,n)
    qc.h(range(n))

    for i in range(0, p):
        qc.unitary(Ug[i], range(4), f'U(-gamma{i})')
        qc.unitary(Ub[i], range(4), f'U(2beta{i})')

    qc.barrier(range(n))
    qc.measure(range(n), range(n))
    return qc

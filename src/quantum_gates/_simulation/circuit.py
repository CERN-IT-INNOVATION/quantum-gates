"""
This module implements the base class to perform noisy quantum simulations with noisy gates approach
"""

import numpy as np
import functools as ft

from .._gates.gates import Gates
from .backend import StandardBackend, EfficientBackend, BackendForOnes


class Circuit(object):
    """ Class that allows to define custom noisy quantum circuit and apply it on a given initial quantum state.
    """

    def __init__(self, nqubit: int, depth: int, gates: Gates):
        """
        Init method for the noisy quantum circuit
        
        Parameters:
            nqubit: number of qubits
            depth: depth of the circuit
        
        Returns:
            None
        """
        self.nqubit = nqubit
        self.depth = depth
        self.j = 0
        self.s = 0
        self.phi = [0 for i in range(nqubit)]
        self.circuit = [[1 for i in range(depth)] for j in range(nqubit)]

        # Gate collection for a specific pulse shape.
        self.gates = gates
    
    def display(self):
        """
        Display the noisy quantum circuit
        
        Returns:
             None
        """
        print(self.circuit)

    def apply(self, gate, i):
        """
        Apply an arbitrary single qubit gate.
        
        Args:
            gate: single qubit gate to apply (array)
            i: index of the qubit (int)
        
        Returns:
             None
        """
        if not isinstance(gate, np.ndarray):
            raise ValueError(f"Circuit.apply() expected gate to be a numpy array but found type {type(gate)}.")

        if gate.shape != (2, 2):
            raise ValueError(f"Circuit.apply() expected single qubit gate of shape (2,2) but found {gate.shape}.")

        if self.s < self.nqubit:
            self.circuit[i][self.j] = gate
            self.s = self.s+1

        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = gate

    def statevector(self, psi0) -> np.array:
        """
        Compute the output statevector of the noisy quantum circuit

        Args:
             psi0: initial statevector, must be 1 x 2^n

        Returns:
             output statevector
        """
        self.circuit = np.array(self.circuit, dtype=object)
        matrix_prod = ft.reduce(np.kron, self.circuit[:, 0])
        for i in range(1, self.depth):
            matrix_prod = ft.reduce(np.kron, self.circuit[:, i]) @ matrix_prod
        psi = matrix_prod @ psi0
        return psi

    def I(self, i: int):
        """
        Apply identity gate on qubit i 
        
        Args:
            i: index of the qubit
        
        Returns:
             None
        """
        Id = np.array([[1, 0], [0, 1]])
        self.apply(gate=Id, i=i)

    def Rz(self, i: int, theta: float):
        """
        Update the phase to implement virtual Rz(theta) gate on qubit i

        Args:
            i: index of the qubit
            theta: angle of rotation on the Bloch sphere

        Returns:
             None
        """
        self.phi[i] = self.phi[i] + theta

    def bitflip(self, i: int, tm: float, rout: float):
        """
        Apply bitflip (bitflip) noise gate on qubit i. Add before measurements or after initial state preparation.

        Args:
            i: index of the qubit
            tm: measurement time in ns
            rout: readout error

        Returns:
             None
        """
        self.apply(gate=self.gates.bitflip(tm, rout), i=i)

    def relaxation(self, i: int, Dt: float, T1: float, T2: float):
        """
        Apply relaxation noise gate on qubit i. Add on idle-qubits.

        Args:
            i: index of the qubit
            Dt: idle time in ns
            T1: qubit's amplitude damping time in ns
            T2: qubit's dephasing time in ns

        Returns:
             None
        """
        self.apply(gate=self.gates.relaxation(Dt, T1, T2), i=i)

    def depolarizing(self, i: int, Dt: float, p: float):
        """
        Apply depolarizing noise gate on qubit i. Add on idle-qubits.

        Args:
            i: index of the qubit
            Dt: idle time in ns
            p: single-qubit depolarizing error probability

        Returns:
             None
        """
        self.apply(gate=self.gates.depolarizing(Dt, p), i=i)

    def X(self, i: int, p: float, T1: float, T2: float) -> np.array:
        """
        Apply X single-qubit noisy quantum gate with depolarizing and
        relaxation errors during the unitary evolution.

        Args:
            i: index of the qubit
            p: single-qubit depolarizing error probability
            T1: qubit's amplitude damping time in ns
            T2: qubit's dephasing time in ns

        Returns:
              None
        """
        self.apply(gate=self.gates.X(-self.phi[i], p, T1, T2), i=i)

    def SX(self, i: int, p: float, T1: float, T2: float):
        """
        Apply SX single-qubit noisy quantum gate with depolarizing and
        relaxation errors during the unitary evolution.

        Args:
            i: index of the qubit
            p: single-qubit depolarizing error probability
            T1: qubit's amplitude damping time in ns
            T2: qubit's dephasing time in ns

        Returns:
              None
        """
        self.apply(gate=self.gates.SX(-self.phi[i], p, T1, T2), i=i)

    def CNOT(self, i: int, k: int, t_cnot: float, p_i_k: float, p_i: float, p_k: float, T1_ctr: float,
             T2_ctr: float, T1_trg: float, T2_trg: float):
        """
        Apply CNOT two-qubit noisy quantum gate with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            i: index of the control qubit (int)
            k: index of the target qubit (int)
            t_cnot: CNOT gate time in ns (double)
            p_i_k: CNOT depolarizing error probability (double)
            p_i: control single-qubit depolarizing error probability (double)
            p_k: target single-qubit depolarizing error probability (double)
            T1_ctr: control qubit's amplitude damping time in ns (double)
            T2_ctr: control qubit's dephasing time in ns (double)
            T1_trg: target qubit's amplitude damping time in ns (double)
            T2_trg: target qubit's dephasing time in ns (double)

        Returns:
              None
        """
        assert abs(i - k) == 1, f"Error, control and target are not neighbours with i={i}, k={k}."

        if self.s < self.nqubit:

            if i < k:
                self.circuit[i][self.j] = self.gates.CNOT(
                    self.phi[i], self.phi[k], t_cnot, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )
                self.phi[i] = self.phi[i] - np.pi/2

            else:
                self.circuit[i][self.j] = self.gates.CNOT_inv(
                    self.phi[i], self.phi[k], t_cnot, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )
                self.phi[i] = self.phi[i] + np.pi/2 + np.pi
                self.phi[k] = self.phi[k] + np.pi/2
            self.s = self.s+2

        elif self.s == self.nqubit:
            self.s = 2
            self.j = self.j+1

            if i < k:
                self.circuit[i][self.j] = self.gates.CNOT(
                    self.phi[i], self.phi[k], t_cnot, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )
                self.phi[i] = self.phi[i] - np.pi/2

            else:
                self.circuit[i][self.j] = self.gates.CNOT_inv(
                    self.phi[i], self.phi[k], t_cnot, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )
                self.phi[i] = self.phi[i] + np.pi/2 + np.pi
                self.phi[k] = self.phi[k] + np.pi/2

    def reset(self):
        """ Reset the circuit to the initial state. """
        self.j = 0
        self.s = 0
        self.circuit = [[1 for i in range(self.depth)] for j in range(self.nqubit)]

        # TODO: The last line was originally not there. Check if it should and whether it has a positive effect.
        self.phi = [0 for i in range(self.nqubit)]


class AlternativeCircuit(object):
    """ Class that allows to define custom noisy quantum circuit and apply it on a given initial quantum state.
        In this version, we provide a backend for the evaluation of the tensor contractions that are performed in the
        creation of the propagator.
    """

    def __init__(self, nqubit: int, gates: Gates, BackendClass: StandardBackend or EfficientBackend):
        self.nqubit = nqubit                        # Number of qubits
        self.gates = gates                          # Gate set to be used (specifies the noisy behaviour)
        self._backend = BackendClass(nqubit)        # Backend for tensor contractions
        self._BackendClass = BackendClass

        # Bookkeeping
        self.phi = [0 for i in range(nqubit)]       # Phases
        self._s = 0                                 # This many gate units have been applied during this gate time
        self._mp = [1 for i in range(self.nqubit)]  # Matrix product of the current gatetime. We start trivially.
        self._mp_list = []

    def apply(self, gate, i):
        """ Applies a single qubit gate to qubit i. If the circuit snippet is full, then matrix product is appended and
            the bookkeeping is reset.
        """
        # Input validation
        if not isinstance(gate, np.ndarray):
            raise ValueError(f"Circuit.apply() expected gate to be a numpy array but found type {type(gate)}.")

        if gate.shape != (2, 2):
            raise ValueError(f"Circuit.apply() expected single qubit gate of shape (2,2) but found {gate.shape}.")

        # Apply gate
        self._mp[i] = gate
        self._s += 1

        # Case: All gates in this gatetime have been applied
        if self._s == self.nqubit:
            self._update_mp_list()

    def statevector(self, psi0) -> np.array:
        """ Compute the output statevector of the noisy quantum circuit, psi1 = U psi0.
        """
        # Handle the trivial case in which no gates were applied.
        if len(self._mp_list) == 0:
            return psi0
        return self._backend.statevector(self._mp_list, psi0)

    def I(self, i: int):
        """
        Apply identity gate on qubit i

        Args:
            i: index of the qubit

        Returns:
             None
        """
        identity_matrix = np.array([[1, 0], [0, 1]])
        self.apply(gate=identity_matrix, i=i)

    def Rz(self, i: int, theta: float):
        """
        Update the phase to implement virtual Rz(theta) gate on qubit i

        Args:
            i: index of the qubit
            theta: angle of rotation on the Bloch sphere

        Returns:
             None
        """
        self.phi[i] = self.phi[i] + theta

    def bitflip(self, i: int, tm: float, rout: float):
        """
        Apply bitflip noise gate on qubit i. Add before measurements or after initial state preparation.

        Args:
            i: index of the qubit
            tm: measurement time in ns
            rout: readout error

        Returns:
             None
        """
        self.apply(gate=self.gates.bitflip(tm, rout), i=i)

    def relaxation(self, i: int, Dt: float, T1: float, T2: float):
        """
        Apply relaxation noise gate on qubit i. Add on idle-qubits.

        Args:
            i: index of the qubit
            Dt: idle time in ns
            T1: qubit's amplitude damping time in ns
            T2: qubit's dephasing time in ns

        Returns:
             None
        """
        self.apply(gate=self.gates.relaxation(Dt, T1, T2), i=i)

    def depolarizing(self, i: int, Dt: float, p: float):
        """
        Apply depolarizing noise gate on qubit i. Add on idle-qubits.

        Args:
            i: index of the qubit
            Dt: idle time in ns
            p: single-qubit depolarizing error probability

        Returns:
             None
        """
        self.apply(gate=self.gates.depolarizing(Dt, p), i=i)

    def X(self, i: int, p: float, T1: float, T2: float) -> np.array:
        """
        Apply X single-qubit noisy quantum gate with depolarizing and
        relaxation errors during the unitary evolution.

        Args:
            i: index of the qubit
            p: single-qubit depolarizing error probability
            T1: qubit's amplitude damping time in ns
            T2: qubit's dephasing time in ns

        Returns:
              None
        """
        self.apply(gate=self.gates.X(-self.phi[i], p, T1, T2), i=i)

    def SX(self, i: int, p: float, T1: float, T2: float):
        """
        Apply SX single-qubit noisy quantum gate with depolarizing and
        relaxation errors during the unitary evolution.

        Args:
            i: index of the qubit
            p: single-qubit depolarizing error probability
            T1: qubit's amplitude damping time in ns
            T2: qubit's dephasing time in ns

        Returns:
              None
        """
        self.apply(gate=self.gates.SX(-self.phi[i], p, T1, T2), i=i)

    def CNOT(self, i: int, k: int, t_cnot: float, p_i_k: float, p_i: float, p_k: float, T1_ctr: float,
             T2_ctr: float, T1_trg: float, T2_trg: float):
        """
        Apply CNOT two-qubit noisy quantum gate with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            i: index of the control qubit (int)
            k: index of the target qubit (int)
            t_cnot: CNOT gate time in ns (double)
            p_i_k: CNOT depolarizing error probability (double)
            p_i: control single-qubit depolarizing error probability (double)
            p_k: target single-qubit depolarizing error probability (double)
            T1_ctr: control qubit's amplitude damping time in ns (double)
            T2_ctr: control qubit's dephasing time in ns (double)
            T1_trg: target qubit's amplitude damping time in ns (double)
            T2_trg: target qubit's dephasing time in ns (double)

        Returns:
              None
        """

        # Add two qubit gate to circuit snippet
        if i < k:
            # Control i
            self._mp[i] = self.gates.CNOT(
                self.phi[i], self.phi[k], t_cnot, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
            )
            self.phi[i] = self.phi[i] - np.pi/2
        else:
            # Control i
            self._mp[i] = self.gates.CNOT_inv(
                self.phi[i], self.phi[k], t_cnot, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
            )

            self.phi[i] = self.phi[i] + np.pi/2 + np.pi

            # Target k
            self.phi[k] = self.phi[k] + np.pi/2

        # Bookkeeping
        self._s += 2

        # Case: All gates in this gatetime have been applied
        if self._s == self.nqubit:
            self._update_mp_list()
        return

    def _update_mp_list(self):
        self._mp_list.append(self._mp)
        self._mp = [1 for i in range(self.nqubit)]
        self._s = 0

    def reset(self):
        """ Reset the circuit to the initial state. """
        self.phi = [0 for i in range(self.nqubit)]
        self._s = 0
        self._backend = self._BackendClass(self.nqubit)
        self._mp = [1 for i in range(self.nqubit)]
        self._mp_list = []
        self.phi = [0 for i in range(self.nqubit)]


class StandardCircuit(AlternativeCircuit):
    """ Class with the same interface as Circuit but built on top of the AlternativeCircuit. Is used as baseline
        for benchmarking Circuits/Backends.
    """

    def __init__(self, nqubit: int, depth: int, gates: Gates):
        super(StandardCircuit, self).__init__(nqubit=nqubit, gates=gates, BackendClass=StandardBackend)


class EfficientCircuit(AlternativeCircuit):
    """ Class with the same interface as Circuit but built on top of the AlternativeCircuit. Separates the matrix
        products in chunks, and contracts them with the statevector.
    """

    def __init__(self, nqubit: int, depth: int, gates: Gates):
        super(EfficientCircuit, self).__init__(nqubit=nqubit, gates=gates, BackendClass=EfficientBackend)


class OneCircuit(AlternativeCircuit):
    """ Class with the same interface as Circuit but built on top of the AlternativeCircuit. Is optimized for the
        Hadamard inv QFT circuit, which contains many 1s.
    """

    def __init__(self, nqubit: int, depth: int, gates: Gates):
        super(OneCircuit, self).__init__(nqubit=nqubit, gates=gates, BackendClass=BackendForOnes)

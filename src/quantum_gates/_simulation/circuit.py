"""
This module implements the base class to perform noisy quantum simulations with noisy gates approach
"""
import numpy as np
import itertools
import functools as ft

from .._gates.gates import Gates
from .backend import StandardBackend, EfficientBackend, BackendForOnes, BinaryBackend

class StepwiseCircuit(object):
    """Circuit class that applies each gate immediately to the statevector
    rather than batching gates per layer.

    Args:
        nqubit (int): Number of qubits.
        gates (Gates): Gate set used to construct noisy operations.
        psi0 (np.ndarray): Initial statevector (shape (2**nqubit,)).

    Attributes:
        nqubit (int): Number of qubits.
        gates (Gates): Noisy gate definitions.
        phi (list[float]): Phase memory for each qubit, used for virtual Rz gates.
        psi (np.ndarray): Current statevector after applying gates so far.
        history (list[np.ndarray]): Optional trajectory of statevectors.
    """

    def __init__(self, nqubit: int, gates: Gates, psi0: np.ndarray, track_history: bool = False):
        # --- Input validation ---
        if psi0.shape != (2**nqubit,):
            raise ValueError(
                f"psi0 must have shape {(2**nqubit,)}, but got {psi0.shape}"
            )

        self.nqubit = nqubit
        self.gates = gates

        # phase accumulator per qubit (like in Circuit)
        self.phi = [0.0 for _ in range(nqubit)]

        # statevector initialized to psi0
        self.psi = psi0.astype(complex)

        # optional storage of intermediate statevectors
        self.track_history = track_history
        self.history = [self.psi.copy()] if track_history else None

    def statevector(self):
        """Return the current statevector."""
        return self.psi
    
    def display(self):
        """Display current statevector."""
        print(self.psi)
        
    def _embed_single(self, gate, target):
        ops = [gate if q == target else np.eye(2) for q in range(self.nqubit)]
        return self._kron_all(ops)

    def _embed_two(self, gate, q0, q1):
        """
        Embed a 4x4 two-qubit gate into the full 2^n x 2^n Hilbert space.
        """
        return self._kron_with_two(gate, q0, q1)


    def _kron_all(self, ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def _kron_with_two(self, gate, q0, q1):
        """
        Embed a 4x4 two-qubit gate acting on qubits q0 and q1
        into the full 2^n x 2^n Hilbert space.
        Optimized for adjacent qubits, falls back to general embedding otherwise.
        """
        n = self.nqubit
        q0, q1 = sorted([q0, q1])  # ensure q0 < q1

        # ✅ Fast path: adjacent qubits (most common case)
        if q1 == q0 + 1:
            ops = []
            for idx in range(n):
                if idx == q0:
                    ops.append(gate)
                elif idx == q1:
                    # handled together with q0
                    continue
                else:
                    ops.append(np.eye(2))
            return self._kron_all(ops)

        # ❌ Slow path: general case (non-adjacent qubits)
        G = gate.reshape(2, 2, 2, 2)
        full_op = np.zeros((2,) * (2*n), dtype=complex)

        for bits_in in itertools.product([0, 1], repeat=n):
            for bits_out in itertools.product([0, 1], repeat=n):
                match_rest = all(bits_in[k] == bits_out[k] for k in range(n) if k not in (q0, q1))
                if match_rest:
                    in_idx = (bits_in[q0], bits_in[q1])
                    out_idx = (bits_out[q0], bits_out[q1])
                    full_op[bits_out + bits_in] = G[out_idx + in_idx]

        return full_op.reshape(2**n, 2**n)

    
    def _update_state(self, full_gate_matrix):
        """Apply a full 2^n x 2^n gate to the current statevector."""
        self.psi = full_gate_matrix @ self.psi
        if self.track_history:
            self.history.append(self.psi.copy())
        return self.psi
    
    def apply_single(self, gate, i):
        full_gate = self._embed_single(gate, i)
        return self._update_state(full_gate)
    
    def apply_two(self, gate, q0, q1):
        full_gate = self._embed_two(gate, q0, q1)
        return self._update_state(full_gate)

    def I(self, i: int):
        """Apply identity gate on qubit i."""
        Id = np.eye(2, dtype=complex)
        return self.apply_single(Id, i)

    def Rz(self, i: int, theta: float):
        """Apply virtual Rz rotation immediately to qubit i."""
        self.phi[i] += theta
        rz_gate = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(+1j * theta / 2)]
        ], dtype=complex)
        return self.apply_single(rz_gate, i)

    def X(self, i: int, p: float, T1: float, T2: float):
        """Apply noisy X gate with depolarizing + relaxation errors."""
        gate = self.gates.X(-self.phi[i], p, T1, T2)
        return self.apply_single(gate, i)

    def SX(self, i: int, p: float, T1: float, T2: float):
        """Apply noisy SX gate with depolarizing + relaxation errors."""
        gate = self.gates.SX(-self.phi[i], p, T1, T2)
        return self.apply_single(gate, i)

    def relaxation(self, i: int, Dt: float, T1: float, T2: float):
        """Apply relaxation (amplitude + phase damping) noise channel."""
        gate = self.gates.relaxation(Dt, T1, T2)
        return self.apply_single(gate, i)

    def depolarizing(self, i: int, Dt: float, p: float):
        """Apply single-qubit depolarizing channel."""
        gate = self.gates.depolarizing(Dt, p)
        return self.apply_single(gate, i)

    def bitflip(self, i: int, tm: float, rout: float):
        """Apply readout error channel (bitflip noise)."""
        gate = self.gates.bitflip(tm, rout)
        return self.apply_single(gate, i)

    def CNOT(self, i: int, k: int, t_int: float, p_i_k: float, p_i: float,
            p_k: float, T1_ctr: float, T2_ctr: float, T1_trg: float, T2_trg: float):
        """Apply noisy CNOT gate on qubits i (control) and k (target)."""
        if i < k:
            gate = self.gates.CNOT(self.phi[i], self.phi[k],
                                t_int, p_i_k, p_i, p_k,
                                T1_ctr, T2_ctr, T1_trg, T2_trg)
            self.phi[i] -= np.pi / 2
        else:
            gate = self.gates.CNOT_inv(self.phi[i], self.phi[k],
                                    t_int, p_i_k, p_i, p_k,
                                    T1_ctr, T2_ctr, T1_trg, T2_trg)
            self.phi[i] += np.pi / 2 + np.pi
            self.phi[k] += np.pi / 2
        return self.apply_two(gate, i, k)

    def ECR(self, i: int, k: int, t_ecr: float, p_i_k: float, p_i: float,
            p_k: float, T1_ctr: float, T2_ctr: float, T1_trg: float, T2_trg: float):
        """Apply noisy echoed cross-resonance (ECR) gate."""
        if i < k:
            gate = self.gates.ECR(self.phi[i], self.phi[k],
                                t_ecr, p_i_k, p_i, p_k,
                                T1_ctr, T2_ctr, T1_trg, T2_trg)
        else:
            gate = self.gates.ECR_inv(self.phi[k], self.phi[i],
                                    t_ecr, p_i_k, p_i, p_k,
                                    T1_ctr, T2_ctr, T1_trg, T2_trg)
        return self.apply_two(gate, i, k)


class Circuit(object):
    """ Class that allows to define custom noisy quantum circuit and apply it on a given initial quantum state.

    Args:
        nqubit (int): number of qubits
        depth (int): depth of the circuit

    Example:
        .. code:: python

            from quantum_gates.circuits import Circuit
            from quantum_gates.gates import standard_gates

            # The depth has to be set correctly for this class.
            circuit = Circuit(nqubit=2, depth=1, gates=standard_gates)

            # We apply gates for one timestep.
            circuit.X(i=0, ...)
            circuit.I(i=1)

            # Evaluate the statevector
            psi1 = circuit.statevector(psi0=np.array([1, 0, 0, 0]))  # Gives [0, 0, 1, 0]

    Attributes:
        nqubit (int): Number of qubits.
        depth (int): Depth of the circuit.
        j (int): Index of the current matrix product during the building of a list of matrix products.
        s (int): Number of qubits on which  gates were applied during this building of the current matrix product.
        phi (list[float]): Phases of the qubits.
        circuit (list[list[np.array]]): Quantum circuit made from the sampled noisy quantum gates.
    """

    def __init__(self, nqubit: int, depth: int, gates: Gates):
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

    def CNOT(self, i: int, k: int, t_int: float, p_i_k: float, p_i: float, p_k: float, T1_ctr: float,
             T2_ctr: float, T1_trg: float, T2_trg: float):
        """
        Apply CNOT two-qubit noisy quantum gate with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            i: index of the control qubit (int)
            k: index of the target qubit (int)
            t_int: CNOT gate time in ns (double)
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
                    self.phi[i], self.phi[k], t_int, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )
                self.phi[i] = self.phi[i] - np.pi/2

            else:
                self.circuit[i][self.j] = self.gates.CNOT_inv(
                    self.phi[i], self.phi[k], t_int, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )
                self.phi[i] = self.phi[i] + np.pi/2 + np.pi
                self.phi[k] = self.phi[k] + np.pi/2
            self.s = self.s+2

        elif self.s == self.nqubit:
            self.s = 2
            self.j = self.j+1

            if i < k:
                self.circuit[i][self.j] = self.gates.CNOT(
                    self.phi[i], self.phi[k], t_int, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )
                self.phi[i] = self.phi[i] - np.pi/2

            else:
                self.circuit[i][self.j] = self.gates.CNOT_inv(
                    self.phi[i], self.phi[k], t_int, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )
                self.phi[i] = self.phi[i] + np.pi/2 + np.pi
                self.phi[k] = self.phi[k] + np.pi/2

    def ECR(self, i: int, k: int, t_ecr: float, p_i_k: float, p_i: float, p_k: float, T1_ctr: float,
             T2_ctr: float, T1_trg: float, T2_trg: float):
        """
        Apply ECR two-qubit noisy quantum gate with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            i: index of the control qubit (int)
            k: index of the target qubit (int)
            t_ecr: ECR gate time in ns (double)
            p_i_k: ECR depolarizing error probability (double)
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
                self.circuit[i][self.j] = self.gates.ECR(
                    self.phi[i], self.phi[k], t_ecr, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )

            else:
                self.circuit[i][self.j] = self.gates.ECR_inv(
                    self.phi[k], self.phi[i], t_ecr, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )
            self.s = self.s+2

        elif self.s == self.nqubit:
            self.s = 2
            self.j = self.j+1

            if i < k:
                self.circuit[i][self.j] = self.gates.ECR(
                    self.phi[i], self.phi[k], t_ecr, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                ) 

            else:
                self.circuit[i][self.j] = self.gates.ECR_inv(
                    self.phi[k], self.phi[i], t_ecr, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
                )
                

    def reset(self):
        """ Reset the circuit to the initial state. """
        self.j = 0
        self.s = 0
        self.circuit = [[1 for i in range(self.depth)] for j in range(self.nqubit)]

        # TODO: The last line was originally not there. Check if it should and whether it has a positive effect.
        self.phi = [0 for i in range(self.nqubit)]


class AlternativeCircuit(object):
    """ Allows to build a circuit and outsource the computations to an optimized backend.

    In this version, we provide a backend for the evaluation of the tensor contractions that are performed in the
    creation of the propagator.

    Args:
        nqubit (int): Number of qubits.
        gates (int): Gateset from which the noisy quantum gates should be sampled.
        backendClass (Union[StandardBackend, EfficientBackend]): Backend for performing the computations.

    Example:
        .. code:: python

            from quantum_gates.circuits import AlternativeCircuit
            from quantum_gates.backends import EfficientBackend
            from quantum_gates.gates import standard_gates

            circuit = AlternativeCircuit(
                nqubit=2,
                gates=standard_gates,
                BackendClass=EfficientBackend
            )

    Attributes:
        nqubit (int): Number of qubits.
        gates (int): Gateset from which the noisy quantum gates should be sampled.
        phi (list[float]): Phases of the qubits.
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
        """Applies a single qubit gate to qubit i.

        If the circuit snippet is full, then matrix product is appended and the bookkeeping is reset.
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
        """Compute the output statevector of the noisy quantum circuit, psi1 = U psi0.
        """
        # Handle the trivial case in which no gates were applied.
        if len(self._mp_list) == 0:
            return psi0
        return self._backend.statevector(self._mp_list, psi0)

    def I(self, i: int):
        """Apply identity gate on qubit i

        Args:
            i: index of the qubit

        Returns:
             None
        """
        identity_matrix = np.array([[1, 0], [0, 1]])
        self.apply(gate=identity_matrix, i=i)

    def Rz(self, i: int, theta: float):
        """Update the phase to implement virtual Rz(theta) gate on qubit i

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
        """Apply relaxation noise gate on qubit i. Add on idle-qubits.

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
        """Apply depolarizing noise gate on qubit i. Add on idle-qubits.

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

    def CNOT(self, i: int, k: int, t_int: float, p_i_k: float, p_i: float, p_k: float, T1_ctr: float,
             T2_ctr: float, T1_trg: float, T2_trg: float):
        """
        Apply CNOT two-qubit noisy quantum gate with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            i: index of the control qubit (int)
            k: index of the target qubit (int)
            t_int: CNOT gate time in ns (double)
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
                self.phi[i], self.phi[k], t_int, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
            )
            self.phi[i] = self.phi[i] - np.pi/2
        else:
            # Control i
            self._mp[i] = self.gates.CNOT_inv(
                self.phi[i], self.phi[k], t_int, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
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
    
    def ECR(self, i: int, k: int, t_ecr: float, p_i_k: float, p_i: float, p_k: float, T1_ctr: float,
             T2_ctr: float, T1_trg: float, T2_trg: float):
        """
        Apply ECR two-qubit noisy quantum gate with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            i: index of the control qubit (int)
            k: index of the target qubit (int)
            t_ecr: ECR gate time in ns (double)
            p_i_k: ECR depolarizing error probability (double)
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
            self._mp[i] = self.gates.ECR(
                self.phi[i], self.phi[k], t_ecr, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
            )
        else:
            # Control i
            self._mp[i] = self.gates.ECR_inv(
                self.phi[k], self.phi[i], t_ecr, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
            )

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


class BinaryCircuit(object):
    """ Allows to build a circuit to deal with non linear topologies of the devices.

    In this version, we provide a backend that must be the BinaryBackend because it provide the execution of the circuit with the algorithm that goes beyond
    the linear topology of a device. It's full general.

    Args:
        nqubit (int): Number of qubits.
        depth (int): Depth of the circuit (Doesn't use, but leave here to follow the structure of the previous Circuit Class)
        gates (int): Gateset from which the noisy quantum gates should be sampled.
        BackendClass (class): Will be ignored as we always use the BinaryBackend in the BinaryCircuit.
        qubit_layout (np.array): Qubit layout in the case of non-linear topology. If not set, a linear topology will be assumed.

    Example:
        .. code:: python

            from quantum_gates.circuits import BinaryCircuit
            from quantum_gates.gates import standard_gates

            n_qubit = 3
            qubit_layout = [0,1,2]

            circuit = BinaryCircuit(
                nqubit=n_qubit,
                depth=1,
                gates=standard_gates,
                qubit_layout=qubit_layout
            )

            X, CNOT = ...

            #apply gate on the circuit
            circ.update_circuit_list(gate=X, qubit = [0])
            circ.update_circuit_list(gate=CNOT, qubit = [0,2])

            # calculate the statevector
            psi0 = [1] + [0] * (2**n_qubit-1)
            psi1 = circ.statevector(psi0 = psi0)

    """

    def __init__(self,
                 nqubit: int,
                 depth: int,
                 gates: Gates,
                 BackendClass: type(BinaryBackend) = BinaryBackend,
                 qubit_layout: np.array=None):
        self.nqubit: int = nqubit                   # Number of qubits
        self.gates: Gates = gates                   # Gate set to be used (specifies the noisy behaviour)
        self._backend = BinaryBackend(nqubit)       # Backend for the computations
        self._BackendClass = BinaryBackend          # Always BinaryBackend
        self.qubit_layout = qubit_layout if qubit_layout else np.arange(self.nqubit)

        # Bookkeeping
        self.phi = [0 for i in range(nqubit)]       # Phases
        # List that contain all the info about the gates applied in the circuit and in which qubits
        self._info_gates_list: list[list[np.ndarray, np.ndarray]] = []

    def apply(self, gate: np.ndarray, i: int, j: int=-1):
        """Update the list of info_gates of the circuit

        Args:
            gate (np.array): The matrix representation of the noisy gate
            i (int): index of the first qubit on which the gate is applied
            j (int): index of the second qubit in case of a two qubit gate
        """
        if not isinstance(gate, np.ndarray):
            raise ValueError(f"Circuit.update_circuit_list() expected gate to be a numpy array but found type {type(gate)}.")
        
        if gate.shape == (4, 4) and j == -1:
            raise ValueError(f"Circuit.update_circuit_list() expected i and j to be set for a two qubit gate.")

        the_info = [gate, [i,j]]
        self._info_gates_list.append(the_info)
            
    def statevector(self, psi0: np.array) -> np.array:
        """Compute the output statevector of the noisy quantum circuit, psi1 = U psi0.
        """
        # Handle the trivial case in which no gates were applied.
        if len(self._info_gates_list) == 0:
            return psi0
        return self._backend.statevector(
            mp_list=self._info_gates_list,
            psi0=psi0,
            qubit_layout=self.qubit_layout,
        )
    
    def I(self, i: int):
        """Apply identity gate on qubit i

        Args:
            i: index of the qubit

        Returns:
             None
        """
        identity_matrix = np.array([[1, 0], [0, 1]])
        self.apply(gate=identity_matrix, i=i)

    def Rz(self, i: int, theta: float):
        """Update the phase to implement virtual Rz(theta) gate on qubit i

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
        """Apply relaxation noise gate on qubit i. Add on idle-qubits.

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
        """Apply depolarizing noise gate on qubit i. Add on idle-qubits.

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

    def CNOT(self, i: int, k: int, t_int: float, p_i_k: float, p_i: float, p_k: float, T1_ctr: float,
             T2_ctr: float, T1_trg: float, T2_trg: float):
        """
        Apply CNOT two-qubit noisy quantum gate with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            i: index of the control qubit (int)
            k: index of the target qubit (int)
            t_int: CNOT gate time in ns (double)
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
            the_gate = self.gates.CNOT(
                self.phi[i], self.phi[k], t_int, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
            )
            self.phi[i] = self.phi[i] - np.pi/2

            self.apply(gate=the_gate, i=i, j=k)
        else:
            # Control i
            the_gate = self.gates.CNOT_inv(
                self.phi[i], self.phi[k], t_int, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
            )

            self.phi[i] = self.phi[i] + np.pi/2 + np.pi

            # Target k
            self.phi[k] = self.phi[k] + np.pi/2

            self.apply(gate=the_gate, i=i, j=k)

        return

    def ECR(self, i: int, k: int, t_ecr: float, p_i_k: float, p_i: float, p_k: float, T1_ctr: float,
             T2_ctr: float, T1_trg: float, T2_trg: float):
        """
        Apply ECR two-qubit noisy quantum gate with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            i: index of the control qubit (int)
            k: index of the target qubit (int)
            t_ecr: ECR gate time in ns (double)
            p_i_k: ECR depolarizing error probability (double)
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
            the_gate = self.gates.ECR(
                self.phi[i], self.phi[k], t_ecr, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
            )

            self.apply(gate=the_gate, i=i, j=k)

        else:
            # Control i
            the_gate = self.gates.ECR_inv(
                self.phi[k], self.phi[i], t_ecr, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg
            )

            self.apply(gate=the_gate, i=k, j=i)

        return

    def reset(self):
        """ Reset the circuit to the initial state. """
        self.phi = [0 for i in range(self.nqubit)]
        self._backend = self._BackendClass(self.nqubit)
        self._info_gates_list = []
        self.phi = [0 for i in range(self.nqubit)]


class StandardCircuit(AlternativeCircuit):
    """Class with the same interface as Circuit but built on top of the AlternativeCircuit.

    Is used as baseline for benchmarking Circuits/Backends.
    """

    def __init__(self, nqubit: int, depth: int, gates: Gates):
        super(StandardCircuit, self).__init__(nqubit=nqubit, gates=gates, BackendClass=StandardBackend)


class EfficientCircuit(AlternativeCircuit):
    """Class with the same interface as Circuit but built on top of the AlternativeCircuit.

    Separates the matrix products in chunks, and contracts them with the statevector.
    """

    def __init__(self, nqubit: int, depth: int, gates: Gates):
        super(EfficientCircuit, self).__init__(nqubit=nqubit, gates=gates, BackendClass=EfficientBackend)


class OneCircuit(AlternativeCircuit):
    """Class with the same interface as Circuit but built on top of the AlternativeCircuit.

    Is optimized for the Hadamard inv QFT circuit, which contains many 1s.
    """

    def __init__(self, nqubit: int, depth: int, gates: Gates):
        super(OneCircuit, self).__init__(nqubit=nqubit, gates=gates, BackendClass=BackendForOnes)

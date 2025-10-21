"""
This module implements the base class to perform noisy quantum simulations with noisy gates approach
"""
import numpy as np
import itertools
import functools as ft

from .._gates.gates import Gates
from .backend import StandardBackend, EfficientBackend, BackendForOnes, BinaryBackend


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
    
    ### NEW CODE HERE ###

    def mid_measurement(self, psi0: np.ndarray, qubit_list=None) -> tuple[np.ndarray, list[int]]:
        """
        Perform a projective mid-circuit measurement on the given qubits.

        Args:
            psi0 (np.ndarray): Initial statevector of shape (2^n,).
            qubit_list (list[int] or None): List of qubit indices (0 = least significant).
                - None → measure all qubits
                - [i1, i2, ...] → measure only those qubits
                - [] or invalid input → raises ValueError

        Returns:
            psi (np.ndarray): Collapsed and renormalized statevector.
            result (list[int]): Measurement outcomes for each qubit in qubit_list.
        """
        
        print("CIRCUIT: Mid-circuit measurement called on qubits:", qubit_list)
        
        dim = psi0.shape[0]
        n = int(np.log2(dim))

        # --- Handle qubit_list argument ---
        if qubit_list is None:
            qubit_list = list(range(n))   # measure all qubits if not specified
        elif not isinstance(qubit_list, (list, tuple)):
            raise ValueError("qubit_list must be a list of qubit indices or None.")
        elif len(qubit_list) == 0:
            raise ValueError("qubit_list cannot be empty. Use None to measure all qubits.")
        elif any((not isinstance(q, int)) or (q < 0) or (q >= n) for q in qubit_list):
            raise ValueError(f"qubit_list must contain valid qubit indices in [0, {n-1}].")
        elif len(set(qubit_list)) != len(qubit_list):
            raise ValueError("qubit_list contains duplicate qubit indices.")

        collapsed = psi0.copy()
        result = []

        for target_qubit in qubit_list:
            # 1. Compute Born probabilities
            probs = [0.0, 0.0]
            for idx, amp in enumerate(collapsed):
                bit = (idx >> target_qubit) & 1
                probs[bit] += abs(amp)**2

            probs = np.array(probs) / sum(probs)

            # 2. Sample measurement outcome
            outcome = np.random.choice([0, 1], p=probs)
            result.append(outcome)

            # 3. Collapse
            for idx in range(dim):
                if ((idx >> target_qubit) & 1) != outcome:
                    collapsed[idx] = 0.0 + 0.0j

            # 4. Renormalize
            norm = np.linalg.norm(collapsed)
            if norm > 0:
                collapsed /= norm

        return collapsed, result
    
    import numpy as np

    def reset(self, psi0: np.ndarray, p: float, T1: float, T2: float, qubit_list=None) -> np.ndarray:
        collapsed, outcomes = self.mid_measurement(psi0, qubit_list)

        for q, outcome in zip(qubit_list, outcomes):
            if outcome == 1:
                # apply noisy X on this qubit to collapsed
                X_gate = self.gates.X(-self.phi[q], p, T1, T2)   # matrix form of noisy X
                collapsed = X_gate @ collapsed
            else:
                I_gate = self.gates.I()  # identity
                collapsed = I_gate @ collapsed

        return collapsed

    def statevector_readout(self, psi0) -> np.array:
        #print("Readout statevector.")
        #print(psi0)
        return psi0
    
    ### NEW CODE HERE ###
    
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
    
    def reset_circuit(self):
        """ Reset the circuit to the initial state. """
        self.phi = [0 for i in range(self.nqubit)]
        self._s = 0
        self._backend = self._BackendClass(self.nqubit)
        self._mp = [1 for i in range(self.nqubit)]
        self._mp_list = []

    
    ##NEW CODE HERE ##    
    def mid_measurement(self, psi0: np.ndarray, device_param, add_bitflip=False, qubit_list=None, cbit_list=None) -> tuple[np.ndarray, list[int]]:
        """
        Perform a projective mid-circuit measurement on the given qubits.

        Args:
            psi0 (np.ndarray): Initial statevector of shape (2^n,).
            qubit_list (list[int] or None): List of qubit indices (0 = least significant).
                - None → measure all qubits
                - [i1, i2, ...] → measure only those qubits
                - [] or invalid input → raises ValueError
            cbit_list (list[int] or None): List of classical bit indices for storing results.
                - None → map qubit i to classical bit i
                - [j1, j2, ...] → assign result of qubit_list[k] to cbit_list[k]
                - Invalid input → raises ValueError

        Returns:
            psi (np.ndarray): Collapsed and renormalized statevector.
            result (list[int]): Measurement outcomes mapped to classical bits.
        """
        print("ALT - CIRCUIT: Mid-circuit measurement called on qubits:", qubit_list)

        dim = psi0.shape[0]
        n = int(np.log2(dim))

        # Unpack device parameters
        T1, T2, p, rout, p_int, t_int, tm, dt = (
            device_param["T1"],
            device_param["T2"],
            device_param["p"],
            device_param["rout"],
            device_param["p_int"],
            device_param["t_int"],
            device_param["tm"],
            device_param["dt"][0]
        )

        # --- Handle qubit_list ---
        if qubit_list is None:
            qubit_list = list(range(n))   # measure all qubits if not specified
        elif not isinstance(qubit_list, (list, tuple)):
            raise ValueError("qubit_list must be a list of qubit indices or None.")
        elif len(qubit_list) == 0:
            raise ValueError("qubit_list cannot be empty. Use None to measure all qubits.")
        elif any((not isinstance(q, int)) or (q < 0) or (q >= n) for q in qubit_list):
            raise ValueError(f"qubit_list must contain valid qubit indices in [0, {n-1}].")
        elif len(set(qubit_list)) != len(qubit_list):
            raise ValueError("qubit_list contains duplicate qubit indices.")

        # --- Handle cbit_list ---
        if cbit_list is None:
            cbit_list = list(range(len(qubit_list)))  # default mapping
        elif not isinstance(cbit_list, (list, tuple)):
            raise ValueError("cbit_list must be a list of classical bit indices or None.")
        elif len(cbit_list) != len(qubit_list):
            raise ValueError("cbit_list must have the same length as qubit_list.")
        elif any((not isinstance(c, int)) or (c < 0) for c in cbit_list):
            raise ValueError("cbit_list must contain valid non-negative integer indices.")
        elif len(set(cbit_list)) != len(cbit_list):
            raise ValueError("cbit_list contains duplicate classical bit indices.")

        # --- Perform measurement ---
        collapsed = psi0.copy()
        raw_results = {}

        for target_qubit, target_cbit in zip(qubit_list, cbit_list):
            # 1. Compute Born probabilities
            probs = [0.0, 0.0]
            for idx, amp in enumerate(collapsed):
                bit = (idx >> (n - 1 - target_qubit)) & 1
                probs[bit] += abs(amp) ** 2
            probs = np.array(probs) / sum(probs)
            print("Born probabilities for qubit", target_qubit, ":", probs)
            # 2. Sample measurement outcome

            outcome = np.random.choice([0, 1], p=probs)
            raw_results[target_cbit] = outcome

            # 3. Collapse the state
            for idx in range(dim):
                if ((idx >> (n - 1 - target_qubit)) & 1) != outcome:
                    collapsed[idx] = 0.0 + 0.0j

            # 4. Renormalize
            norm = np.linalg.norm(collapsed)
            if norm > 0:
                collapsed /= norm


        # Sort by classical bit index for final output
        result = [raw_results[c] for c in sorted(raw_results.keys())]

        print("Statevector before mid-circuit measurement:", psi0)
        print("Collapsed statevector after mid-circuit measurement:", collapsed)
        print("Measurement outcomes (mapped to cbits):", result)
        return collapsed, result


    def reset_qubits(self, psi0: np.ndarray, p: float, T1: float, T2: float, qubit_list=None):
        """
        Reset one or more qubits:
        - Measure qubits in qubit_list
        - For each qubit measured as 1, apply X to reset to |0>

        Args:
            psi0 (np.ndarray): Input statevector (length 2^n).
            qubit_list (list[int] or None): Qubits to reset.
                - None → reset all qubits
                - [i1, i2, ...] → reset only those qubits

        Returns:
            psi (np.ndarray): Statevector after reset.
            outcomes (list[int]): Measurement outcomes for each qubit.
        """
        # --- 1. Measure specified qubits ---
        collapsed, outcomes = self.mid_measurement(psi0, qubit_list)

        dim = collapsed.shape[0]
        n = int(np.log2(dim))
        self.reset_circuit()
        # --- 2. For each qubit with outcome=1, apply X ---
        for q, outcome in zip(qubit_list, outcomes):
            if outcome == 1:
                self.X(i=q, p=p, T1=T1, T2=T2)
            else: self.I(i=q)
        
        psi = self.statevector(collapsed)
        self.reset_circuit()

        return psi, outcomes


    def statevector_readout(self, psi0) -> np.array:
        return psi0
    
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

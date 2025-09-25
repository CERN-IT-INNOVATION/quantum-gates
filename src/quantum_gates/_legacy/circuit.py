from numpy import kron, array, pi
import numpy as np
import functools as ft
from .gates import relaxation, bitflip, depolarizing, X, SX, CNOT, CNOT_inv


class LegacyCircuit(object):
    """Allows to define custom noisy quantum circuit and apply it on a given initial quantum state.

    Parameters:
        n (int): Number of qubits.
        d (int): Depth of the circuit.

    Example:
        .. code:: python

           from quantum_gates.circuits import LegacyCircuit

           n = 2   # Number of qubits
           d = 1   # Depth of the circuit

           circuit = LegacyCircuit(n, d)

           # Apply gates
           circuit.X(i=0, p=..., T1=..., T2=...)
           circuit.I(i=1)

           # Statevector simulation
           psi0 = np.array([1, 0, 0, 0])
           psi1 = circuit.statevector(psi0)  # Gives [0  0  1  0]

    Attributes:
        nqubit (int): Number of qubits.
        depth (int): Depth of the circuit.
        j (int): Index of the timestep which we currently apply gates on.
        s (in): Number of qubits already treated in the current timestep.
        phi (list): Phases of the qubits, this is kept track of because Rz gates are virtual in Qiskit.
        circuit (list[list[np.array]]): Representation of the circuit, contains the sampled gates.
    """
    def __init__(self, n: int, d: int):
        self.nqubit = n
        self.depth = d
        self.j = 0
        self.s = 0
        self.phi = [0 for i in range(0, n)]
        self.circuit = [[1 for i in range(0, d)] for j in range(0, n)]
    
    def display(self):
        """Display the noisy quantum circuit.
        
        Returns:
             None
        """
        print(self.circuit)
    
    def apply(self, gate, i):
        """Apply an arbitrary gate.
        
        Args:
            gate: arbitrary gate to apply (array)
            i: index of the qubit (int)
        
        Returns:
             None
        """
        if self.s < self.nqubit:
            self.circuit[i][self.j] = gate
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = gate
        
    def statevector(self, psi0) -> np.array:
        """Compute the output statevector of the noisy quantum circuit.
        
        Args:
             psi0 (np.array): initial statevector, must be 1 x 2^n
       
        Returns:
             Output statevector as array.
        """        
        self.circuit = array(self.circuit, dtype=object)
        tensor_prod = [ft.reduce(kron, self.circuit[:, i]) for i in range(0, self.depth)]
        matrix_prod = tensor_prod[0]
        for i in range(1, self.depth):
            matrix_prod = tensor_prod[i] @ matrix_prod
        psi = matrix_prod @ psi0
        return psi

    def I(self, i: int):
        """Apply identity gate on qubit i.
        
        Args:
            i (int): Index of the qubit.
        
        Returns:
             None
        """
        Id = array(
            [[1, 0],
             [0, 1]]
        )
        
        if self.s < self.nqubit:
            self.circuit[i][self.j] = Id
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = Id

    def Rz(self, i: int, theta: float):
        """Update the phase to implement virtual Rz(theta) gate on qubit i.
        
        Args:
            i (int): Index of the qubit.
            theta (float): Angle of rotation on the Bloch sphere.
        
        Returns:
             None
        """
        self.phi[i] = self.phi[i] + theta
    
    def bitflip(self, i: int, tm: float, rout: float):
        """Apply bitflip (bitflip) noise gate on qubit i. Add before measurements or after initial state preparation.
        
        Args:
            i (int): Index of the qubit
            tm (float): Measurement time in ns.
            rout (float): Readout error
        
        Returns:
             None
        """        
        if self.s < self.nqubit:
            self.circuit[i][self.j] = bitflip(tm, rout)
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            
            self.s = 1
            
            self.j = self.j+1
            
            self.circuit[i][self.j] = bitflip(tm,rout)

    def relaxation(self, i: int, Dt: float, T1: float, T2: float):
        """Apply relaxation noise gate on qubit i. Add on idle-qubits.
        
        Args:
            i (int): Index of the qubit.
            Dt (int): Idle time in ns.
            T1 (int): Qubit's amplitude damping time in ns.
            T2 (int): Qubit's dephasing time in ns.
        
        Returns:
             None
        """ 
        if self.s < self.nqubit:
            self.circuit[i][self.j] = relaxation(Dt, T1, T2)
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = relaxation(Dt, T1, T2)

    def depolarizing(self, i: int, Dt: float, p: float):
        """Apply depolarizing noise gate on qubit i. Add on idle-qubits.
        
        Args:
            i (int): Index of the qubit.
            Dt (float): Idle time in ns.
            p (float): Single-qubit depolarizing error probability.
        
        Returns:
             None
        """ 
        if self.s < self.nqubit:
            self.circuit[i][self.j] = depolarizing(Dt, p)
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = depolarizing(Dt,p)
        
    def X(self, i: int, p: float, T1: float, T2: float) -> np.array:
        """Apply X single-qubit noisy quantum gate with depolarizing and relaxation errors during the unitary evolution.

        Args:
            i (int): Index of the qubit.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.

        Returns:
              None
        """
        if self.s < self.nqubit:
            self.circuit[i][self.j] = X(-self.phi[i], p, T1, T2)
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = X(-self.phi[i], p, T1, T2)

    def SX(self, i: int, p: float, T1: float, T2: float):
        """Apply SX single-qubit noisy quantum gate with depolarizing and relaxation errors during the unitary evolution.

        Args:
            i (int): Index of the qubit.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.

        Returns:
              None
        """
        if self.s < self.nqubit:
            self.circuit[i][self.j] = SX(-self.phi[i], p, T1, T2)
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = SX(-self.phi[i], p, T1, T2)

    def CNOT(self, i: int, k: int, t_cnot: float, p_i_k: float, p_i: float, p_k: float, T1_ctr: float,
             T2_ctr: float, T1_trg: float, T2_trg: float):
        """ Apply CNOT two-qubit noisy quantum gate with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            i (int): Index of the control qubit.
            k (int): Index of the target qubit.
            t_cnot (float): CNOT gate time in ns.
            p_i_k (float): CNOT depolarizing error probability.
            p_i (float): Control single-qubit depolarizing error probability.
            p_k (float): Target single-qubit depolarizing error probability.
            T1_ctr (float): Control qubit's amplitude damping time in ns.
            T2_ctr (float): Control qubit's dephasing time in ns.
            T1_trg (float): Target qubit's amplitude damping time in ns.
            T2_trg (float): Target qubit's dephasing time in ns.

        Returns:
              None
        """
        if self.s < self.nqubit:
            
            if i < k:
                self.circuit[i][self.j] = CNOT(self.phi[i], self.phi[k], t_cnot, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg)
                self.phi[i] = self.phi[i] - pi/2
                
            else: 
                self.circuit[i][self.j] = CNOT_inv(self.phi[i], self.phi[k], t_cnot, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg)
                self.phi[i] = self.phi[i] + pi/2 + pi
                self.phi[k] = self.phi[k] + pi/2
            self.s = self.s+2 
            
        elif self.s == self.nqubit:
            self.s = 2
            self.j = self.j+1
            
            if i < k:
                self.circuit[i][self.j] = CNOT(self.phi[i], self.phi[k], t_cnot, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg)
                self.phi[i] = self.phi[i] - pi/2
                
            else: 
                self.circuit[i][self.j] = CNOT_inv(self.phi[i], self.phi[k], t_cnot, p_i_k, p_i, p_k, T1_ctr, T2_ctr, T1_trg, T2_trg)
                self.phi[i] = self.phi[i] + pi/2 + pi
                self.phi[k] = self.phi[k] + pi/2

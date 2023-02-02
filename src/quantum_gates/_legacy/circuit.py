"""
This module implements the base class to perform noisy quantum simulations with noisy gates approach
"""

from numpy import kron, array, pi
import numpy as np
import functools as ft
from .gates import relaxation, bitflip, depolarizing, X, SX, CNOT, CNOT_inv


class Circuit(object):
    """
    Class that allows to define custom noisy quantum circuit and apply it on a given initial quantum state. 
    """
    def __init__(self, n: int, d: int):
        """
        Init method for the noisy quantum circuit
        
        Parameters:
            n: number of qubits
            d: depth of the circuit
        
        Returns:
            None
        """
        self.nqubit = n
        self.depth = d
        self.j = 0
        self.s = 0
        self.phi = [0 for i in range(0, n)]
        self.circuit = [[1 for i in range(0, d)] for j in range(0, n)]
    
    def display(self):
        """
        Display the noisy quantum circuit
        
        Returns:
             None
        """
        print(self.circuit)
    
    def apply(self, gate, i):
        """
        Apply an arbitrary gate
        
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
        """
        Compute the output statevector of the noisy quantum circuit
        
        Args:
             psi0: initial statevector, must be 1 x 2^n
       
        Returns:
             output statevector
        """        
        self.circuit = array(self.circuit,dtype=object)
        tensor_prod = [ft.reduce(kron, self.circuit[:, i]) for i in range(0, self.depth)]
        matrix_prod = tensor_prod[0]
        for i in range(1, self.depth):
            matrix_prod = tensor_prod[i] @ matrix_prod
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
        if self.s < self.nqubit:
            self.circuit[i][self.j] = bitflip(tm, rout)
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            
            self.s = 1
            
            self.j = self.j+1
            
            self.circuit[i][self.j] = bitflip(tm,rout)

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
        if self.s < self.nqubit:
            self.circuit[i][self.j] = relaxation(Dt, T1, T2)
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = relaxation(Dt, T1, T2)

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
        if self.s < self.nqubit:
            self.circuit[i][self.j] = depolarizing(Dt, p)
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = depolarizing(Dt,p)
        
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
        if self.s < self.nqubit:
            self.circuit[i][self.j] = X(-self.phi[i], p, T1, T2)
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = X(-self.phi[i], p, T1, T2)

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
        if self.s < self.nqubit:
            self.circuit[i][self.j] = SX(-self.phi[i], p, T1, T2)
            self.s = self.s+1
            
        elif self.s == self.nqubit:
            self.s = 1
            self.j = self.j+1
            self.circuit[i][self.j] = SX(-self.phi[i], p, T1, T2)

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

import numpy as np
from numpy import array, absolute, mean

from .circuit import Circuit
from .._utility.device_parameters import DeviceParameters


class MrAndersonSimulator():
    """Simulates a quantum circuit with the Noisy quantum gates approach.

    Note:
        This version is only meant for unit testing and the documentation
    """

    def run(self, qiskit_circ,
            backend,
            qubits_layout: list,
            psi0: np.array,
            shots: int,
            device_param: DeviceParameters):
        """Performs a run of the simulator.

        Takes as input a qiskit circuit on a given backend with a given qubits layout  and runs noisy quantum gates.

        Args:
            qiskit_circ: qiskit circuit (QuantumCircuit)
            backend: IMBQ backend (backend)
            qubits_layout: qubits layout with linear topology (list)
            psi0: initial state (array)
            shots: number of realizations (int)
            device_param: object that contains the measured noise (DeviceParameters)

        Returns:
            vector of probabilities and density matrix

        Note:
            Qubits layout must have a linear topology.
        """
        # Prepare variables
        nqubit = len(qubits_layout)
        t_qiskit_circ = qiskit_circ
        raw_data = t_qiskit_circ.data
        data = []
        prop = backend.properties()
        tm = prop.readout_length(0)

        # Load device parameters
        T1, T2, p, rout, p_cnot, t_cnot = device_param.get_as_tuple()

        # Preprocess of QuantumCircuit.data
        n_rz = 0
        data_measure = []
        swap_detector = [a for a in range(nqubit)]

        for i in range(t_qiskit_circ.__len__()):
            if raw_data[i][0].name =='cx':
                q_ctr = raw_data[i][1][0].index
                q_trg = raw_data[i][1][1].index
                if q_ctr in qubits_layout and q_trg in qubits_layout:
                    raw_data[i][1][0] = qubits_layout.index(q_ctr)
                    raw_data[i][1][1] = qubits_layout.index(q_trg)
                    data.append(raw_data[i])
                    
            elif raw_data[i][0].name =='measure':
                q = raw_data[i][1][0].index
                q = qubits_layout.index(q) 
                c = raw_data[i][2][0].index
                data_measure.append((q,c))
                
            else:
                q = raw_data[i][1][0].index
                if q in qubits_layout:
                    if raw_data[i][0].name == 'rz':
                        n_rz = n_rz + 1
                    if raw_data[i][0].name != 'measure' and raw_data[i][0].name != 'barrier':
                        raw_data[i][1][0] = qubits_layout.index(q) 
                        data.append(raw_data[i]) 
                        
        for i in range(len(data_measure)):
            swap_detector[data_measure[i][0]] = data_measure[i][1]

        # Initialize Circuit, depth without rz (in IBMQ devices rz gates are virtual --> noiseless)
        # Add measurements
        depth = len(data) - n_rz + 1
        circ = Circuit(nqubit, depth)
        
        # Read data and apply Noisy Quantum gates
        r = np.zeros((shots, 2**nqubit))

        for m in range(shots):
            
            for j in range(len(data)):
                
                if data[j][0].name == 'rz':
                    theta = float(data[j][0].params[0])
                    q = data[j][1][0]
                    circ.Rz(q, theta)
                        
                if data[j][0].name == 'sx':
                    q = data[j][1][0]
                
                    for k in range(nqubit):
                        if k == q:
                            circ.SX(k, p[k], T1[k], T2[q])
                        else:
                            circ.I(k) 

                if data[j][0].name == 'x':
                    q = data[j][1][0]
                
                    for k in range(nqubit):
                        if k == q:
                            circ.X(k, p[k], T1[k], T2[q])
                        else:
                            circ.I(k) 

                if data[j][0].name == 'cx':
                    q_ctr = data[j][1][0]
                    q_trg = data[j][1][1]
                
                    for k in range(nqubit):
                        if k == q_ctr:
                            circ.CNOT(k, q_trg, t_cnot[k][q_trg], p_cnot[k][q_trg], p[k], p[q_trg], T1[k], T2[k], T1[q_trg], T2[q_trg])
                        elif k == q_trg:
                            pass
                        else:
                            circ.I(k)          

                if data[j][0].name == 'delay':
                    q = data[j][1][0]
                    time = data[j][0].duration
                    dt = backend.configuration().dt
                    time = time * dt
                    
                    for k in range(nqubit):
                        if k == q:
                            circ.relaxation(k, time, T1[k], T2[k])
                        else:
                            circ.I(k)

            for k in range(nqubit):
                circ.bitflip(k, tm, rout[k])
                
            # Use statevector method to compute ensemble of final states
            psi = circ.statevector(psi0)
            
            for i in range(2**nqubit):
                r[m][i] = (absolute(psi[i]))**2

            circ.j = 0
            circ.s = 0
            circ.circuit = [[1 for i in range(depth)]for j in range(nqubit)]
            circ.phi = [0 for i in range(0, nqubit)]
         
        probs = array([mean(r[:, i]) for i in range (2**nqubit)])
        final_probs = self.fix_probs(probs, swap_detector, nqubit)
        return final_probs

    def fix_probs(self, wrong_probs: np.array, qubits_order: list, nqubit: int):
        """ This function fix the final probabilities in the right way (in case of swaps in the circuit)
        """

        wrong_counts = {format(i, 'b').zfill(nqubit): wrong_probs[i] for i in range(2**nqubit)}
        a = list(wrong_counts.keys())
        a2 = [0 for i in range(2**nqubit)]

        for k in range(2**nqubit):
            b = {i: j for i, j in zip(qubits_order, a[k])}
            c = sorted(b.items())
            x = ''
            for i in range(nqubit):
                x = x + c[i][1]
            a2[k] = x

        right_counts = {a2[k]:wrong_counts[a[k]] for k in range(2**nqubit)}
        d = sorted(right_counts.items())
        new_probs = [d[j][1] for j in range(2**nqubit)]
        return new_probs

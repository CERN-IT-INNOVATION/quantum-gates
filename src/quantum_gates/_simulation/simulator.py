"""Performs simulations with the Noisy quantum gates approach.
"""
import numpy as np
import copy
import typing

from qiskit import QuantumCircuit

from .._gates.gates import Gates, standard_gates
from .._simulation.circuit import EfficientCircuit, BinaryCircuit
from .._simulation.circuit import Circuit, StandardCircuit


class MrAndersonSimulator(object):
    """Simulates a noisy quantum circuit, extracting the gate instructions from a transpiled qiskit QuantumCircuit.

    Note that the shots can be parallelized, but this comes with a large overhead. Thus, we recommend to parallelize
    the usage of the simulator instead. This can be done with the function
    src.utility.simulation_utility.perform_parallel_simulation.

    Args:
        gates (Union[Gates, ScaledNoiseGates, NoiseFreeGates]): Gateset to be used, contains the pulse information.
        CircuitClass (Union[Circuit, EfficientCircuit]): Performs the computations with the backend.
        parallel (bool): Whether or not the shots should be run in parallel. False by default.

    Note:
        You must use the BinaryCircuit for non linear topologies, the other Circuit classes don't support non linear topologies.
        If you use the BinaryCircuit for non-linear topologies, make sure to import the parameters of all qubits up to the one with the maximum index, even if some qubits are not used, when importing device parameters.

    Example:
        .. code:: python

           from quantum_gates.simulators import MrAndersonSimulator
           from quantum_gates.gates import standard_gates
           from quantum_gates.circuits import EfficientCircuit

           sim = MrAndersonSimulator(
               gates==standard_gates,
               CircuitClass=EfficientCircuit,
               parallel=False
           )

           sim.run(t_qiskit_circ=...,
                   qubits_layout=...,
                   psi0=np.array([1.0, 0.0, 0.0, 0.0]),
                   shots=1000,
                   device_param=...,
                   nqubit=2)

    Attributes:
        gates (Union[Gates, ScaledNoiseGates, NoiseFreeGates]): Gateset to be used, contains the pulse information.
        CircuitClass (Union[Circuit, EfficientCircuit]): Performs the computations with the backend.
        parallel (bool): Whether or not the shots should be run in parallel. False by default.
    """

    def __init__(self, gates: Gates=standard_gates, CircuitClass=BinaryCircuit, parallel: bool=False):
        self.gates = gates  # Contains the information about the pulses.
        self.CircuitClass = CircuitClass
        self.parallel = parallel

    def run(self,
            t_qiskit_circ,
            qubits_layout: list,
            psi0: np.array,
            shots: int,
            device_param: dict,
            nqubit: int,
            level_opt:int = 4) -> dict:
        """
            Takes as input a transpiled qiskit circuit on a given backend with a given qubits layout
            and runs noisy quantum gates.
            Args:
                t_qiskit_circ: transpiled qiskit circuit (QuantumCircuit)
                qubits_layout: qubits layout with linear topology (list)
                psi0: initial state (array)
                shots: number of realizations (int)
                device_param: noise and device configurations as dict with the keys specified by DeviceParameters (dict)
                nqubit: number of qubits used in the circuit, must be compatible with psi0 (int)
                level_opt (int): Level of optimization for the circuit optimizator

            Returns:
                  dictionary of probabilities: the keys are the binary strings and the values the probabilities (dict)
        """

        # process layout circuit
        qubits_layout_t, qubit_bit, n_qubit_t = self._process_layout(t_qiskit_circ)

        n_measured_qubit = len(qubit_bit) # number of measured qubit
        if n_measured_qubit == 0:
            raise ValueError("None qubit measured")

        # Validate input
        self._validate_input_of_run(t_qiskit_circ, qubits_layout_t, psi0, shots, device_param, nqubit)

        # Count rz gates, construct swap lookup, generate data (representation of circuit compatible with simulation)
        n_rz, swap_detector, data = self._preprocess_circuit(t_qiskit_circ, qubits_layout_t, nqubit)

        # Read data and apply Noisy Quantum gates for many shots to get preliminary probabilities
        probs = self._perform_simulation(shots, data, n_rz, nqubit, device_param, psi0, qubits_layout_t, level_opt)

        # Reorder the probabilities to take the swaps into account
        #reordered_probs = self._fix_probabilities(probs, swap_detector, nqubit)

        # Normalize the result
        reordered_arr = np.array(probs)
        total_prob = np.sum(reordered_arr)
        assert total_prob > 0, f"Found unphysical probability vector {reordered_arr}."
        final_arr = reordered_arr / total_prob

        counts_ng = self._measurament(prob=final_arr, q_meas_list=qubit_bit, n_qubit=n_qubit_t, qubits_layout=qubits_layout_t)

        return counts_ng

    def _process_layout(self, circ : QuantumCircuit):
        """Take a (transpiled) circuit in input and get in output the list of used qubit and which qubit are measured and in which classical bits the
        information is stored

        Args:
            circ (QuantumCircuit): A quantum circuit, possibly transpiled

        Returns:
            used_q (list): List of real used qubit in this circuit
            measure_qc(list): List of tuples, each tuples contain the measured virtual qubit and the classical bit in which is stored the information
            n_qubit(int): number of used qubits in the circuit
        """
        used_q = []
        measure_qc = []

        for x in circ.data:
            if x[0].name != 'delay':
                if len(x[1]) == 1:
                    q = x[1][0]._index
                    if q not in used_q:
                        used_q.append(q)
                elif len(x[1]) == 2:
                    q1 = x[1][0]._index
                    q2 = x[1][1]._index
                    if q1 not in used_q:
                        used_q.append(q1)
                    if q2 not in used_q:
                        used_q.append(q2)
                if x[0].name == 'measure':
                    measure_qc.append((x[1][0]._index, x[2][0]._index))
        n_qubit = len(used_q)
        
        return used_q, measure_qc, n_qubit
    
    def _validate_input_of_run(self, t_qiskit_circ, qubits_layout, psi0, shots, device_param, nqubit):
        """ Performs sanity checks on the input of the run() method. Raises an Exception if any mistakes are found. """

        # Check types
        if not isinstance(t_qiskit_circ, QuantumCircuit):
            raise ValueError(f"Expected argument t_qiskit_circ to be of type QuantumCircuit, but found {type(t_qiskit_circ)}.")
        if not isinstance(qubits_layout, list):
            raise ValueError(f"Expected argument qubits_layout to be of type list, but found {type(qubits_layout)}.")
        if not isinstance(shots, int):
            raise ValueError(f"Expected argument shots to be of type int, but found {type(shots)}.")
        if not isinstance(device_param, dict):
            raise ValueError(f"Expected argument device_param to be of type dict, but found {type(device_param)}.")
        if not isinstance(nqubit, int):
            raise ValueError(f"Expected argument nqubit to be of type int, but found {type(nqubit)}.")

        # Check values
        if shots < 1:
            raise ValueError(f"Expected positive number of shots but found {shots}.")

        # Cross check number of qubits
        if psi0.shape != (2**nqubit,):
            raise ValueError(
                f"Expected shape of psi0 to be ({2**nqubit},) compatible with number of qubits ({nqubit}), but found " +
                f"shape {psi0.shape}."
            )
        if nqubit > len(qubits_layout):
            raise ValueError(
                f"Expected qubits layout to cover at least as many qubits as the transpiled circuit, but found "
              + f"{t_qiskit_circ.num_qubits} qubits in circuit and {len(qubits_layout)} qubits in layout."
            )
        if nqubit > len(device_param["T1"]):
            raise ValueError(
                f"Expected device parameters to cover at least as many qubits as the transpiled circuit, but found "
              + f"{t_qiskit_circ.num_qubits} qubits in circuit and {len(device_param['T1'])} qubits in device parameters."
            )

        return

    def _preprocess_circuit(self, t_qiskit_circ, qubits_layout: list, nqubit: int) -> tuple:
        """ Preprocess of QuantumCircuit.data. We count the number of RZ gates (n_rz), keep track of the swaps
        (swap_detector), and bring the circuit in a format (data) that is compatible with the rest of the simulation.
        """
        n_rz = 0
        data = []
        data_measure = []
        swap_detector = [a for a in range(nqubit)]
        raw_data = t_qiskit_circ.data

        for i in range(t_qiskit_circ.__len__()):
            if raw_data[i][0].name == 'ecr':
                q_ctr = raw_data[i][1][0]._index
                q_trg = raw_data[i][1][1]._index
                if q_ctr in qubits_layout and q_trg in qubits_layout:
                    raw_data[i][1][0] = qubits_layout.index(q_ctr)
                    raw_data[i][1][1] = qubits_layout.index(q_trg)  # TODO: Change such shared raw_data is not modified.
                    data.append(raw_data[i])

            elif raw_data[i][0].name == 'cx':
                q_ctr = raw_data[i][1][0]._index
                q_trg = raw_data[i][1][1]._index
                if q_ctr in qubits_layout and q_trg in qubits_layout:
                    raw_data[i][1][0] = qubits_layout.index(q_ctr)
                    raw_data[i][1][1] = qubits_layout.index(q_trg)  # TODO: Change such shared raw_data is not modified.
                    data.append(raw_data[i])

            elif raw_data[i][0].name == 'measure':
                q = raw_data[i][1][0]._index
                q = qubits_layout.index(q)
                c = raw_data[i][2][0]._index
                data_measure.append((q, c))

            else:
                q = raw_data[i][1][0]._index
                if q in qubits_layout:
                    if raw_data[i][0].name == 'rz':
                        n_rz = n_rz + 1
                    if raw_data[i][0].name != 'measure' and raw_data[i][0].name != 'barrier':
                        raw_data[i][1][0] = qubits_layout.index(q)
                        data.append(raw_data[i])

        for i in range(len(data_measure)):
            swap_detector[data_measure[i][0]] = data_measure[i][1]

        return n_rz, swap_detector, data

    def _perform_simulation(self,
                            shots: int,
                            data: list,
                            n_rz: int,
                            nqubit: int,
                            device_param: dict,
                            psi0: np.array,
                            qubit_layout: list,
                            level_opt:int) -> np.array:
        """ Performs the simulation shots many times and returns the resulting probability distribution.
        """

        # Setup results
        r_sum = np.zeros(2**nqubit)
        r_square_sum = np.zeros(2**nqubit)

        # Constants
        depth = len(data) - n_rz + 1

        # Create list of args
        arg_list = [
            {
                "data": copy.deepcopy(data),
                "circ": self.CircuitClass(nqubit, depth, copy.deepcopy(self.gates)),
                "device_param": copy.deepcopy(device_param),
                "psi0": copy.deepcopy(psi0),
                "qubit_layout": copy.deepcopy(qubit_layout),
                "level_opt": copy.deepcopy(level_opt)
            } for i in range(shots)
        ]

        # Perform computation parallel or sequentual
        if self.parallel:
            import multiprocessing

            # Configure pool
            cpu_count = multiprocessing.cpu_count()
            print(f"Our CPU count is {cpu_count}")

            n_processes = max(int(0.8 * cpu_count), 2)
            print(f"Use 80% of the cores, so {n_processes} processes.")

            chunksize = max(1, int(shots / n_processes) + (1 if shots % n_processes > 0 else 0))
            print(f"As we perform {shots} shots, we use a chunksize of {chunksize}.")

            # Compute
            p = multiprocessing.Pool(n_processes)
            for shot_result in p.imap_unordered(func=_single_shot, iterable=arg_list, chunksize=chunksize):
                # Add shot
                r_sum += shot_result
                r_square_sum += np.square(shot_result)

            # Shut down pool
            p.close()
            p.join()

        else:
            for arg in arg_list:
                # Compute
                shot_result = _single_shot(arg)

                # Add shot
                r_sum += shot_result
                r_square_sum += np.square(shot_result)

        # Calculate result
        r_mean = r_sum / shots
        r_var = r_square_sum / shots - np.square(r_mean)

        return r_mean

    def _fix_probabilities(self, wrong_probs: np.array, qubits_order: list, nqubit: int):
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
    
    def _measurament(self, prob : np.array, q_meas_list : list, n_qubit: int, qubits_layout: list) -> dict: 
        """This function take in input the measured qubits and the classical bits to store the information regarding also the swapping and give in ouput the probailities of the possible outcomes.

        Args:
            prob (np.array): probabilities after the application of all the gates 
            q_meas_list (list): list of tuples, where each tuples indicate the qubit measured and the corresponding classic bit 
            n_qubit (int): total number of qubits 
            qubit_layout(list): qubit layout after the transpilation 

        Returns:
            dict: the keys are the possible states and the value the probabilities of measurament each state.
        """

        # create the vector with the bit strings
        binary_vector = np.array([format(i, f'0{n_qubit}b') for i in np.arange(2**n_qubit)], dtype=str)

        qc_v = [] # list of tuples for virtual measured qubits and classic bits
        for t in q_meas_list:
            qc_v.append((qubits_layout.index(t[0]), t[1]))

        q_meas = [x[0] for x in qc_v] # virtual qubit

        # create a list of strings with the only string measured
        res = []

        for binary_str in binary_vector:
            temp = [binary_str[i] for i in q_meas]
            res.append(''.join(temp)) 

        
        # create a dictionary to store the probabilities of measure one of the possible state
        sums = {}

        for value, bit_string in zip(prob, res):
            if bit_string not in sums:
                sums[bit_string] = 0.0
            sums[bit_string] += value

        return sums


def _apply_gates_on_circuit(
        data: list,
        circ: Circuit or StandardCircuit or EfficientCircuit or BinaryCircuit, # type: ignore
        device_param: dict,
        qubit_layout:list):
    """ Applies the operations specified in data on the circuit.

    The constants regarding the device and noise are passed in device_param.

    Args:
        data (list): List of circuit instructions as preprocessed by the simulator.
        circ (Union[Circuit, StandardCircuit, EfficientCircuit]): Performs the computations.
        device_param (dict): Lookup for the noise information.
        qubit_layout(list): Layout of the used qubit
    """

    # Unpack dict
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
    nqubit = circ.nqubit

    if isinstance(circ, BinaryCircuit): # if the class of circuit is Binary class the application is different
        # Apply gates
        for j in range(len(data)):
            if data[j][0].name == 'rz':
                theta = float(data[j][0].params[0])
                q_r = data[j][1][0]._index #real qubit
                q_v = qubit_layout.index(q_r) #virtual qubit
                circ.Rz(q_v, theta)

            if data[j][0].name == 'sx':
                q_r = data[j][1][0]._index #real qubit
                q_v = qubit_layout.index(q_r) #virtual qubit
                for k in range(nqubit):
                    if k == q_v:
                        circ.SX(q_v, p[q_r], T1[q_r], T2[q_r])

            if data[j][0].name == 'x':
                q_r = data[j][1][0]._index #real qubit
                q_v = qubit_layout.index(q_r) #virtual qubit
                for k in range(nqubit):
                    if k == q_v:
                        circ.X(q_v, p[q_r], T1[q_r], T2[q_r])

            if data[j][0].name == 'ecr':
                q_ctr_r = data[j][1][0]._index # index control real qubit
                q_trg_r = data[j][1][1]._index # index target real qubit
                q_ctr_v = qubit_layout.index(q_ctr_r) # index control virtual qubit
                q_trg_v = qubit_layout.index(q_trg_r) # index control virtual qubit
                for k in range(nqubit):
                    if k == q_ctr_v:
                        circ.ECR(q_ctr_v, q_trg_v, t_int[q_ctr_r][q_trg_r], p_int[q_ctr_r][q_trg_r], p[q_ctr_r], p[q_trg_r], T1[q_ctr_r], T2[q_ctr_r], T1[q_trg_r], T2[q_trg_r])
                    elif k == q_trg_v:
                        pass

            if data[j][0].name == 'cx':
                q_ctr_r = data[j][1][0]._index # index control real qubit
                q_trg_r = data[j][1][1]._index # index target real qubit
                q_ctr_v = qubit_layout.index(q_ctr_r) # index control virtual qubit
                q_trg_v = qubit_layout.index(q_trg_r) # index control virtual qubit
                for k in range(nqubit):
                    if k == q_ctr_v:
                        circ.CNOT(q_ctr_v, q_trg_v, t_int[q_ctr_r][q_trg_r], p_int[q_ctr_r][q_trg_r], p[q_ctr_r], p[q_trg_r], T1[q_ctr_r], T2[q_ctr_r], T1[q_trg_r], T2[q_trg_r])
                    elif k == q_trg:
                        pass
            
            if data[j][0].name == 'delay':
                q_r = data[j][1][0]._index #real qubit
                q_v = qubit_layout.index(q_r) #virtual qubit
                time = data[j][0].duration * dt
                for k in range(nqubit):
                    if k == q_v:
                        circ.relaxation(q_v, time, T1[q_r], T2[q_r])

        for k in range(nqubit):
            q_r = qubit_layout[k]
            circ.bitflip(k, tm[q_r], rout[q_r])
        return
    else:
        # Apply gates
        for j in range(len(data)):

            if data[j][0].name == 'rz':
                theta = float(data[j][0].params[0])
                q = data[j][1][0]._index
                circ.Rz(q, theta)

            if data[j][0].name == 'sx':
                q = data[j][1][0]._index
                for k in range(nqubit):
                    if k == q:
                        circ.SX(k, p[k], T1[k], T2[q])
                    else:
                        circ.I(k)

            if data[j][0].name == 'x':
                q = data[j][1][0]._index
                for k in range(nqubit):
                    if k == q:
                        circ.X(k, p[k], T1[k], T2[q])
                    else:
                        circ.I(k)

            if data[j][0].name == 'ecr':
                q_ctr = data[j][1][0]._index # index control qubit
                q_trg = data[j][1][1]._index # index target qubit
                for k in range(nqubit):
                    if k == q_ctr:
                        circ.ECR(k, q_trg, t_int[k][q_trg], p_int[k][q_trg], p[k], p[q_trg], T1[k], T2[k], T1[q_trg], T2[q_trg])
                    elif k == q_trg:
                        pass
                    else:
                        circ.I(k)

            if data[j][0].name == 'cx':
                q_ctr = data[j][1][0]._index # index control qubit
                q_trg = data[j][1][1]._index # index target qubit
                for k in range(nqubit):
                    if k == q_ctr:
                        circ.CNOT(k, q_trg, t_int[k][q_trg], p_int[k][q_trg], p[k], p[q_trg], T1[k], T2[k], T1[q_trg], T2[q_trg])
                    elif k == q_trg:
                        pass
                    else:
                        circ.I(k)
            
            if data[j][0].name == 'delay':
                q = data[j][1][0]._index
                time = data[j][0].duration * dt
                for k in range(nqubit):
                    if k == q:
                        circ.relaxation(k, time, T1[k], T2[k])
                    else:
                        circ.I(k)

        for k in range(nqubit):
            circ.bitflip(k, tm[k], rout[k])
        return


def _single_shot(args: dict) -> np.array:
    """ Function used to simulate a single shot and return the corresponding result.

    Note:
        We have this method outside of the class such that it can be pickled.
    """

    circ = args["circ"]
    data = args["data"]
    device_param = args["device_param"]
    psi0 = args["psi0"]
    qubit_layout = args["qubit_layout"]
    level_opt = args["level_opt"]

    # Apply gates on the circuit.
    _apply_gates_on_circuit(data, circ, device_param, qubit_layout)

    # Propagate psi with  the state vector method
    if isinstance(circ, BinaryCircuit):
        psi = circ.statevector(psi0, level_opt, qubit_layout)
    else:
        psi = circ.statevector(psi0)

    # Calculate probabilities with the Born rule.
    shot_result = np.square(np.absolute(psi))

    return shot_result



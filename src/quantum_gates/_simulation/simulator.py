"""Performs simulations with the Noisy quantum gates approach.
"""
import numpy as np
import copy
import inspect
from typing import List, Tuple
from collections import Counter

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

           probs = sim.run(t_qiskit_circ=...,
                        qubits_layout=...,
                        psi0=np.array([1.0, 0.0, 0.0, 0.0]),
                        shots=1000,
                        device_param=...,
                        nqubit=2)
        
            print(probs)

        Expected output:

        .. code-block:: text

           {'00': 0.014743599038964704,
            '01': 0.000334214280332552,
            '10': 0.9612000084536643,
            '11': 0.02372217822703839}


    Attributes:
        gates (Union[Gates, ScaledNoiseGates, NoiseFreeGates]): Gateset to be used, contains the pulse information.
        CircuitClass (Union[Circuit, EfficientCircuit]): Performs the computations with the backend.
        parallel (bool): Whether or not the shots should be run in parallel. False by default.
    """

    def __init__(self, gates: Gates=standard_gates, CircuitClass=BinaryCircuit, parallel: bool=False):
        self.gates = gates() if inspect.isclass(gates) else gates  # Contains the information about the pulses.
        self.CircuitClass = CircuitClass
        self.parallel = parallel
        
    
    def run(self,
        t_qiskit_circ,
        qubits_layout: list,
        psi0: np.array,
        shots: int,
        device_param: dict,
        nqubit: int,
        bit_flip_bool=True,) -> dict:
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

        Returns:
            dictionary of probabilities: the keys are the binary strings and the values the probabilities (dict)
        
        Note: The output follow the Big Endian order for the bit strings
                
        """
        # Process layout circuit
        used_logicals, q_meas_list, n_qubit_used = self._process_layout(t_qiskit_circ)
        
        # Get total classical bits (for Aer-style output)
        num_clbits = len(t_qiskit_circ.clbits)
        
        # Infer width from psi0 (must be power of two)
        nqubit_state = int(round(np.log2(psi0.size)))
        if 2**nqubit_state != psi0.size:
            raise ValueError(f"psi0 length {psi0.size} is not a power of two.")

        # Choose simulation width = state width (ensures shape consistency)
        nqubit = nqubit_state

        # strong validation against the FULL layout (not the used subset)
        self._validate_input_of_run(t_qiskit_circ, qubits_layout, psi0, shots, device_param, nqubit)
        
        # optional: warn if there are idle qubits (kept in simulation; correct but may cost perf)
        if n_qubit_used < nqubit and hasattr(self, "_logger"):
            self._logger.warning(
                f"{nqubit - n_qubit_used} qubit(s) are idle (no operations). They will be simulated as idling qubits."
            )
        # build ACTIVE physical layout for only the used logicals
        #    e.g., qubits_layout = [3,0,2,5], used_logicals=[0,1]  => active_layout=[3,0]
        active_layout = [qubits_layout[q] for q in used_logicals]

        # Count rz gates, construct swap lookup, generate data (representation of circuit compatible with simulation)
        # preprocess circuit with the ACTIVE layout; nqubit is the full simulated width
        n_rz, swap_detector, data, data_measure = self._preprocess_circuit(
            t_qiskit_circ, active_layout, nqubit
        )

        # Read data and apply Noisy Quantum gates for many shots to get preliminary probabilities
        #  perform simulation (psi0 spans full width nqubit; active_layout routes gates)
        probs, all_results = self._perform_simulation(
            shots, data, n_rz, nqubit, device_param, psi0, active_layout, data_measure, bit_flip_bool
        )

        # Normalize final probabilities
        reordered_arr = np.asarray(probs, dtype=float)
        total_prob = reordered_arr.sum()
        if total_prob <= 0.0:
            raise ValueError(f"Unphysical probability vector: sum={total_prob}.")
        final_arr = reordered_arr / total_prob
        
        #    this keeps things correct whether 'probs' is over all qubits or only the used subset
        prob_width = int(round(np.log2(final_arr.size)))
        if 2**prob_width != final_arr.size:
            raise ValueError("Internal error: probability vector length is not a power of two.")

        # 10) produce final counts-style readout
        counts_ng = self._measurament(
            prob=final_arr,
            q_meas_list=q_meas_list,
            n_qubit=prob_width,       # if your simulator always returns full width, this equals nqubit
            qubits_layout=active_layout,
        )
        
        # --- FIXED: Build mid-circuit bitstrings with chronological processing ---
        combined_mid_strings = []
        mid_results = all_results

        for shot in mid_results:
            # initialize all clbits to '0' (Aer default for unused)
            clbit_values = ['0'] * num_clbits

            # FIX: Sort events by step in ascending order (chronological)
            sorted_events = sorted(shot["mid"], key=lambda x: x["step"])
            
            # Process events in chronological order (later measurements overwrite earlier ones)
            for event in sorted_events:
                for c, val in zip(event["clbits"], event["outcome"]):
                    clbit_values[c] = str(val)

            # sort descending by clbit index (Aer display order)
            bitstring = ''.join(clbit_values[::-1])
            combined_mid_strings.append(bitstring)

        # --- Count occurrences ---
        mid_counts = Counter(combined_mid_strings)

        return {
            "probs": counts_ng,
            "results": all_results, # mid-circuit measurement results
            "num_clbits": num_clbits, # number of classical bits in circuit
            "mid_counts": dict(mid_counts), # mid-circuit measurement counts
        }
    

    def _process_layout(self, circ : QuantumCircuit) -> Tuple[List[int], List[Tuple[int, int]], int]:
        """Take a (transpiled) circuit in input and get in output the list of used qubit and which qubit are measured and in which classical bits the
        information is stored

        Args:
            circ (QuantumCircuit): A quantum circuit, possibly transpiled

        Returns:
            used_q (list): List of real used qubit in this circuit
            measure_qc(list): List of tuples, each tuples contain the measured virtual qubit and the classical bit in which is stored the information
            n_qubit(int): number of used qubits in the circuit
        """
        used_logicals: list[int] = []
        measure_qc: list[tuple[int,int]] = []
        for instr in circ.data:
            op = instr.operation
            if op.name == 'delay':
                continue
            # support any arity
            for qb in instr.qubits:
                q = qb._index
                if q not in used_logicals:
                    used_logicals.append(q)
            if op.name == 'measure':
                measure_qc.append((instr.qubits[0]._index, instr.clbits[0]._index))
        return used_logicals, measure_qc, len(used_logicals)
    
    
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
        # psi0 must match the number of **used** qubits
        if psi0.shape != (2**nqubit,):
            raise ValueError(f"psi0 shape {psi0.shape} is incompatible with nqubit={nqubit} (expected {(2**nqubit,)}).")

        # layout must cover nqubit simulated qubits
        if nqubit > len(qubits_layout):
            raise ValueError(f"Layout too small: need {nqubit} entries, have {len(qubits_layout)}.")

        # device params must cover the largest physical index referenced by the first nqubit layout entries
        max_phys = max(qubits_layout[:nqubit]) if nqubit > 0 else -1
        if max_phys >= len(device_param["T1"]):
            raise ValueError(
                f"Device params do not cover physical qubit index {max_phys} "
                f"(have {len(device_param['T1'])} entries)."
            )

        return

        
    def _pretty_print_data(self, data):
        """Print human-readable view of preprocessed circuit data."""
        for idx, (chunk, flag) in enumerate(data):
            if flag == 0:
                ops_str = " , ".join(
                    f"{op.operation.name}[{', '.join(str(q._index) for q in op.qubits)}]"
                    for op in chunk
                )
                print(f"Chunk {idx}: {ops_str}")
            else:
                op = chunk
                # handle mid_measurement tuple
                if isinstance(op, tuple) and op[0] == "mid_measurement":
                    meas_op = op[1]
                    q_idx = meas_op["q_idx"]
                    c_idx = meas_op["c_idx"]
                    print(f"Fancy {idx}: mid_measurement qubits={q_idx} clbits={c_idx}")
                    
                elif isinstance(op, tuple) and op[0] == "reset_qubits":
                    meas_op = op[1]
                    q_idx = meas_op["q_idx"]
                    print(f"Fancy {idx}: reset_qubits qubits={q_idx}")
                    
                else:
                    # normal fancy gate (Instruction)
                    print(
                        f"Fancy {idx}: {op.operation.name} "
                        f"qubits={[q._index for q in op.qubits]} "
                        f"clbits={[c._index for c in op.clbits]}"
                    )

    
    def _preprocess_circuit(self, t_qiskit_circ, qubits_layout: list, nqubit: int,) -> tuple:
        """ Preprocess QuantumCircuit.data:
        - Count number of RZ gates (n_rz).
        - Track swaps (swap_detector).
        - Structure data into chunks split around 'fancy-gates':
        - bring the circuit in a format (data) that is compatible with the rest of the simulation.
        data = [ [ops until first fancy], fancy_gate, [ops until next fancy], fancy_gate, ... ]
        """
        # Initialize
        n_rz = 0
        data = []
        data_measure = []
        current_chunk = []
        swap_detector = [a for a in range(nqubit)]
        raw_data = t_qiskit_circ.data
        
        #  Build lookup: Clbit → register name
        # --- NEW: flat index maps (circuit-wide ordering) ---
        q2i = {q: i for i, q in enumerate(t_qiskit_circ.qubits)}     # Qubit  -> flat index
        c2i = {c: i for i, c in enumerate(t_qiskit_circ.clbits)}     # Clbit  -> flat index

        # Optional: map clbit -> (register_name, index_in_that_register)
        clbit_to_reg = {}
        for reg in t_qiskit_circ.cregs:
            for local_i, bit in enumerate(reg):   # <- get local index safely
                clbit_to_reg[bit] = (reg.name, local_i)

        # define which gates are considered "fancy"
        ## fancy_gates = {"reset_qubits","mid_measurement", "statevector_readout", "if_else"}

        for i, op in enumerate(raw_data):
            op_name = op.operation.name
            
            # ---------------------- MEASUREMENT ----------------------
            if op_name == "measure":
                # Is this a mid measurement? (any quantum op after it)
                remaining_ops = raw_data[i+1:]
                has_future_quantum = any(
                    (future_op.operation.name not in {"measure", "barrier"}) 
                    for future_op in remaining_ops
                )

                # Build flat index lists for ALL qubits/clbits in this instruction
                q_idx = [q2i[q] for q in op.qubits]      # flat qubit indices
                c_idx = [c2i[c] for c in op.clbits]      # flat clbit indices
                
                if has_future_quantum:
                    # mid-circuit → treat as fancy gate
                    if current_chunk:
                        data.append((current_chunk, 0))
                        current_chunk = []
                    # instead of mutating, just store a tuple with a tag
                    data.append((
                        ("mid_measurement", {
                            "op": op,       # keep original CircuitInstruction for debugging if you like
                            "q_idx": q_idx, # flat qubit indices (aligned with op.qubits)
                            "c_idx": c_idx, # flat clbit indices (aligned with op.clbits)
                        }),
                        1
                    ))

                else:
                    # final measurement → store for later
                    q = op.qubits[0]._index
                    q = qubits_layout.index(q)
                    c = op.clbits[0]

                    c_reg = clbit_to_reg.get(op.clbits[0], "unknown")
                    c_idx = op.clbits[0]._index
                    data_measure.append((q, (c_reg, c_idx)))

            # ---------------------- RESET ----------------------
            elif op_name == "reset":
                q_idx = [q2i[q] for q in op.qubits]      # flat qubit indices
                
                if current_chunk:
                    data.append((current_chunk, 0))
                    current_chunk = []

                # Expand multi-qubit reset into separate entries.
                # Use the PHYSICAL/flat index from Qiskit (._index) for consistency.
                data.append((("reset_qubits", {"op": op, "q_idx": q_idx}), 1))
            
            
            elif op_name == "statevector_readout":
                print("Warning: statevector_readout found in circuit, which is not implemented yet. Ignoring.")
                data.append((current_chunk,0))  # flush accumulated chunk before fancy gate
                current_chunk = []
                #TO DO: if q in qubits_layout:
                data.append((op,1))  # fancy gate goes as standalone
            elif op_name == "barrier":
                continue

            elif op_name == "rz":
                q = op.qubits[0]._index
                if q in qubits_layout:
                    n_rz += 1  # track rz count
                    current_chunk.append(op)  # probably you also want to keep it
            else:
                # normal operation (only if in layout)
                q = op.qubits[0]._index
                if q in qubits_layout:
                    current_chunk.append(op)

        # flush final chunk if non-empty
        if current_chunk:
            data.append((current_chunk,0))

        # update swap detector using measurements
        for q, c in data_measure:
            if q >= len(swap_detector):
                print(f"Skipping measurement update for qubit {q} (only {len(swap_detector)} qubits)")
                continue
            swap_detector[q] = c

        #print("Data after preprocessing:")
        #print(data)
            
        print("---- Preprocessed data ----")
        self._pretty_print_data(data)
        print("---------------------------")

        return n_rz, swap_detector, data, data_measure


    
    def _perform_simulation(self,
                            shots: int,
                            data: list,
                            n_rz: int,
                            nqubit: int,
                            device_param: dict,
                            psi0: np.array,
                            qubit_layout: list,
                            data_measure: list,
                            bit_flip_bool: bool,) -> np.array:
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
                "data_measure": copy.deepcopy(data_measure),   # <--- added
                "circ": self.CircuitClass(nqubit, depth, copy.deepcopy(self.gates)),
                "device_param": copy.deepcopy(device_param),
                "psi0": copy.deepcopy(psi0),
                "qubit_layout": copy.deepcopy(qubit_layout),
                "bit_flip_bool": bit_flip_bool,
            } for i in range(shots)
        ]
        
        all_results = []  # Store mid-circuit measurement results if needed

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
            for results, shot_result, final_outcomes in p.imap_unordered(func=_single_shot, iterable=arg_list, chunksize=chunksize):
                # Add shot
                r_sum += shot_result
                r_square_sum += np.square(shot_result)

                all_results.append({
                    "mid": results,
                    "final": final_outcomes
                })
            # Shut down pool
            p.close()
            p.join()

        else:
            for arg in arg_list:
                # Compute
                results, shot_result, final_outcomes = _single_shot(arg)

                r_sum += shot_result
                r_square_sum += np.square(shot_result)

                all_results.append({
                    "mid": results,
                    "final": final_outcomes
                })

        # Calculate result
        r_mean = r_sum / shots
        r_var = r_square_sum / shots - np.square(r_mean)
        
        ''' Debug print 
        print("---- Simulation shot results ----")
        for i, res in enumerate(all_results):
            print(f"Shot {i}: mid={res['mid']}, final={res['final']}")
        print("---------------------------------")
        '''
        return r_mean, all_results
    
    
    def _measurament(self, prob : np.array, q_meas_list : list, n_qubit: int, qubits_layout: list) -> dict: 
        """This function take in input the measured qubits and the classical bits to store the information regarding also the swapping and give in ouput the probabilities of the possible outcomes.

        Args:
            prob (np.array): probabilities after the application of all the gates 
            q_meas_list (list): list of tuples, where each tuples indicate the qubit measured and the corresponding classic bit 
            n_qubit (int): total number of qubits 
            qubit_layout(list): qubit layout after the transpilation 

        Returns:
            dict: the keys are the possible states and the value the probabilities of measurement each state.
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
        qubit_layout:list,
    ) -> None:
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
            if data[j].operation.name == 'rz':
                theta = float(data[j].operation.params[0])
                q_r = data[j].qubits[0]._index #real qubit
                q_v = qubit_layout.index(q_r) #virtual qubit
                circ.Rz(q_v, theta)

            if data[j].operation.name == 'sx':
                q_r = data[j].qubits[0]._index #real qubit
                q_v = qubit_layout.index(q_r) #virtual qubit
                circ.SX(q_v, p[q_r], T1[q_r], T2[q_r])
                        
            if data[j].operation.name == 'x':
                q_r = data[j].qubits[0]._index #real qubit
                q_v = qubit_layout.index(q_r) #virtual qubit
                circ.X(q_v, p[q_r], T1[q_r], T2[q_r])
                        
            if data[j].operation.name == 'ecr':
                q_ctr_r = data[j].qubits[0]._index # index control real qubit
                q_trg_r = data[j].qubits[1]._index # index target real qubit
                q_ctr_v = qubit_layout.index(q_ctr_r) # index control virtual qubit
                q_trg_v = qubit_layout.index(q_trg_r) # index control virtual qubit
                circ.ECR(q_ctr_v, q_trg_v, t_int[q_ctr_r][q_trg_r], p_int[q_ctr_r][q_trg_r], p[q_ctr_r], p[q_trg_r], T1[q_ctr_r], T2[q_ctr_r], T1[q_trg_r], T2[q_trg_r])

            if data[j].operation.name == 'cx':
                q_ctr_r = data[j].qubits[0]._index # index control real qubit
                q_trg_r = data[j].qubits[1]._index # index target real qubit
                q_ctr_v = qubit_layout.index(q_ctr_r) # index control virtual qubit
                q_trg_v = qubit_layout.index(q_trg_r) # index control virtual qubit
                circ.CNOT(q_ctr_v, q_trg_v, t_int[q_ctr_r][q_trg_r], p_int[q_ctr_r][q_trg_r], p[q_ctr_r], p[q_trg_r], T1[q_ctr_r], T2[q_ctr_r], T1[q_trg_r], T2[q_trg_r])
            
            if data[j].operation.name == 'delay':
                q_r = data[j].qubits[0]._index #real qubit
                q_v = qubit_layout.index(q_r) #virtual qubit
                time = data[j].operation.duration * dt
                circ.relaxation(q_v, time, T1[q_r], T2[q_r])              
        return
    
    else:
        # Apply gates
        for j in range(len(data)):

            if data[j].operation.name == 'rz':
                theta = float(data[j].operation.params[0])
                q = data[j].qubits[0]._index
                circ.Rz(q, theta)

            if data[j].operation.name == 'sx':
                q = data[j].qubits[0]._index
                for k in range(nqubit):
                    if k == q:
                        circ.SX(k, p[k], T1[k], T2[q])
                    else:
                        circ.I(k)

            if data[j].operation.name == 'x':
                q = data[j].qubits[0]._index
                for k in range(nqubit):
                    if k == q:
                        circ.X(k, p[k], T1[k], T2[q])
                    else:
                        circ.I(k)

            if data[j].operation.name == 'ecr':
                q_ctr = data[j].qubits[0]._index # index control qubit
                q_trg = data[j].qubits[1]._index # index target qubit
                for k in range(nqubit):
                    if k == q_ctr:
                        circ.ECR(k, q_trg, t_int[k][q_trg], p_int[k][q_trg], p[k], p[q_trg], T1[k], T2[k], T1[q_trg], T2[q_trg])
                    elif k == q_trg:
                        pass
                    else:
                        circ.I(k)

            if data[j].operation.name == 'cx':
                q_ctr = data[j].qubits[0]._index # index control qubit
                q_trg = data[j].qubits[1]._index # index target qubit
                for k in range(nqubit):
                    if k == q_ctr:
                        circ.CNOT(k, q_trg, t_int[k][q_trg], p_int[k][q_trg], p[k], p[q_trg], T1[k], T2[k], T1[q_trg], T2[q_trg])
                    elif k == q_trg:
                        pass
                    else:
                        circ.I(k)
            
            if data[j].operation.name == 'delay':
                q = data[j].qubits[0]._index
                time = data[j].operation.duration * dt
                for k in range(nqubit):
                    if k == q:
                        circ.relaxation(k, time, T1[k], T2[k])
                    else:
                        circ.I(k)

        return

def _expZ(psi: np.ndarray, q: int) -> float:
    """Return ⟨Z_q⟩ using BIG-endian: bit position = (n-1-q)."""
    n = int(np.log2(psi.size))
    pos = n - 1 - q
    p0 = 0.0
    for idx, amp in enumerate(psi):
        bit = (idx >> pos) & 1
        if bit == 0:
            p0 += (amp.real*amp.real + amp.imag*amp.imag)
    return 2.0*p0 - 1.0  # ⟨Z⟩ = P(0) - P(1) = 2P(0)-1


def _single_shot(args: dict) -> np.array:
    circ = args["circ"]
    data = args["data"]
    data_measure = args.get("data_measure", [])  # <--- added
    device_param = args["device_param"]
    psi0 = args["psi0"]
    qubit_layout = args["qubit_layout"]

    bit_flip_bool =  args["bit_flip_bool"]
    psi = psi0
    results = []  # mid results
    
    print("Layout mapping (phys→virt):", qubit_layout)
    phys_to_logical = {phys: log for log, phys in enumerate(qubit_layout)}

    for idx, (d, flag) in enumerate(data):
        if flag == 0:
            _apply_gates_on_circuit(d, circ, device_param, qubit_layout)
            psi = circ.statevector(psi)
            circ.reset_circuit()  # reset internal state for next chunk

        elif flag == 1:
            
            if isinstance(d, tuple) and d[0] == "mid_measurement":
                op = d[1]
                qubits  = op["q_idx"]
                clbits  = op["c_idx"]
                # Perform the mid-circuit measurement
                # TODO add_bitflip can be parameterized per measurement
                print("------ Mid-circuit measurement (physical) targets:", qubits, '-------------')
                print()
                psi, outcome = circ.mid_measurement(psi, device_param, add_bitflip=bit_flip_bool, qubit_list=qubits, cbit_list = clbits)
                
                outcome1 = [int(x) for x in np.atleast_1d(outcome)]
                assert len(outcome1) == len(clbits), "Outcome/clbits length mismatch"
                
                # Normalize again (defensive) TODO: is this needed?
                norm = np.linalg.norm(psi)
                if norm > 0:
                    psi /= norm

                print(f"-------------------Mid-measurement END-----------")
                # record debug info
                results.append({
                    "step": idx,
                    "qubits": qubits,
                    "clbits": clbits,
                    "outcome": outcome,
                })

            elif isinstance(d, tuple) and d[0] == "reset_qubits":
                op = d[1]
                qubits = op["q_idx"]   # already flat (transpiled indices)

                print("\n========== RESET BEGIN ==========")
                print(f"Reset called on qubits (transpiled indices): {qubits}")

                # --- 1) Collapse without classical writes ---
                psi, outcomes = circ.mid_measurement(
                    psi0=psi,
                    device_param=device_param,
                    add_bitflip=bit_flip_bool,
                    qubit_list=qubits,
                    cbit_list=None,
                )

                print(f"Reset measurement outcomes: {outcomes}")

                # Extract noise params *once*
                T1, T2, p = device_param["T1"], device_param["T2"], device_param["p"]

                # --- 2) Apply X/I layer ---
                # 2) BEFORE-correction sanity (what is ⟨Z⟩ after collapse?)
                for q in qubits:
                    print(f"  ⟨Z_{q}⟩ before correction: { _expZ(psi, q): .3f}")
                print("Applying reset correction layer:")
                touched = set()
                for q, res in zip(qubits, outcomes):
                    touched.add(q)
                    if res == 1:
                        print(f"  • Applying X on qubit {q}")
                        circ.X(i=q, p=p[q], T1=T1[q], T2=T2[q])
                    else:
                        print(f"  • Applying I on qubit {q} (already |0⟩)")
                        circ.I(i=q)

                for k in range(circ.nqubit):
                    if k not in touched:
                        circ.I(i=k)   # maintain full layer (no gaps)

                # --- 3) Apply layer to state and reset builder ---
                psi_before = psi.copy()
                print("Statevector before reset correction layer:")
                print(psi_before)
                psi = circ.statevector(psi)
                circ.reset_circuit()
                print("Statevector after reset correction layer:")
                print(psi)

                # --- 4) Defensive normalize ---
                norm = np.linalg.norm(psi)
                if norm > 0:
                    psi /= norm

                # 5) AFTER-correction: ⟨Z⟩ should be ~+1 for each target
                print("Post-reset expectation ⟨Z⟩ check:")
                for q in qubits:
                    ez = _expZ(psi, q)
                    print(f"  ⟨Z_{q}⟩ ≈ {ez: .3f}  (≈ +1 means reset successful)")
                print("=========== RESET END ===========\n")

            else:
                op_name = d.operation.name
                if op_name == "statevector_readout":
                    print("Statevector readout not tested!!")
                    print("Statevector readout:", psi.copy())

                elif op_name == "if_else":
                    raise NotImplementedError("if_else not implemented yet")

                else:
                    raise ValueError(f"Unknown fancy gate: {op_name}")

    # --- Final Measurements ---
    # Born rule → probability distribution
    shot_result = np.square(np.abs(psi))

    # Final outcomes from last measurement
    final_outcomes = None
    if data_measure:
        probs = np.square(np.abs(psi))
        if probs.sum() == 0:
            raise ValueError("Statevector collapsed to zero norm after circuit execution.")
        probs = probs / probs.sum()   # normalize before sampling

        outcome_index = np.random.choice(len(probs), p=probs)

        bitstring = format(outcome_index, f"0{circ.nqubit}b")
        final_outcomes = {}
        for q, (c_reg, c_idx) in data_measure:
            final_outcomes[(c_reg, c_idx)] = int(bitstring[-(q+1)])

    return results[::-1], shot_result, final_outcomes

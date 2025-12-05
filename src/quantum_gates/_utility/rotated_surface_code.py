import math
from qiskit import QuantumCircuit, transpile

from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import standard_gates, noise_free_gates
from quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from quantum_gates.utilities import DeviceParameters
from quantum_gates.utilities import setup_backend

import random
import numpy as np
import matplotlib.pyplot as plt
import pymatching
from qiskit.transpiler import InstructionProperties  
from qiskit.circuit.library import XGate, YGate, ZGate
import pymatching
from scipy.sparse import csr_matrix, lil_matrix
from qiskit.circuit.controlflow import ControlFlowOp 

from qiskit.transpiler import Target, CouplingMap, InstructionProperties
from qiskit.circuit import Instruction
#from qiskit.providers.models import BackendConfiguration
#from qiskit.providers import BackendProperties
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
#from qiskit.providers.fake_provider import FakeBackendV2
from copy import deepcopy
from scipy.linalg import sqrtm


class RotatedSurfaceCode:
    def __init__(self, distance=3, cycles = 1):
        self.d = distance
        self.n_rows = 2 * distance + 1
        self.cycles = cycles
        self.shots = 1
        self.n_data = distance**2 
        self.n_stabilizers = distance**2 - 1 
        self.n_clbits = self.n_stabilizers * cycles
        #self.width = 
        self.qc = QuantumCircuit(self.n_data + self.n_stabilizers, self.n_clbits)

        # define data and stabilizer indices based on even/odd parity

        self.stabilizers = [i for i in range((self.n_data + self.n_stabilizers)) if i % 2 == 1]
        data_qubits, x_stabilizers, z_stabilizers,  neighbors = self._get_surface_code_layout()
        self.data = data_qubits
        self.x_stabilizers = x_stabilizers
        self.z_stabilizers = z_stabilizers
        self.neighbors = neighbors

        # maps stabilizer index -> list of classical bit indices (one per cycle)
        self.stabilizer_to_clbits = {
            stab: [cycle * self.n_stabilizers + i for cycle in range(self.cycles)]
            for i, stab in enumerate(self.stabilizers)
        }

        self._build_stabilizer_layer()
        # Compute stabilizer connections BEFORE setting up decoder
        self.stabilizer_connections = self._compute_stabilizer_connections()
    
        # Now setup the decoder (needs x_stabilizers, z_stabilizers, and stabilizer_connections)
        self.setup_decoder()
        
    def _get_surface_code_layout(self):
        """
        Construct the rotated surface code layout following user rules:

        Row structure:
        - Row 0: floor(d/2) Z stabilizers, then d Z stabilizers, then floor(d/2) Z stabilizers
        - Rows 1,3,... : d data qubits centered
        - Rows 2,4,... : d alternating X/Z stabilizers centered, starting with X
        - Last row (2d-2): same as row 0
        """

        d = self.d
        n_rows = self.n_rows
        half = d // 2

        data = []
        stab_x = []
        stab_z = []
        neighbors = {} 

        index = 0  # simple linear indexing

        for r in range(n_rows):

            # ---- Row 0 (top) and last row: Z boundary stabilizers ----
            if r == 0 or r == n_rows-1:
                if r == 0: factor = +1 * math.ceil(d/2)
                else: factor = -1 * d
                index_neighbor = 0
                for k in range(half):
                    stab_z.append((r, index)); 
                    neighbor = []
                    neighbor.append(index + factor  + index_neighbor )
                    neighbor.append(index + factor + 1  + index_neighbor )
                    neighbors[index] = neighbor
                    index_neighbor += 1
                    index += 1
                
                continue

            # ---- Data row (odd rows) ----
            if r % 2 == 1:
                for _ in range(d):
                    data.append((r, index))
                    index += 1
                continue

            # ---- Stabilizer row (even rows except boundary) ----
            if r % 2 == 0:
                # Alternating X/Z, starting with X
                for k in range(d):
                    if k % 2 == 0 :
                        stab_x.append((r, index))
                        if r %4 == 2:
                            if k == 0:
                                neighbor = []
                                neighbor.append(index - d)
                                neighbor.append(index + d)
                                neighbors[index] = neighbor
                            else:
                                neighbor = []
                                neighbor.append(index - d -1)
                                neighbor.append(index - d )
                                neighbor.append(index + d -1)
                                neighbor.append(index + d )
                                neighbors[index] = neighbor

                        else: 
                            if k == d -1:
                                neighbor = []
                                neighbor.append(index - d)
                                neighbor.append(index + d)
                                neighbors[index] = neighbor
                            else:
                                neighbor = []
                                neighbor.append(index - d )
                                neighbor.append(index - d +1)
                                neighbor.append(index + d )
                                neighbor.append(index + d +1)
                                neighbors[index] = neighbor
                    else:
                        stab_z.append((r, index))
                        if r % 4 == 2:
                            neighbor = []
                            neighbor.append(index - d -1)
                            neighbor.append(index - d )
                            neighbor.append(index + d -1)
                            neighbor.append(index + d )
                            neighbors[index] = neighbor
                        else: 
                            neighbor = []
                            neighbor.append(index - d)
                            neighbor.append(index - d +1)
                            neighbor.append(index + d)
                            neighbor.append(index + d +1)
                            neighbors[index] = neighbor
                    index += 1
                continue



        return data, stab_x, stab_z, neighbors


    
    

    def _build_stabilizer_layer(self):
        """Build one stabilizer measurement cycle for a distance-n surface code."""

        qubits = self.d
        cycles = self.cycles

        # repeat the stabilizer-measurement process for all cycles
        for cycle in range(cycles):
            # --- Reset all stabilizers at the beginning of the cycle ---
            for stabilizer in self.x_stabilizers + self.z_stabilizers:
                anc = stabilizer[1]
                self.qc.reset(anc)
                
            # --- X stabilizers ---
            for stabilizer in self.x_stabilizers:
                anc = stabilizer[1]
                self.qc.h(anc)  # prepare X stabilizer in |+>
                #self.qc.barrier()

                # entangle with data qubits (ancilla is control)
                neighbor = self.neighbors[anc]
                for nb in neighbor:
                    self.qc.cx(anc, nb)

                #self.qc.barrier()
                self.qc.h(anc)  # rotate back before measurement

            # --- Z stabilizers ---
            for stabilizer in self.z_stabilizers:
                anc = stabilizer[1]
                # entangle with data qubits (data is control, ancilla target)
                neighbor = self.neighbors[anc]
                for nb in neighbor:
                    self.qc.cx(nb, anc)

                #self.qc.barrier()
            
            #conditional gate on the mid-circuit measurement results to run the decoder and determine the basis for the end measrement.

            # --- Measure all stabilizers for this cycle --- 
            classical_offset = cycle * self.n_stabilizers
            stab_list = sorted(self.x_stabilizers + self.z_stabilizers)
            for i, stabilizer in enumerate(stab_list):
                anc = stabilizer[1]
                self.qc.measure(anc, classical_offset + i)

            self.qc.save_statevector(label=f"save_sv_{cycle}")
            self.qc.barrier(label=f"save_sv_{cycle}")


    def run_surfacecode(self, noise):
        """Run the surface code circuit on the specified backend with noise model."""
        #backend = self._create_backend()
        backend = FakeBrisbane()
        if noise:
            set_gate = standard_gates
            bit_flip_bool = True
        else:
            set_gate = noise_free_gates
            bit_flip_bool = False

        sim = MrAndersonSimulator(gates=set_gate, CircuitClass=EfficientCircuit)
        N_q = self.n_data + self.n_stabilizers 
        

        needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in self.qc.data)

        t_circ = transpile(
            self.qc,
            backend,
            initial_layout=list(range(N_q)),
            seed_transpiler=42,
            scheduling_method=needs_controlflow,           # no scheduling

        )

        # Check which qubits are actually used in transpiled circuit
        used_qubits: list[int] = []
        for instr in t_circ.data:
            op = instr.operation
            if op.name == 'delay':
                continue
            # support any arity
            for qb in instr.qubits:
                q = qb._index
                if q not in used_qubits:
                    used_qubits.append(q)
                    
        print(f"Qubits used in transpiled circuit: {sorted(used_qubits)}")

        max_qubit = max(used_qubits)
        nqubit_actual = max_qubit + 1

        initial_psi = np.zeros(2**nqubit_actual)

        initial_psi[0] = 1.0  # set |00...0⟩

        qubits_layout = list(range(nqubit_actual))

        #  Load via YOUR class and save JSON next to the script
        device_param = DeviceParameters(qubits_layout)
        device_param.load_from_backend(backend)
        device_param_lookup = device_param.__dict__()

        res  = sim.run( 
            t_qiskit_circ=t_circ, 
            qubits_layout=qubits_layout, 
            psi0=initial_psi, 
            shots=self.shots, 
            device_param=device_param_lookup,
            nqubit=nqubit_actual,
            bit_flip_bool=bit_flip_bool,
            )

        probs = res["probs"]
        results = res["results"]
        num_clbits = res["num_clbits"]
        mid_counts = res["mid_counts"]
        statevector_readout = res["statevector_readout"]
        """
        #Calucate Fidelity per cycle

        # extract only the final statevector of each cycle
        self._fidelity_per_cycle(statevector_readout, mid_counts)

        decoder_x, decoder_z = self.decode_full(mid_counts)
        print("Decoder output X:", decoder_x)
        print("Decoder output Z:", decoder_z)
        print("Measurement results:", mid_counts)
        # XOR over cycles → net correction
        
        for cycle in range(1, self.cycles):
            U_map = self._create_U_map(decoder_x[cycle], decoder_z[cycle])
            print(f"U map after cycle {cycle}:", U_map)
            #apply U to the statevector of the dataqubits only and save as statevctor list where statevector[c] 
            #is the error corrected statevector after cycle c 

        # now compute fidelity of statevector[c-1] and statevector[c] for each cycle c and plot it 


        final_x_correction = np.bitwise_xor.reduce(decoder_x, axis=0)
        final_z_correction = np.bitwise_xor.reduce(decoder_z, axis=0)

        print("Final X correction on data qubits:", final_x_correction)
        print("Final Z correction on data qubits:", final_z_correction)

        U_map = self._create_U_map(final_x_correction, final_z_correction)
        print("U map (data qubit corrections):", U_map)

        #return statevector on dataqubits
        """
        return mid_counts
    
    def extract_statevectors_per_cycle(self, statevector_readout):
        """
        Extracts only the final statevector of each QEC cycle.

        Parameters
        ----------
        statevector_readout : list
            A flat list of statevectors, one per stabilizer measurement.
        n_stabilizers : int
            Number of stabilizer measurements per cycle.

        Returns
        -------
        list
            A list containing only the final statevector of each cycle.
        """
        n_stabilizers = len(self.stabilizers)
        sv_per_cycle = [
            statevector_readout[i]
            for i in range(n_stabilizers - 1, len(statevector_readout), n_stabilizers)
        ]

        if len(sv_per_cycle) != self.cycles:
            raise ValueError(
                f"Expected {self.cycles} cycles, but got {len(sv_per_cycle)} "
                f"final statevectors. Readout length={len(statevector_readout)}, "
                f"n_stabilizers={n_stabilizers}."
            )

        return sv_per_cycle


    def _reduced_density_matrix(self, statevector):
        """
        Computes the reduced density matrix on the data qubits by
        tracing out all stabilizer qubits.

        Parameters
        ----------
        statevector : np.ndarray
            Full statevector of size 2**total_qubits.
        data_qubits : list[int]
            Indices (0 = least significant) of qubits to keep.
        total_qubits : int
            Total number of qubits.

        Returns
        -------
        rho_data : np.ndarray
            2**|data_qubits| × 2**|data_qubits| reduced density matrix.
        """



        total_qubits = self.n_data + self.n_stabilizers

        psi = statevector.reshape([2] * total_qubits)

        # --- Helper to get the global qubit index ---
        def _extract_pos(q):
            if isinstance(q, tuple):     # (row, pos)
                return q[1]
            else:                        # int (old stabilizers)
                return q

        # Extract only flat qubit positions
        data_qubits = sorted(_extract_pos(q) for q in self.data)
        stab_qubits = [_extract_pos(q) for q in self.stabilizers]

        perm = data_qubits + stab_qubits

        psi_perm = np.transpose(psi, axes=perm)

        nd = len(data_qubits)
        ns = len(stab_qubits)

        psi_matrix = psi_perm.reshape(2**nd, 2**ns)

        rho_full = psi_matrix @ psi_matrix.conj().T
        return rho_full




    def _fidelity_per_cycle(self, statevector_readout, bitstring):
        """Compute and plot fidelity of data qubits' statevector after each cycle."""
        if isinstance(bitstring, dict):
            bitstring = list(bitstring.keys())[0]

        statevector_per_cycle = self.extract_statevectors_per_cycle(statevector_readout)

        fidelities = []
        n_data = len(self.data)
        
        
        sv_last= statevector_per_cycle[0]
        rho_full_last= self._reduced_density_matrix(sv_last)
        rho_full_last_corrected = rho_full_last
        x_syndromes, z_syndromes = self._extract_stabilizer_measurements(bitstring)
        x_last = x_syndromes[0]
        z_last = z_syndromes[0]
        

        for cycle in range(1, self.cycles):
            sv = statevector_per_cycle[cycle] 
            # Extract data qubits' statevector
            rho_full = self._reduced_density_matrix(sv) 
            x = x_syndromes[cycle] 
            z = z_syndromes[cycle] 

            prediction_x_cycle = self._decode_per_cycle(x_last, x, which="X")
            prediction_z_cycle = self._decode_per_cycle(z_last, z, which="Z")

            print(f"Cycle {cycle + 1}:")
            print("X syndrome:", x_last, "->", x, "Prediction:", prediction_x_cycle)
            print("Z syndrome:", z_last, "->", z, "Prediction:", prediction_z_cycle)
            rho_copy = rho_full
            # Apply corrections to rho_full_next
            for i, data_qubit in enumerate(self.data):
                if prediction_x_cycle[i] == 1:
                    # Apply X correction
                    X = np.array([[0, 1], [1, 0]])
                    rho_copy = np.kron(np.eye(2**i), np.kron(X, np.eye(2**(n_data - i - 1)))) @ rho_copy @ np.kron(np.eye(2**i), np.kron(X, np.eye(2**(n_data - i - 1))))
                if prediction_z_cycle[i] == 1:
                    # Apply Z correction
                    Z = np.array([[1, 0], [0, -1]])
                    rho_copy = np.kron(np.eye(2**i), np.kron(Z, np.eye(2**(n_data - i - 1)))) @ rho_copy @ np.kron(np.eye(2**i), np.kron(Z, np.eye(2**(n_data - i - 1))))
            # Compute fidelity
            # Compute Uhlmann fidelity with logical |0_L>
            A = sqrtm(rho_full_last_corrected)
            F = np.real(np.trace(sqrtm(A @ rho_copy @ A)))**2
            fidelities.append(F)


            sv_last = sv
            rho_full_last = rho_full
            rho_full_last_corrected = rho_copy
            x_last = x
            z_last = z

        # Plot fidelities
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, self.cycles), fidelities, marker='o')
        plt.title("Fidelity of Data Qubits' Statevector per Cycle")
        plt.xlabel("Cycle")
        plt.ylabel("Fidelity with the previous cycle")
        plt.ylim(0, 1)
        plt.grid()
        plt.show()   


    
    def _create_U_map(self, final_x_correction, final_z_correction):
        """
        Create a mapping of data qubits to their final corrections.
        Returns a dict: data_qubit_index -> (X_correction, Z_correction)
        """
        if len(final_x_correction) != len(final_z_correction):
            raise ValueError("Length of X and Z correction arrays must match.")
        U_map = []
        for i in range(len(final_x_correction)):
            if final_x_correction[i] == 0 and final_z_correction[i] == 0:
                U_map.append("pauli_I")
            elif final_x_correction[i] == 1 and final_z_correction[i] == 0:
                U_map.append("pauli_X")
            elif final_x_correction[i] == 0 and final_z_correction[i] == 1:
                U_map.append("pauli_Z")
            elif final_x_correction[i] == 1 and final_z_correction[i] == 1:
                U_map.append("pauli_Y")
            
        return U_map

    def _extract_stabilizer_measurements(self, bitstring):
        """
        Extract measurement results for all stabilizers across all cycles.
        Returns separate arrays for X and Z stabilizers.
        """
        bits = bitstring[::-1]  # reverse Qiskit order
        
        x_syndromes = np.zeros((self.cycles, len(self.x_stabilizers)), dtype=int)
        z_syndromes = np.zeros((self.cycles, len(self.z_stabilizers)), dtype=int)
        
        for i, stab in enumerate(self.x_stabilizers):
            clbits = self.stabilizer_to_clbits[stab]
            x_syndromes[:, i] = [int(bits[cb]) for cb in clbits]
        
        for i, stab in enumerate(self.z_stabilizers):
            clbits = self.stabilizer_to_clbits[stab]
            z_syndromes[:, i] = [int(bits[cb]) for cb in clbits]
        
        return x_syndromes, z_syndromes
   
    def analyze_results(self, counts):
        """
        Analyze measurement results using stabilizer-to-classical-bit mapping.
        Returns a dict: stabilizer → {bitstring_pattern: frequency}.
        """
        results = {stab: {} for stab in (self.x_stabilizers + self.z_stabilizers)}

        for bitstring, count in counts.items():
            # Use the helper function
            x_syndromes, z_syndromes = self._extract_stabilizer_measurements(bitstring)
            
            # Process X stabilizers
            for i, stab in enumerate(self.x_stabilizers):
                # Get measurements for this stabilizer across all cycles
                bits_for_stab = tuple(x_syndromes[:, i])
                results[stab][bits_for_stab] = results[stab].get(bits_for_stab, 0) + count
            
            # Process Z stabilizers
            for i, stab in enumerate(self.z_stabilizers):
                bits_for_stab = tuple(z_syndromes[:, i])
                results[stab][bits_for_stab] = results[stab].get(bits_for_stab, 0) + count

        # pretty string output
        pretty = {
            stab: {','.join(map(str, k)): v for k, v in res.items()} 
            for stab, res in results.items()
        }

        return pretty

    def _plot_single_shot(self, bitstring, shot_idx=None):
        """Plot single-shot stabilizer measurements timeline."""
        plt.figure(figsize=(10, 3))
        
        # Use the helper function
        x_syndromes, z_syndromes = self._extract_stabilizer_measurements(bitstring)
        
        # Plot X stabilizers
        for i, stab in enumerate(self.x_stabilizers):
            bits = x_syndromes[:, i]  # Get column i (all cycles for this stabilizer)
            plt.step(
                range(1, len(bits) + 1),
                bits,
                where='mid',
                label=f'X stabilizer {stab}',
                marker='o',
                linewidth=2,
                markersize=6,
                alpha=0.7,
                linestyle='-'
            )

        # Plot Z stabilizers
        for i, stab in enumerate(self.z_stabilizers):
            bits = z_syndromes[:, i]  # Get column i (all cycles for this stabilizer)
            plt.step(
                range(1, len(bits) + 1),
                bits,
                where='mid',
                label=f'Z stabilizer {stab}',
                marker='s',
                linewidth=2,
                markersize=6,
                alpha=0.7,
                linestyle='--'
            )

        title = "Single-shot stabilizer measurements"
        if shot_idx is not None:
            title += f" — Shot {shot_idx + 1}"
        plt.title(title, fontsize=14)
        plt.xlabel("Cycle", fontsize=12)
        plt.ylabel("Measurement", fontsize=12)
        plt.yticks([0, 1])
        plt.xticks(range(1, self.cycles + 1))
        plt.grid(alpha=0.4)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    def plot_results(self, counts, plot_each_shot):
        """
        Plot measurement results for the surface code.
        - If 1 shot: timeline plot of stabilizer outcomes per cycle.
        - If >1 shots:
            - If plot_each_shot=True: plot each shot individually like single-shot mode.
            - If plot_each_shot=False: plot aggregated histogram (default behavior).
        """
        total_shots = self.shots
        # --- Case 1: only one shot ---
        if total_shots == 1:
            bitstring = list(counts.keys())[0][::-1]
            self._plot_single_shot(bitstring)
            return

        # --- Case 2: multiple shots ---
        if plot_each_shot:
            all_bitstrings = list(counts.keys())
            for i in range(total_shots):
                bitstring = all_bitstrings[i][::-1]   # get i-th bitstring, reversed
                self._plot_single_shot(bitstring, shot_idx=i)
        else:
            analyzed = self.analyze_results(counts)
            flat_results = {
                f"stab{stab}:{key}": val
                for stab, patterns in analyzed.items()
                for key, val in patterns.items()
            }

            labels = list(flat_results.keys())
            values = list(flat_results.values())

            plt.figure(figsize=(10, 3))
            plt.bar(range(len(labels)), values, color='steelblue', edgecolor='black')

            plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
            plt.xlabel("Stabilizer syndromes")
            plt.ylabel("Counts")
            plt.title(f"{total_shots} shots - Stabilizer syndromes")

            plt.tight_layout()
            plt.show()

    
    def _compute_stabilizer_connections(self):
        """
        Compute which data qubits each stabilizer measures.
        Returns dict: stabilizer_index -> list of data qubit indices
        """
        connections = {}
        
        for stab in self.x_stabilizers + self.z_stabilizers:
            anc = stab[1]
                # entangle with data qubits (data is control, ancilla target)
            neighbors = self.neighbors[anc]
            # Filter to only include data qubits
            data_neighbors = [nb for nb in neighbors if nb in self.data]
            connections[stab] = data_neighbors
            
            # Debug print
            stab_type = 'X' if stab in self.x_stabilizers else 'Z'
            print(f"Stabilizer {stab} ({stab_type}) measures data qubits: {data_neighbors}")
        return connections

# ========================================================================
#  DECODER METHODS
#  ========================================================================

    def _build_parity_check_matrix(self, stabilizer_type):
        """
        Build the parity-check matrix for the specified stabilizer type ('X' or 'Z').
        Returns a scipy sparse matrix in CSR format.
        """
        if stabilizer_type == 'X':
            stabs = self.x_stabilizers
        elif stabilizer_type == 'Z':
            stabs = self.z_stabilizers
        else:
            raise ValueError("stabilizer_type must be 'X' or 'Z'")

        n_stabs = len(stabs)
        n_data = len(self.data)

        # Create a sparse matrix in LIL format for easy construction
        H = lil_matrix((n_stabs, n_data), dtype=int)

        for i, stab in enumerate(stabs):
            data_qubits = self.stabilizer_connections[stab]
            for dq in data_qubits:
                data_idx = self.data.index(dq)
                H[i, data_idx] = 1
        
        print(f"Parity-check matrix for {stabilizer_type} stabilizers built.")
        print(H.toarray())
        return H.tocsr()
    
    def setup_decoder(self):
        """
        Setup the decoder by building parity-check matrices for X and Z stabilizers.
        """
        self.H_X = self._build_parity_check_matrix('X')
        self.H_Z = self._build_parity_check_matrix('Z')
        print("Decoder setup complete.")

 

    def _decode_per_cycle(self, syndrome1, syndrome2, which='X'):
        # Bitwise XOR between consecutive cycles
        syndrome = np.bitwise_xor(syndrome1, syndrome2).astype(np.uint8).flatten()

        # Choose which stabilizer matrix to use
        if which == 'X':
            H = self.H_X
        elif which == 'Z':
            H = self.H_Z
        else:
            raise ValueError("which must be 'X' or 'Z'")

        matching = pymatching.Matching(H)
    
        # Sanity check
        assert syndrome.shape[0] == H.shape[0], \
            f"Syndrome length {syndrome.shape[0]} != {H.shape[0]} stabilizers"

        prediction = matching.decode(syndrome)
        return prediction
    
    def decode_full(self, bitstring):
        """
        Decode the full syndrome history across all cycles.
        The bitstring contains concatenated stabilizer measurements
        """
        if isinstance(bitstring, dict):
            bitstring = list(bitstring.keys())[0]

        x_syndromes, z_syndromes = self._extract_stabilizer_measurements(bitstring)
        print("Extracted X syndromes:\n", x_syndromes)
        print("Extracted Z syndromes:\n", z_syndromes)
        n_cycles = self.cycles

        prediction_x = []
        prediction_z = []

        for c in range(n_cycles - 1):
            # decode between consecutive cycles
            x1, x2 = x_syndromes[c], x_syndromes[c + 1]
            z1, z2 = z_syndromes[c], z_syndromes[c + 1]

            prediction_x_cycle = self._decode_per_cycle(x1, x2, which="X")
            prediction_z_cycle = self._decode_per_cycle(z1, z2, which="Z")

            prediction_x.append(prediction_x_cycle)
            prediction_z.append(prediction_z_cycle)

        return np.array(prediction_x), np.array(prediction_z)
        
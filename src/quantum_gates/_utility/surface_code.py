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



class SurfaceCode:
    def __init__(self, distance=3, cycles = 1):
        self.d = distance
        self.cycles = cycles
        self.shots = 1
        self.n_data = distance**2 + (distance - 1)**2
        self.n_stabilizers = 2*(distance-1)* distance
        self.n_clbits = self.n_stabilizers * cycles
        self.width = 2*distance-1
        self.qc = QuantumCircuit(self.n_data + self.n_stabilizers, self.n_clbits)

        # define data and stabilizer indices based on even/odd parity

        self.stabilizers = [i for i in range((self.n_data + self.n_stabilizers)) if i % 2 == 1]
        data_qubits, x_stabilizers, z_stabilizers = self._get_surface_code_layout()
        self.data = data_qubits
        self.x_stabilizers = x_stabilizers
        self.z_stabilizers = z_stabilizers

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
        Return indices of X stabilizers, Z stabilizers, and data qubits
        for a planar surface code of given distance.
        Layout follows the checkerboard pattern on a (2d-1) x (2d-1) grid.
        """
        grid_size = 2*self.d - 1

        def index(r, c):
            return r * grid_size + c

        # create coordinate grid
        rows, cols = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")

        # checkerboard masks
        mask_data = (rows + cols) % 2 == 0
        mask_x    = (rows % 2 == 0) & (cols % 2 == 1)
        mask_z    = (rows % 2 == 1) & (cols % 2 == 0)

        # collect indices
        data_qubits   = [index(r, c) for r, c in zip(rows[mask_data], cols[mask_data])]
        x_stabilizers = [index(r, c) for r, c in zip(rows[mask_x], cols[mask_x])]
        z_stabilizers = [index(r, c) for r, c in zip(rows[mask_z], cols[mask_z])]

        return data_qubits, x_stabilizers, z_stabilizers

    def _qubit_index(self, r, c):
            return r * self.width + c
    
    def _get_neighbors(self, r, c):
            """Return indices of data qubits adjacent to stabilizer at (r,c)."""
            neighbors = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r+dr, c+dc
                if 0 <= rr < self.width and 0 <= cc < self.width:
                    neighbors.append(self._qubit_index(rr, cc))
            return neighbors

    def _build_stabilizer_layer(self):
        """Build one stabilizer measurement cycle for a distance-n surface code."""

        qubits = self.width
        cycles = self.cycles

        # repeat the stabilizer-measurement process for all cycles
        for cycle in range(cycles):
            # --- Reset all stabilizers at the beginning of the cycle ---
            for anc in self.x_stabilizers + self.z_stabilizers:
                self.qc.reset(anc)
                
            # --- X stabilizers ---
            for anc in self.x_stabilizers:
                r, c = divmod(anc, qubits) # converts a 1D index into a (row, col) pair
                
                self.qc.h(anc)  # prepare X stabilizer in |+>
                self.qc.barrier()

                # entangle with data qubits (ancilla is control)
                for nb in self._get_neighbors(r, c):
                    self.qc.cx(anc, nb)

                self.qc.barrier()
                self.qc.h(anc)  # rotate back before measurement

            # --- Z stabilizers ---
            for anc in self.z_stabilizers:
                r, c = divmod(anc, qubits)
                # entangle with data qubits (data is control, ancilla target)
                for nb in self._get_neighbors(r, c):
                    self.qc.cx(nb, anc)

                self.qc.barrier()
            
            #conditional gate on the mid-circuit measurement results to run the decoder and determine the basis for the end measrement.

            # --- Measure all stabilizers for this cycle --- 
            classical_offset = cycle * self.n_stabilizers
            stab_list = sorted(self.x_stabilizers + self.z_stabilizers)
            for i, anc in enumerate(stab_list):
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
        t_circ.draw('mpl')

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

        #Calucate Fidelity per cycle

        # extract only the final statevector of each cycle
        self._fidelity_per_cycle(statevector_readout, mid_counts)

        decoder_x, decoder_z = self.decode_full(mid_counts)
        print("Decoder output X:", decoder_x)
        print("Decoder output Z:", decoder_z)
        print("Measurement results:", mid_counts)
        # XOR over cycles → net correction
        """
        for cycle in range(1, self.cycles):
            U_map = self._create_U_map(decoder_x[cycle], decoder_z[cycle])
            print(f"U map after cycle {cycle}:", U_map)
            #apply U to the statevector of the dataqubits only and save as statevctor list where statevector[c] 
            #is the error corrected statevector after cycle c 
        """
        # now compute fidelity of statevector[c-1] and statevector[c] for each cycle c and plot it 


        final_x_correction = np.bitwise_xor.reduce(decoder_x, axis=0)
        final_z_correction = np.bitwise_xor.reduce(decoder_z, axis=0)

        print("Final X correction on data qubits:", final_x_correction)
        print("Final Z correction on data qubits:", final_z_correction)

        U_map = self._create_U_map(final_x_correction, final_z_correction)
        print("U map (data qubit corrections):", U_map)

        #return statevector on dataqubits
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
        # Sort to get consistent ordering
        data_qubits = sorted(self.data)
        stab_qubits = self.stabilizers
        # reshape full statevector into tensor form
        psi = statevector.reshape([2] * total_qubits)

        # axes order: data qubits first, stabilizers last
        perm = data_qubits + stab_qubits
        psi_perm = np.transpose(psi, axes=perm)

        nd = len(data_qubits)
        ns = len(stab_qubits)

        # reshape to matrix form: (2**nd) × (2**ns)
        psi_matrix = psi_perm.reshape(2**nd, 2**ns)

        # density matrix on full register
        rho_full = psi_matrix @ psi_matrix.conj().T  # trace over stabilizers

        return rho_full

    def _fix_density_matrix(self, rho, eps=1e-12):
        # Enforce Hermiticity
        rho = 0.5 * (rho + rho.conj().T)

        # Diagonalize
        vals, vecs = np.linalg.eigh(rho)

        # Clamp negative eigenvalues (due to rounding)
        vals = np.maximum(vals, eps)

        # Rebuild PSD matrix
        rho = vecs @ np.diag(vals) @ vecs.conj().T

        # Normalize trace
        rho /= np.trace(rho)

        return rho


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
            A = sqrtm(self._fix_density_matrix(rho_full_last_corrected))
            F = np.real(np.trace(sqrtm(A @ self._fix_density_matrix(rho_copy) @ A)))**2
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
            r, c = divmod(stab, self.width)
            neighbors = self._get_neighbors(r, c)
            # Filter to only include data qubits
            data_neighbors = [nb for nb in neighbors if nb in self.data]
            connections[stab] = data_neighbors
            
            # Debug print
            stab_type = 'X' if stab in self.x_stabilizers else 'Z'
            print(f"{stab_type}-stab at qubit {stab} (r={r}, c={c}) -> neighbors {neighbors} -> data: {data_neighbors}")
        
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
        
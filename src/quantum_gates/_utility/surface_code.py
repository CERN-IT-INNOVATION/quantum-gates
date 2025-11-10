from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import random
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import XGate, YGate, ZGate
import pymatching
from scipy.sparse import csr_matrix, lil_matrix

class SurfaceCode:
    def __init__(self, distance=3, cycles = 1):
        self.d = distance
        self.cycles = cycles
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

            # --- Measure all stabilizers for this cycle --- 
            classical_offset = cycle * self.n_stabilizers
            stab_list = sorted(self.x_stabilizers + self.z_stabilizers)
            for i, anc in enumerate(stab_list):
                self.qc.measure(anc, classical_offset + i)

            self.qc.barrier()

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

    def plot_results(self, counts, total_shots, plot_each_shot):
        """
        Plot measurement results for the surface code.
        - If 1 shot: timeline plot of stabilizer outcomes per cycle.
        - If >1 shots:
            - If plot_each_shot=True: plot each shot individually like single-shot mode.
            - If plot_each_shot=False: plot aggregated histogram (default behavior).
        """

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
        print("Correction:", which, prediction)
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
        
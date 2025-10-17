from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import random
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import XGate, YGate, ZGate


class SurfaceCode:
    def __init__(self, qc, distance=3, cycles = 1):
        self.d = distance
        self.cycles = cycles
        self.n_data = distance**2 + (distance - 1)**2
        self.n_stabilizers = 2*(distance-1)* distance
        self.n_clbits = self.n_stabilizers * cycles
        self.width = 2*distance-1

        expected_qubits = self.n_data + self.n_stabilizers
        expected_clbits = self.n_clbits

        if qc.num_qubits != expected_qubits:
            raise ValueError(f"Expected {expected_qubits} qubits, got {qc.num_qubits}")

        if qc.num_clbits != expected_clbits:
            raise ValueError(f"Expected {expected_clbits} classical bits, got {qc.num_clbits}")

        # define data and stabilizer indices based on even/odd parity

        self.stabilizers = [i for i in range((self.n_data + self.n_stabilizers)) if i % 2 == 1]
        data_qubits, x_stabilizers, z_stabilizers = self.get_surface_code_layout()
        self.data = data_qubits
        self.x_stabilizers = x_stabilizers
        self.z_stabilizers = z_stabilizers

        # maps stabilizer index -> list of classical bit indices (one per cycle)
        self.stabilizer_to_clbits = {
            stab: [cycle * self.n_stabilizers + i for cycle in range(self.cycles)]
            for i, stab in enumerate(self.stabilizers)
        }

        self._build_stabilizer_layer()

    def index(self,r, c):
            return r * self.width + c
  
    def get_surface_code_layout(self):
        """
        Return indices of X stabilizers, Z stabilizers, and data qubits
        for a planar surface code of given distance.
        Layout follows the checkerboard pattern on a (2d-1) x (2d-1) grid.
        """

        
        # create coordinate grid
        rows, cols = np.meshgrid(np.arange(self.width), np.arange(self.width), indexing="ij")

        # checkerboard masks
        mask_data = (rows + cols) % 2 == 0
        mask_x    = (rows % 2 == 0) & (cols % 2 == 1)
        mask_z    = (rows % 2 == 1) & (cols % 2 == 0)

        # collect indices
        data_qubits   = [self.index(r, c) for r, c in zip(rows[mask_data], cols[mask_data])]
        x_stabilizers = [self.index(r, c) for r, c in zip(rows[mask_x], cols[mask_x])]
        z_stabilizers = [self.index(r, c) for r, c in zip(rows[mask_z], cols[mask_z])]

        return data_qubits, x_stabilizers, z_stabilizers


    def _build_stabilizer_layer(self):
        """Build one stabilizer measurement cycle for a distance-n surface code."""

        cycles = self.cycles


        def get_neighbors(r, c):
            """Return indices of data qubits adjacent to stabilizer at (r,c)."""
            neighbors = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r+dr, c+dc
                if 0 <= rr < self.width and 0 <= cc < self.width:
                    neighbors.append(self.index(rr, cc))
            return neighbors

        # repeat the stabilizer-measurement process for all cycles
        for cycle in range(cycles):
            # --- Reset all stabilizers at the beginning of the cycle ---
            for anc in self.x_stabilizers + self.z_stabilizers:
                self.qc.reset(anc)
                
            # --- X stabilizers ---
            for anc in self.x_stabilizers:
                r, c = divmod(anc, self.width) # converts a 1D index into a (row, col) pair
                
                self.qc.h(anc)  # prepare X stabilizer in |+>
                self.qc.barrier()

                # entangle with data qubits (ancilla is control)
                for nb in get_neighbors(r, c):
                    self.qc.cx(anc, nb)

                self.qc.barrier()
                self.qc.h(anc)  # rotate back before measurement

            # --- Z stabilizers ---
            for anc in self.z_stabilizers:
                r, c = divmod(anc, self.width)
                # entangle with data qubits (data is control, ancilla target)
                for nb in get_neighbors(r, c):
                    self.qc.cx(nb, anc)

                self.qc.barrier()

            self.qc.barrier()
            # --- Measure all stabilizers for this cycle --- 
            classical_offset = cycle * self.n_stabilizers
            stab_list = sorted(self.x_stabilizers + self.z_stabilizers)
            for i, anc in enumerate(stab_list):
                self.qc.measure(anc, classical_offset + i)

            self.qc.barrier()


    def run_cycle(self, shots=256):
        simulator = Aer.get_backend('qasm_simulator') #'aer_simulator'
        transpiled_circuit = transpile(self.qc, simulator)
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        self.plot_results(counts, shots)
        return counts
    
   
    def analyze_results(self, counts):
        """
        Analyze measurement results using stabilizer-to-classical-bit mapping.
        Returns a dict: stabilizer → {bitstring_pattern: frequency}.
        """
        results = {stab: {} for stab in (self.x_stabilizers + self.z_stabilizers)}

        for bitstring, count in counts.items():
            bits = bitstring[::-1]  # reverse Qiskit order

            for stab in (self.x_stabilizers + self.z_stabilizers):
                # classical bits for this stabilizer across all cycles
                clbits = self.stabilizer_to_clbits[stab]
                bits_for_stab = [bits[i] for i in clbits]
                key = tuple(bits_for_stab)

                results[stab][key] = results[stab].get(key, 0) + count

        # pretty string output
        pretty = {
            stab: {','.join(k): v for k, v in res.items()} for stab, res in results.items()
        }

        return pretty



    def plot_results(self, counts, total_shots):
        """
        Plot measurement results for the surface code.
        - If 1 shot: timeline plot of stabilizer outcomes per cycle.
        - If >1 shots: histogram of syndromes per stabilizer.
        """
        # --- Single-shot timeline ---
        if total_shots == 1:
            bitstring = list(counts.keys())[0][::-1]

            plt.figure(figsize=(10, 3))

            # Plot X stabilizers
            for stab in self.x_stabilizers:
                bits = [int(bitstring[b]) for b in self.stabilizer_to_clbits[stab]]
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
            for stab in self.z_stabilizers:
                bits = [int(bitstring[b]) for b in self.stabilizer_to_clbits[stab]]
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

            plt.title("Single-shot stabilizer measurements", fontsize=14)
            plt.xlabel("Cycle", fontsize=12)
            plt.ylabel("Measurement", fontsize=12)
            plt.yticks([0, 1])
            plt.xticks(range(1, self.cycles + 1))
            plt.grid(alpha=0.4)
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.show()

        # --- Multi-shot histogram ---
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




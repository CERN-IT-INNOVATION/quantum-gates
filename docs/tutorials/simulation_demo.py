"""
Run the Noisy Quantum Gates simulation for demonstration purposes.

Usage:
- Do pip install quantum-gates.
- Go to IBM Quantum Lab and get your token.
- Copy paste your IBM token into configuration/token.py as variable IBM_TOKEN. This file should not be versioned.
- Set the hub, group, project and device name in this script.
- Run the script with path on top level of the repository.
"""

# Standard libraries
import numpy as np
import json

# Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram

# Own library
from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import gates
from quantum_gates.circuits import EfficientCircuit
from quantum_gates.utilities import DeviceParameters
from quantum_gates.utilities import setup_backend
from configuration.token import IBM_TOKEN


""" Setup backend """

config = {
    "backend": {
        "hub": "ibm-q",
        "group": "open",
        "project": "main",
        "device_name": "ibmq_manila"
    },
    "run": {
        "shots": 1000,
        "qubits_layout": [0, 1],
        "psi0": [1, 0, 0, 0]
    }
}
backend_config = config["backend"]
backend = setup_backend(Token=IBM_TOKEN, **backend_config)


""" Create a Quantum circuit with Qiskit """

circ = QuantumCircuit(2, 2)

circ.h(0)
circ.cx(0, 1)
circ.barrier(range(2))
circ.measure(range(2), range(2))
circ.draw('mpl')

""" Execute simulation """

sim = MrAndersonSimulator(gates=gates, CircuitClass=EfficientCircuit)
shots = config["run"]["shots"]
qubits_layout = config["run"]["qubits_layout"]
psi0 = np.array(config["run"]["psi0"])
t_circ = transpile(
    circ,
    backend,
    scheduling_method='asap',
    initial_layout=qubits_layout,
    seed_transpiler=42
)

# Load device parameters (noise)
device_param = DeviceParameters(qubits_layout)
device_param.load_from_backend(backend)
device_param_lookup = device_param.__dict__()

probs = sim.run(
    t_qiskit_circ=t_circ,
    qubits_layout=qubits_layout,
    psi0=psi0,
    shots=shots,
    device_param=device_param_lookup,
    nqubit=2
)
counts_ng = {format(i, 'b').zfill(2): probs[i] for i in range(0, 4)}


""" Result """

print(f"The output probabilities are {probs}.")

legend = ['Noisy Gates simulation']
plot_histogram(counts_ng, bar_labels=False, legend=legend)

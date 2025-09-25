Quantum Algorithms
==================

Functions to set up common `quantum
algorithms <https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html>`__
in Qiskit for a given number of qubits.


All functions take the number of qubits as argument and return the
corresponding Qiskit circuit. One can then use the Qiskit backend to
transpile the circuit for a specific quantum device.

.. code:: python

   from quantum_gates.quantum_algorithms import (
       hadamard_reverse_qft_circ,
       ghz_circ,
       qft_circ
       qaoa_circ
   )

   nqubit = 2
   circuit = hadamard_reverse_qft_circ(nqubit=2)
   circ.draw('mpl')


.. automodule:: quantum_gates.quantum_algorithms
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:

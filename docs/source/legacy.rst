Legacy
======

The original versions of the simulator, circuit and gates are provided
for use in unit testing.

.. _legacy_classes:

Classes
-------

.. _legacymrandersonsimulator:

LegacyMrAndersonSimulator
~~~~~~~~~~~~~~~~~~~~~~~~~

The simulator has a slightly different interface than the
:ref:`MrAndersonSimulator <mrandersonsimulator>`. 

.. code:: python

   from quantum_gates.simulators import LegacyMrAndersonSimulator

   sim = LegacyMrAndersonSimulator()

   sim.run(qiskit_circ,
           backend,
           qubits_layout,
           psi0,
           shots,
           device_param)

.. _legacycircuit:

LegacyCircuit
~~~~~~~~~~~~~

.. code:: python

   from quantum_gates.circuits import LegacyCircuit

   n = 2   # Number of qubits
   d = 1   # Depth of the circuit

   circuit = LegacyCircuit(n, d)

   # Apply gates
   circuit.X(i=0, p=..., T1=..., T2=...)
   circuit.I(i=1)

   # Statevector simulation
   psi0 = np.array([1, 0, 0, 0])
   psi1 = circuit.statevector(psi0)  # Gives [0  0  1  0]

.. _legacygates:

LegacyGates
~~~~~~~~~~~

See :ref:`standard gates <standard_gates>` for the interface.

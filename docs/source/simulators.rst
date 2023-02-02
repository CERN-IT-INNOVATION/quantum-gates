Simulators
==========

The MrAndersonSimulator is a statevector simulator for the noisy quantum
gates approach.

.. _simulators_classes:

Classes
-------

.. _mrandersonsimulator:

MrAndersonSimulator
~~~~~~~~~~~~~~~~~~~

We refer to the `demo <../tutorials/simulation_demo.py>`__ for a usage
sample of the simulator. The simulator has three attributes. While the
`gates <./gates.md>`__ attribute implicitely specifies the pulse shape,
the `CircuitClass <circuits.md>`__ knows how to perform the
matrix-vector multiplication seen in the statevector simulation. Last,
the parallel attribute specifies whether or not the shots should be
executed in parallel with multiprocessing. In general, we recommend to
parallelize the outer loop of the simulation, and not the shots, as it
comes with a large overhead.

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

LegacyMrAndersonSimulator
~~~~~~~~~~~~~~~~~~~~~~~~~

This version is only meant for unit testing and the documentation can be
found :ref:`here <mrandersonsimulator>`.

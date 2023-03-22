Simulators
==========

We refer to the `demo <../tutorials/simulation_demo.py>`__ for a usage
sample of the simulator. The simulator has three attributes. While the
`gates <./gates.md>`__ attribute implicitely specifies the pulse shape,
the `CircuitClass <circuits.md>`__ knows how to perform the
matrix-vector multiplication seen in the statevector simulation. Last,
the parallel attribute specifies whether or not the shots should be
executed in parallel with multiprocessing. In general, we recommend to
parallelize the outer loop of the simulation, and not the shots, as the
latter comes with a large overhead.


.. automodule:: quantum_gates.simulators
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:

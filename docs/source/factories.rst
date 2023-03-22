Factories
=========

The gate factories generate noisy quantum gates stochastically and return 
them as as numpy matrices.


.. _factories_theory:

Theory
------

The two main factors that influence the distribution of noisy gates are
the following. First, the :ref:`device parameters <deviceparameters>` 
for a specific quantum device, containing information such as the 
T1, T2 times. Second, the :doc:`pulse shape <pulses>` used to execute a 
specific gate. While the pulse shape is independent of the qubit we act on, 
the device parameters are qubit specific. Thus, we choose to add the pulse 
shape as attribute of the :doc:`gate factory <factories>`, while the 
specific noise parameters are provided as method arguments when creating a gate.


.. _factories_usage:

Usage
-----

The are two types of classes, namely the ones representing pure noise
and the ones representing the noisy gates. The former do not need the
pulse as argument, as they do not come from the execution of a gate, and
thus the pulse does not matter.

.. code:: python

   from quantum_gates.factories import (
       BitflipFactory, 
       DepolarizingFactory,
       RelaxationFactory
   )

   bitflip_factory = BitflipFactory()

   tm = ...    # measurement time in ns
   rout = ...  # readout error 
   bitflip_gate = bitflip_factory.construct(tm, rout)

In the latter, we specify the pulse shape used in the execution of the
gates.

.. code:: python

   from quantum_gates.factories import (
       SingleQubitGateFactory,
       XFactory, 
       SXFactory, 
       CRFactory, 
       CNOTFactory,
       CNOTInvFactory
   )
   from quantum_gates.pulses import GaussianPulse

   pulse = GaussianPulse(loc=0.5, scale=0.5)
   x_factory = XFactory(pulse)

   phi = ...   # Phase of the drive defining axis of rotation on the Bloch sphere
   p = ...     # Single-qubit depolarizing error probability
   T1 = ...    # Qubit's amplitude damping time in ns 
   T2 = ...    # Qubit's dephasing time in ns
   x_gate = x_factory.construct(phi, p, T1, T2)


.. _factories_classes:

Classes
-------

.. automodule:: quantum_gates.factories
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:



Circuits
========

Quantum circuit can be constructed easily when we have a circuit object
to which we can apply gates by simply calling method like
circuit.H(i=0). This functionality is provided in the Circuit classes,
and packed together with the knowledge on how to sample the noisy gates
as stochastic matrices from a specific gate set.


.. _circuits_classes:

Classes
-------

We provide different versions, some of which can be used with a specific
backend to speed up the computations.

.. _circuit:

Circuit
~~~~~~~

The Circuit class uses trivial matrix multiplications to perform the
statevector simulation. Thus, it only scales to 13 qubits, and we only
use it for unit testing.

.. code:: python

   from quantum_gates.circuits import Circuit
   from quantum_gates.gates import standard_gates

   # The depth has to be set correctly for this class. 
   circuit = Circuit(nqubit=2, depth=1, gates=standard_gates)

   # We apply gates for one timestep.
   circuit.X(i=0, ...)
   circuit.I(i=1) 

   # Evaluate the statevector
   psi1 = circuit.statevector(psi0=np.array([1, 0, 0, 0]))  # Gives [0, 0, 1, 0]

.. _alternative_circuit:

AlternativeCircuit
~~~~~~~~~~~~~~~~~~

We can us the AlternativeCircuit class to build a circuit with a custom
`backend <backends.md>`__ for performing the computations.

.. code:: python

   from quantum_gates.circuits import AlternativeCircuit
   from quantum_gates.backends import EfficientBackend
   from quantum_gates.gates import standard_gates

   circuit = AlternativeCircuit(
       nqubit=2, 
       gates=standard_gates, 
       BackendClass= EfficientBackend
   )

This class does not have a depth attribute, it is flexible.


.. _standard_circuit:

StandardCircuit
~~~~~~~~~~~~~~~

Combining the AlternativeCircuit with the trivial
`StandardBackend <backends.md#standardbackend>`__, we get the
StandardCircuit. Again, we only recommend to use it for unit testing.


.. _efficient_circuit:

EfficientCircuit
~~~~~~~~~~~~~~~~

This is the class we recommend to use for simulations, as it uses the
optimized `EfficientBackend <backends.md#efficientbackend>`__ for the
computations.


.. _circuits_usage:

Usage
-----

The class is easy to use besides one tricky detail. For each gatetime
(timestep), one has to fill in all gates. For example, consider the two
qubit case. We apply an X gate on qubit 0 and then a CNOT gate on both
qubits with control on 0 and target on 1. Then we use the following
code:

.. code:: python

   from quantum_gates.circuits import EfficientCircuit
   from quantum_gates.gates import standard_gates

   # The depth can be set arbitrarily for this circuit class. 
   circuit = EfficientCircuit(nqubit=2, depth=0, gates=standard_gates)

   # First timestep -> Each qubit has to get a gate, this is why we apply even identities.
   circuit.H(i=0, ...)
   circuit.I(i=1) 

   # Second timestep -> All qubits received a gate, because CNOT is a two-qubit gate.
   circuit.CNOT(i=0, k=1, ...)

   # Evaluate the statevector
   psi1 = circuit.statevector(psi0=np.array([1, 0, 0, 0])) 

Not applying gates to each qubit will lead to errors.


.. _circuits_possible_extensions:

Possible extensions
-------------------

In the future, we could add checks that warns the user in case not all
gates are applied.

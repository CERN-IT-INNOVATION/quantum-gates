Backends
========

While the circuit class samples the matrices making up a quantum circuit, 
the backend performs the actual matrix multiplications which appear in 
the statevector simulation.

.. _backends_theory:

Theory
------

Statevector simulation
~~~~~~~~~~~~~~~~~~~~~~

The statevector simulation takes an input state psi0, and propagates it
to the final state psi1 by repeated left matrix multiplications.
Assuming that our circuit has k steps, then we compute psi1 = U psi0 =
U_k … U_1 psi0 where U_i are matrices of dimension (2^n, 2^n) for n
qubit psi0 of shape (2^n,).

Tensors
~~~~~~~

Tensors are simply generalized versions of scalars, vectors and
matrices. We can imagine it to be a box with incoming and outgoing legs.
While a scalar has no legs, a vector has one leg, and a matrix has one
incoming and one outgoing leg. We can interpret matrix-vector
multiplication as connecting the right leg of the matrix with the left
leg of the vector. This will create an object with one free left leg,
which represents a vector as expected. Similarly, matrix-matrix
multiplication will led to an object with one incoming and one outgoing
leg, so a matrix.

Optimal tensor contractions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

As explained in `this
resource <https://optimized-einsum.readthedocs.io/en/stable/>`__, the
ordering of the tensor contractions (think matrix multiplications) can
influence the computational cost dramatically. In the EfficientBackend,
we leverage optimizations based on the property of the quantum circuit,
as well as optimizations performed by the library opt_einsum. This
library computes the contractions for us.


.. _backends_classes:

Classes
-------

.. automodule:: quantum_gates.backends
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:


Usage
-----

We apply the GHZ algorithm on the two-qubit zero state as an example.

.. code:: python

   from quantum_gates.backend import EfficientBackend

   backend = EfficientBackend(nqubit=2)

   H, CNOT, identity = ...
   mp_list = [[H, np.eye(2)], [CNOT]]

   psi0 = np.array([1, 0, 0, 0])
   psi1 = backend.statevector(mp_list, psi0)  # Gives [1, 0, 0, 1] / sqrt(2)

Not applying gates to each qubit will lead to errors. In each matrix
product, the dimension of the combined Kronecker product has to match
the dimension of psi.

.. code:: python

   from quantum_gates.backend import EfficientBackend

   backend = EfficientBackend(nqubit=2)

   H, CNOT = ...

   mp_list1 = [[CNOT, np.eye(2)]]              # Gates for 2 + 1 = 3 qubits -> wrong.
   mp_list2 = [[CNOT]]                         # Gates for 2 qubits -> fine. 
   mp_list3 = [[np.eye(2), np.eye(2)], [H, H]] # Gates for 1 + 1 = 2 qubits -> fine.




.. _backends_possible_extensions:

Possible extensions
-------------------

The backend class has a simple, generic public interface. This makes
implementing a custom backend easy. One can optimize the backend
performance by exploiting properties of a specific quantum algorithm,
use another simulation method like Matrix Product State (MPS)
simulation, or use another library as backend for the computations.

Welcome to the Quantum Gates documentation
==========================================

**Quantum Gates** is a Python library for simulating the noisy behaviour of 
real quantum devices. The noise is incorporated directly into the gates, 
which become stochastic matrices. 

.. note::

   This project is under active development.


How to install
--------------

Requirements
~~~~~~~~~~~~

The Python version should be 3.9 or later. We recommend using the library
together with a |ibm_quantum_lab_link| account, as it necessary for 
circuit compilation with Qiskit.

.. |ibm_quantum_lab_link| raw:: html

   <a href="https://quantum-computing.ibm.com/lab" target="_blank">IBM Quantum Lab</a>

Installation as a user
~~~~~~~~~~~~~~~~~~~~~~

The library is available on the |pip_link| with ``pip install quantum-gates``.

.. |pip_link| raw:: html

   <a href="https://pypi.org/project/quantum-gates/" target="_blank">Python Package Index (PyPI)</a>


Installation as a contributor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For users who want to have access to the source code, we recommend cloning 
the repository from |github_link|.

.. |github_link| raw:: html

   <a href="https://github.com/CERN-IT-INNOVATION/quantum-gates" target="_blank">Github</a>


Functionality
--------------

Gates
~~~~~

To sample quantum gates with the noise incorporated in them, one can set
up a :doc:`gates` instance. The methods of this object return the matrices
as numpy array. The sampling is stochastic, and one can set the numpy seed
to always get the same sequence of noise.


Factories
~~~~~~~~~

To produce :doc:`gates <gates>`, we use :doc:`factories <factories>`, such as the
CNOTFactory. One can combine factories into a custom :doc:`gates <gates>`
class. The factories have a construct() method, with a well documented 
signature. 

Pulses
~~~~~~

When constructing a set of quantum gates with the Gates class, one can
specify a Pulse instance. This pulse describes the shape of the RF pulses
used to implement the gates.

Integrators
~~~~~~~~~~~

Behind the scenes, we solve Ito integrals to deal with the different
pulse shapes. This is handled by the :doc:`integrator <integrators>`.

Simulators
~~~~~~~~~~

The :doc:`LegacyMrAndersonSimulator <simulators>` can be used to simulate
a quantum circuit transpiled with Qiskit with a specific 
:doc:`noisy gate set <gates>`.

Backends
~~~~~~~~

For the computation, we provide :doc:`backends <backends>` out of the box, 
such as the EfficientBackend that uses optimized tensor contractions
to simulate 20+ qubits with the statevector method.

Circuits
~~~~~~~~

The simulators can be configured with a :doc:`circuits` class, such as 
EfficientCircuit. This class is responsible for sampling the
noisy gates. The class can be configured with a :doc:`gates` instance and one of 
the :doc:`backends` that executes the statevector simulation.

Quantum Algorithms
~~~~~~~~~~~~~~~~~~

Four quantum algorithms are provided as functions which return the Qiskit circuit
for a specific number of qubits

* Hadamard reverse QFT circuit
* GHZ circuit
* QFT circuit
* Quantum Approximate Optimization Algorithm circuit


Legacy
~~~~~~

We also provide the :doc:`legacy <legacy>` implementations of the 
gates, simulator and circuit classes. They can be used for unit testing.

Utility
~~~~~~~

In performing quantum simulation, there are many steps that are performed repeatedly, such as setup the IBM backend with
setup_backend(), loading the noise information as DeviceParameters, transpiling the quantum circuits with
create_qc_list(), and executing the simulation in parallel  with multiprocessing_parallel_simulation() on a
powerful machine. For this reason, the most frequently used functions are part of the utilities.


Structure
---------

.. toctree::
   :maxdepth: 3
   :caption: Gates

   gates

.. toctree::
   :maxdepth: 3
   :caption: Factories

   factories

.. toctree::
   :maxdepth: 3
   :caption: Pulses

   pulses

.. toctree::
   :maxdepth: 3
   :caption: Integrators

   integrators


.. toctree::
   :maxdepth: 2
   :caption: Simulators

   simulators

.. toctree::
   :maxdepth: 3
   :caption: Backends

   backends

.. toctree::
   :maxdepth: 3
   :caption: Circuits

   circuits

.. toctree::
   :maxdepth: 3
   :caption: Quantum Algorithm

   quantum_algorithms

.. toctree::
   :maxdepth: 3
   :caption: Legacy

   legacy

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   utilities


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Authors
-------

This project has been developed thanks to the effort of the following
people:

-  Giovanni Di Bartolomeo (dibartolomeo.giov@gmail.com)
-  Michele Vischi (vischimichele@gmail.com)
-  Francesco Cesa
-  Michele Grossi (michele.grossi@cern.ch)
-  Sandro Donadi
-  Angelo Bassi
-  Roman Wixinger (roman.wixinger@gmail.com)

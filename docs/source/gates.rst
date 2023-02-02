Gates
=====

Here you learn how to use the different gate sets to sample noisy quantum
gates. The gate set combines multiple gate factories into a single
object, which is easy to import and pass in arguments.

Usage
-----

While the usage is fairly trivial, there is one dangerous subtlety that
the user should be aware of. As computer programs are deterministic, the
sampling from probability distributions often works by using a
pseudo-random number generator underneath. Thus, the user should not
accidentaly sample the same gates multiple times, which can happen when
the same simulation is run in parallel many times, and the seed of the
pseudo-random number generator is the same in each run. An easy fix would
be to set a random seed in the start of each simulation.

.. code:: python

   # Set random seed, otherwise each experiment gets the same result
   seed = (os.getpid() * int(time.time())) % 123456789
   np.random.seed(seed)

Another method would be to pass the seed to the simulation. When
reproducability of the simulation results is important, then this is the
better option. Note how the seed is set inside the simulation and not
outside.

.. code:: python

   import multiprocessing

   def simulation(seed) -> float: 
       """ Returns a random number in [0, 1] for a specific random seed. """ 
       np.random.seed(seed)
       return np.random.rand(1)

   args = range(100)
   p = multiprocessing.Pool(4)

   for result in p.imap_unordered(func=simulation, iterable=args, chunksize=100//4):
       print(f"Result: {result}")

Note that this possible issue is specific to multiprocessing on Linux
operating system, as it depends on how new processes are created.


.. _gates_classes:

Classes
-------

One can construct gate sets with specific pulse shapes, either by
creating and instance or by using inheritance.

Gates
~~~~~

One can create a pulse (optional) and provide it to the constructor.

.. code:: python

   from quantum_gates.pulses import GaussianPulse
   from quantum_gates.gates import Gates

   pulse = GaussianPulse(loc=1, scale=1)
   gateset = Gates(pulse)

   sampled_x = gateset.X(phi, p, T1, T2)

By default, the standard_pulse is used, so we can also do

.. code:: python

   gateset = Gates()


.. _noisefreegates:

NoiseFreeGates
~~~~~~~~~~~~~~

Independent of our choice of input parameters, this class will always
return the noise free result.

.. code:: python

   from quantum_gates.gates import NoiseFreeGates

   gateset = NoiseFreeGates()
   sampled_x = gateset.X(phi, p, T1, T2)


.. _scalednoisegates:

ScaledNoiseGates
~~~~~~~~~~~~~~~~

In some cases it can be interesting to see what happens when we scale
the amount of noise by a certain factor. We map error probabilities as p
-> noise_scaling \* p and times like the depolarization time as T -> T /
noise_scaling.

.. code:: python

   from quantum_gates.gates import ScaledNoiseGates

   gateset = ScaledNoiseGates(noise_scaling=0.1, pulse=pulse)  # 10x less noise
   sampled_x = gateset.X(phi, p, T1, T2)


.. _gates_instances:

Instances
---------

For common cases we provide working gate set instances out of the box.

.. _standard_gates:

standard_gates
~~~~~~~~~~~~~~

Uses a constant pulse shape.

.. code:: python

   from quantum_gates.gates import standard_gates, noise_free_gates, legacy_gates

   sampled_x = standard_gates.X(phi, p, T1, T2)


.. _noise_free_gates:

noise_free_gates
~~~~~~~~~~~~~~~~

Uses a constant pulse shape and returns the result in the noise free
regime irrespective of the arguments provided to its methods.


.. _legacy_gates:

legacy_gates
~~~~~~~~~~~~

Original implementation of the gates, which we use for unit testing.


.. _gates_supported_gates:

Supported gates
---------------

At the moment, we support the following gates: - X - SX - CNOT -
CNOT_inv - CR - SingleQubitGate

The signature is the same for each gate class. This makes changing the
gate class easy.

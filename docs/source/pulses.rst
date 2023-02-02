Pulses
======

Gates on real quantum devices are commonly implemented as radiofrequency
(RF) pulses. The present classes represent the pulse shapes, which can
then be used to build to construct a gateset.


.. _pulses_theory:

Theory
------

Coming soon.


.. _pulses_classes:

Classes
-------

.. _pulse:


Pulse
~~~~~

One can create a custom pulse. Note that the pulse and its
parametrization have to match. When the perform_checks flag is set to
True, this is verified upon instantiation. In the following, we create a
constant “rectangle” pulse.

.. code:: python

   from quantum_gates.pulses import Pulse

   pulse = lambda x: 1
   parametrization = lambda x: x

   constant_pulse = Pulse(
       pulse=pulse, 
       parametrization=parametrization, 
       perform_checks=False
       )


.. _gaussian_pulse:

GaussianPulse
~~~~~~~~~~~~~

To create a pulse with a Gaussian shape, one can use this pulse. Note
that not all values of loc and scale are valid. The integral over the
interval [0,1] of the choosen Gaussian should be larger than 1e-6. This
is because the shape of the pulse is the shape that the Gaussian has in
this interval.

.. code:: python

   from quantum_gates.pulses import GaussianPulse

   loc = 0.5   # Location of the Gaussian
   scale = 0.5 # Standard deviation of the Gaussian

   constant_pulse = GaussianPulse(loc=loc, scale=scale)


.. _pulses_instances:

Instances
---------

Coming soon.

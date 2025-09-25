import pytest
import numpy as np
import time

from src.quantum_gates.integrators import Integrator
from src.quantum_gates.pulses import constant_pulse, constant_pulse_numerical


def test_integrator_for_constant_pulses():
    """ As the result of the integrals is known for the constant pulse, we can use it to unit test the numerical
        integration.
    """

    # Create instances
    default_integrator = Integrator(pulse=constant_pulse)
    constant_integrator = Integrator(pulse=constant_pulse_numerical)  # Pulse is distributed evenly and thus constant

    # Test
    thetas = [1e-3, 1e-1, 1e-2, np.pi/8, np.pi/4, np.pi/2, np.pi]
    a_list = [0.1, 1, 10]
    integrands = [
        "sin(theta/a)**2",
        "sin(theta/(2*a))**4",
        "sin(theta/a)*sin(theta/(2*a))**2",
        "sin(theta/(2*a))**2",
        "cos(theta/a)**2",
        "sin(theta/a)*cos(theta/a)",
        "sin(theta/a)",
        "cos(theta/(2*a))**2"
    ]
    tuples = [(theta, integrand, a) for theta in thetas for integrand in integrands for a in a_list]

    for theta, integrand, a in tuples:
        default = default_integrator.integrate(integrand, theta, a)
        numerical = constant_integrator.integrate(integrand, theta, a)
        assert default == pytest.approx(numerical), f"Found error for Integrand {integrand}, theta {theta} and a {a}."


def test_integrator_caching():
    """ We test that the second evaluation of an integral is faster than the first.
    """

    # Create instances
    constant_integrator = Integrator(pulse=constant_pulse)

    # Test
    integrands = [
        "sin(theta/a)**2",
        "sin(theta/(2*a))**4",
        "sin(theta/a)*sin(theta/(2*a))**2",
        "sin(theta/(2*a))**2",
        "cos(theta/a)**2",
        "sin(theta/a)*cos(theta/a)",
        "sin(theta/a)",
        "cos(theta/(2*a))**2"
    ]
    thetas = np.linspace(1e-9, np.pi, 1000)

    for integrand in integrands:

        # First evaluation
        start = time.time()
        for theta in thetas:
            constant_integrator.integrate(integrand, theta, 1)
        end = time.time()
        first_time = end - start

        # Second evaluation
        start1 = time.time()
        for theta in thetas:
            constant_integrator.integrate(integrand, theta, 1)
        end1 = time.time()
        second_time = end1 - start1

        assert(first_time >= second_time)

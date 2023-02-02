import pytest
import numpy as np
import time
import pickle

from src.quantum_gates.pulses import Pulse, GaussianPulse, StandardPulse, StandardPulseNumerical


def test_gaussian_pulse_init():
    locs = np.linspace(0.0, 1.0, 5)
    scales = np.linspace(0.1, 1.0, 5)
    for loc in locs:
        for scale in scales:
            GaussianPulse(loc=loc, scale=scale)


def test_gaussian_pulse_get_pulse():
    gaussian_pulse = GaussianPulse(loc=0.5, scale=0.5)
    pulse = gaussian_pulse.get_pulse()
    pulse(0.5)


def test_gaussian_pulse_get_parametrization():
    gaussian_pulse = GaussianPulse(loc=0.5, scale=0.5)
    parametrization = gaussian_pulse.get_parametrization()
    parametrization(0.5)


def test_initialization_is_faster_without_checks():
    locs = np.linspace(0.0, 1.0, 10)

    # Without checks
    start_without = time.time()
    for loc in locs:
        GaussianPulse(loc=loc, scale=0.5, perform_checks=False)
    end_without = time.time()
    time_without = end_without - start_without

    # With checks
    start_with = time.time()
    for loc in locs:
        GaussianPulse(loc=loc, scale=0.5, perform_checks=True)
    end_with = time.time()
    time_with = end_with - start_with

    print("time_without", time_without)
    print("time_with", time_with)

    # Check that checks cost time
    assert time_with > time_without


def test_standard_pulse():
    sp = StandardPulse()
    pulse = sp.get_pulse()
    parametrization = sp.get_parametrization()
    pulse(0.5)
    parametrization(0.5)


def test_pickle_standard_pulse():
    pulse = StandardPulse()
    pickle.dumps(pulse)


def test_pickle_standard_pulse_numerical():
    pulse = StandardPulseNumerical()
    pickle.dumps(pulse)


def test_pickle_gaussian_pulse():
    pulse = GaussianPulse(loc=0.5, scale=1.0)
    pickle.dumps(pulse)
    assert abs(pulse.parametrization(0) - 0) < 1e-6 and abs(pulse.parametrization(1) - 1) < 1e-6

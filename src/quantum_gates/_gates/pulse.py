"""
Code for defining and parametrizing pulse shapes.
"""

import numpy as np
import scipy.integrate
import scipy.stats


class Pulse(object):
    """ Parent class for pulses with basic utility.
    """

    epsilon = 1e-6
    check_n_points = 10

    def __init__(self, pulse: callable, parametrization: callable, perform_checks: bool=False):
        if perform_checks:
            assert self._pulse_is_valid(pulse), "Pulse was not valid"
            assert self._parametrization_is_valid(parametrization), "Parametrization was not valid"
            assert self._are_compatible(pulse, parametrization), "Pulse and parametrization are incompatible. "
        self.pulse = pulse
        self.parametrization = parametrization

    def get_pulse(self):
        return self.pulse

    def get_parametrization(self):
        return self.parametrization

    def _pulse_is_valid(self, pulse) -> bool:
        """ Check that pulse is a probability distribution on [0,1]."""
        integrates_to_1 = abs(scipy.integrate.quad(pulse, 0, 1)[0] - 1) < self.epsilon
        is_non_negative = all((pulse(x) >= 0) for x in np.linspace(0, 1, self.check_n_points))
        return integrates_to_1 and is_non_negative

    def _parametrization_is_valid(self, parametrization) -> bool:
        """ Check that parametrization is monotone and has valid bounds. """
        starts_at_0 = abs(parametrization(0) - 0) < self.epsilon
        stops_at_0 = abs(parametrization(1) - 1) < self.epsilon
        is_monotone = all((parametrization(x + self.epsilon) >= parametrization(x))
                          for x in np.linspace(0, 1-self.epsilon, self.check_n_points))
        return starts_at_0 and stops_at_0 and is_monotone

    def _are_compatible(self, pulse, parametrization) -> bool:
        """ Checks if the integral of the pulse is the parametrization. """
        for x in np.linspace(self.epsilon, 1-self.epsilon, self.check_n_points):
            difference = abs(scipy.integrate.quad(pulse, 0, x)[0] - parametrization(x))
            if difference > self.epsilon:
                return False
        return True


class StandardPulse(Pulse):

    use_lookup = True  # We just lookup the result in the Integrator

    def __init__(self):
        super(StandardPulse, self).__init__(
            pulse=one,
            parametrization=identity,
            perform_checks=False
        )


class StandardPulseNumerical(Pulse):

    use_lookup = False  # We perform numerical integration in the Integrator

    def __init__(self):
        super(StandardPulseNumerical, self).__init__(
            pulse=one,
            parametrization=identity,
            perform_checks=False
        )


class GaussianPulse(Pulse):
    """ Pulse based on a Gaussian located at loc with variance according to scale. Make sure that loc is near to the
        interval [0,1] or has a high variance. Otherwise, the overlap with the interval [0,1] is too small.
    """

    use_lookup = False  # We perform numerical integration in the Integrator

    def __init__(self, loc: float, scale: float, perform_checks: bool=False):
        self._validate_inputs(loc, scale)
        self._loc = loc
        self._scale = scale
        super(GaussianPulse, self).__init__(
            pulse=self._gaussian_pulse,
            parametrization=self._gaussian_parametrization,
            perform_checks=perform_checks
        )

    def _gaussian_pulse(self, x):
        return scipy.stats.norm.pdf(x, self._loc, self._scale) / (scipy.stats.norm.cdf(1, self._loc, self._scale) - scipy.stats.norm.cdf(0, self._loc, self._scale))

    def _gaussian_parametrization(self, x):
        return (scipy.stats.norm.cdf(x, self._loc, self._scale) - scipy.stats.norm.cdf(0, self._loc, self._scale)) \
               / (scipy.stats.norm.cdf(1, self._loc, self._scale) - scipy.stats.norm.cdf(0, self._loc, self._scale))

    @staticmethod
    def _validate_inputs(loc, scale):
        # Validate type
        valid_types = [int, float, np.float64]
        assert type(scale) in valid_types, f"InputError in GaussianPulse: loc must be float but found {type(loc)}."
        assert type(scale) in valid_types, f"InputError in GaussianPulse: scale must be float but found {type(scale)}."

        # Validate that the denominator used in the further calculation does not evaluate to 0
        denominator = scipy.stats.norm.cdf(1, loc, scale) - scipy.stats.norm.cdf(0, loc, scale)
        assert denominator != 0, \
            "InputError in GaussianPulse: Denominator is zero because of the choice of loc and scale."


""" Helper functions """


def one(x):
    return 1.0


def identity(x: float):
    return x


""" Instances """

standard_pulse = StandardPulse()
standard_pulse_numerical = StandardPulseNumerical()
gaussian_pulse = GaussianPulse(loc=0.5, scale=0.3)

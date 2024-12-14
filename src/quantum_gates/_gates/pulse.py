"""Define pulse shapes and their parametrizations.

Attributes:
    constant_pulse (ConstantPulse): Pulse of constant height which uses an analytical lookup in the integrator.
    constant_pulse_numerical (ConstantPulseNumerical): Pulse of constant height which uses numerical integration.
    gaussian_pulse (GaussianPulse): Gaussian pulse with location = 0.5 and scale = 0.25.

Todo:
    * Add parametrized pulses based on Power Series or Fourier Series.
"""

import numpy as np
import scipy.integrate
import scipy.stats


class Pulse(object):
    """ Parent class for pulses with basic utility.

    Args:
        pulse (callable): Function f: [0,1] -> R>=0: Waveform of the pulse, must integrate up to 1.
        parametrization (callable): Function F: [0,1] -> [0,1]: Parameter integral of the pulse. Monotone with
            F(0) = 0 and F(1) = 1, as well as x <= y implies F(x) <= F(y).
        perform_checks (bool): Tells whether the properties of the pulse and parametrization should be validated.
        use_lookup (bool): Bool whether the pulse is constant. Then one can lookup the integration result in the
            integrator.

    Example:
        .. code:: python

           from quantum_gates.pulses import Pulse

           pulse = lambda x: 1
           parametrization = lambda x: x

           constant_pulse = Pulse(
               pulse=pulse,
               parametrization=parametrization,
               perform_checks=False
               )

    Attributes:
        pulse:              Waveform of the pulse as function, f: [0,1] -> R, f >= 0
        parametrization:    Parameter integral of the waveform, F: [0,1] -> [0,1], F >= 0, monotonically increasing
        use_lookup:         In the Integrator, should a integration result lookup be used. True if pulse is constant
    """

    epsilon = 1e-6
    check_n_points = 10

    def __init__(self, pulse: callable, parametrization: callable, perform_checks: bool=False, use_lookup: bool=False):
        if perform_checks:
            assert self._pulse_is_valid(pulse), "Pulse was not valid"
            assert self._parametrization_is_valid(parametrization), "Parametrization was not valid"
            assert self._are_compatible(pulse, parametrization), "Pulse and parametrization are incompatible. "
        self.pulse = pulse
        self.parametrization = parametrization
        self.use_lookup = use_lookup

    def get_pulse(self):
        """Get the waveform f of the pulse as callable.
        """
        return self.pulse

    def get_parametrization(self):
        """Get the parametrization F of the pulse as callable.
        """
        return self.parametrization

    def _pulse_is_valid(self, pulse: callable) -> bool:
        """Returns whether the pulse is a probability distribution on [0,1].

        Args:
            pulse (callable): The waveform which is to be checked.

        Returns:
            Result of the check as boolean.
        """
        integrates_to_1 = abs(scipy.integrate.quad(pulse, 0, 1)[0] - 1) < self.epsilon
        is_non_negative = all((pulse(x) >= 0) for x in np.linspace(0, 1, self.check_n_points))
        return integrates_to_1 and is_non_negative

    def _parametrization_is_valid(self, parametrization: callable) -> bool:
        """ Returns whether the parametrization is monotone and has valid bounds.

        Args:
            parametrization (callable): The parametrization which is to be checked.

        Returns:
            Result of the check as boolean.
        """
        starts_at_0 = abs(parametrization(0) - 0) < self.epsilon
        stops_at_0 = abs(parametrization(1) - 1) < self.epsilon
        is_monotone = all((parametrization(x + self.epsilon) >= parametrization(x))
                          for x in np.linspace(0, 1-self.epsilon, self.check_n_points))
        return starts_at_0 and stops_at_0 and is_monotone

    def _are_compatible(self, pulse, parametrization) -> bool:
        """ Returns whether the integral of the pulse is the parametrization.

        Args:
            pulse (callable): The waveform which is to be checked.
            parametrization (callable): The parametrization which is to be checked.

        Returns:
            Result of the check as boolean.
        """
        for x in np.linspace(self.epsilon, 1-self.epsilon, self.check_n_points):
            difference = abs(scipy.integrate.quad(pulse, 0, x)[0] - parametrization(x))
            if difference > self.epsilon:
                return False
        return True


class ConstantPulse(Pulse):
    """Constant pulse which uses the lookup in the integrator.
    """

    def __init__(self):
        super().__init__(
            pulse=one,
            parametrization=identity,
            perform_checks=False,
            use_lookup=True
        )


class ConstantPulseNumerical(Pulse):
    """Constant pulse which uses numerical integration.

    Note:
        We can use this class for unit testing the ConstantPulse class.
    """
    def __init__(self):
        super(ConstantPulseNumerical, self).__init__(
            pulse=one,
            parametrization=identity,
            perform_checks=False,
            use_lookup=False
        )


class GaussianPulse(Pulse):
    """ Pulse based on a Gaussian located at loc with variance according to scale.

    Make sure that loc is near to the interval [0,1] or has a high variance. Otherwise, the overlap with the
    interval [0,1] is too small.

    Note:
        The integral over the interval [0,1] of the choosen Gaussian should be larger than 1e-6. This is because the
        shape of the pulse is the shape that the Gaussian has in this interval.

    Example:
        .. code:: python

            from quantum_gates.pulses import GaussianPulse

            loc = 0.5   # Location of the Gaussian
            scale = 0.5 # Standard deviation of the Gaussian

            constant_pulse = GaussianPulse(loc=loc, scale=scale)

    Args:
        loc (float): Location of the pulse on the real axis.
        scale (float): Standard deviation or size of the Gaussian pulse.
        perform_check (bool): Whether the pulse should be verified.
    """

    use_lookup = False  # We perform numerical integration in the Integrator

    def __init__(self, loc: float, scale: float, perform_checks: bool=False):
        self._validate_inputs(loc, scale)
        self._loc = loc
        self._scale = scale
        super(GaussianPulse, self).__init__(
            pulse=self._gaussian_pulse,
            parametrization=self._gaussian_parametrization,
            perform_checks=perform_checks,
            use_lookup=False
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


def one(x):
    """ Always returns 1.0.
    """
    return 1.0


def identity(x: float):
    """ Always returns the input.
    """
    return x


constant_pulse = ConstantPulse()
constant_pulse_numerical = ConstantPulseNumerical()
gaussian_pulse = GaussianPulse(loc=0.5, scale=0.25)

""" Evaluates the integrals coming up in the Noisy gates approach for different pulse waveforms.

Because many of the integrals are evaluated many times with the same parameters, we can apply caching to speed things
up.
"""

import numpy as np
import scipy.integrate

from .pulse import Pulse


class Integrator(object):
    """Calculates the integrals for a specific pulse parametrization.

    Args:
        pulse (Pulse): Object specifying the pulse waveform and parametrization.

    Attributes:
        pulse_parametrization (callable): Function F: [0,1] -> [0,1] representing the parametrization of the pulse.
        use_lookup (bool): Tells whether or not the lookup table of the analytical solution should be used.
    """

    _INTEGRAL_LOOKUP = {
        "sin(theta/a)**2": lambda theta, a: np.sin(theta/a)**2,
        "sin(theta/(2*a))**4": lambda theta, a: np.sin(theta/(2*a))**4,
        "sin(theta/a)*sin(theta/(2*a))**2": lambda theta, a: np.sin(theta/a)*np.sin(theta/(2*a))**2,
        "sin(theta/(2*a))**2": lambda theta, a: np.sin(theta/(2*a))**2,
        "cos(theta/a)**2": lambda theta, a: np.cos(theta/a)**2,
        "sin(theta/a)*cos(theta/a)": lambda theta, a: np.sin(theta/a)*np.cos(theta/a),
        "sin(theta/a)": lambda theta, a: np.sin(theta/a),
        "cos(theta/(2*a))**2": lambda theta, a: np.cos(theta/(2*a))**2
    }
    # For each key (integrand), we calculated the result (parametric integral from 0 to theta) using the parametrization
    # theta(t,t0) = omega(t-t0)/a, corresponding to a square pulse, which is one that has constant magnitude.
    _RESULT_LOOKUP = {
        "sin(theta/a)**2": lambda theta, a: a*(2*theta - np.sin(2*theta))/(4*theta),
        "sin(theta/(2*a))**4": lambda theta, a: a*(6*theta-8*np.sin(theta)+np.sin(2*theta))/(16*theta),
        "sin(theta/a)*sin(theta/(2*a))**2": lambda theta, a: a*((np.sin(theta/2))**4)/theta,
        "sin(theta/(2*a))**2": lambda theta, a: a*(theta - np.sin(theta))/(2 * theta),
        "cos(theta/a)**2": lambda theta, a: a*(2*theta + np.sin(2*theta))/(4*theta),
        "sin(theta/a)*cos(theta/a)": lambda theta, a: a*(np.sin(theta))**2/(2*theta),
        "sin(theta/a)": lambda theta, a: a*(1-np.cos(theta))/theta,
        "cos(theta/(2*a))**2": lambda theta, a: a*(theta + np.sin(theta))/(2*theta)
    }

    def __init__(self, pulse: Pulse):
        self.pulse_parametrization = pulse.get_parametrization()
        self.use_lookup = pulse.use_lookup
        self._cache = dict()

    def integrate(self, integrand: str, theta: float, a: float) -> float:
        """ Evaluates the integrand provided as string from zero to a based on the implicit pulse shape scaled by theta.

        If the pulse (pulse_parametrization) is None, we assume that the pulse height is constant. In this case, we do
        not perform numerical calculation but just lookup the result.

        Args:
            integrand (str): Name of the integrand.
            theta (str): Upper limit of the integration. Total area of the pulse waveform.
            a (str): Scaling parameter.

        Returns:
            Integration result as float.
        """

        # Caching
        if (integrand, theta, a) in self._cache:
            return self._cache[(integrand, theta, a)]

        # Input validation
        assert integrand in self._INTEGRAL_LOOKUP.keys(), "Unknown integrand."
        assert a > 0, f"Require non-vanishing gate time but found a = {a}."

        # Pulse is constant -> We can lookup the analytical result
        if self.use_lookup:
            y = self._analytical_integration(integrand, theta, a)

        # Pulse is variable
        else:
            y = self._numerical_integration(integrand, theta, a)

        # Caching
        self._cache[(integrand, theta, a)] = y

        return y

    def _analytical_integration(self, integrand_str: str, theta: float, a: float) -> float:
        """Lookups up the result of the integration for the case that the parametrization is None.

        Note:
            This method can/should only be used when the pulse height is constant. Otherwise, the result would be wrong.

        Args:
            integrand_str (str): Name of the integrand.
            theta (float): Upper limit of the integration. Total area of the pulse waveform.
            a (float): Scaling parameter.
        """
        integral = self._RESULT_LOOKUP[integrand_str]
        return integral(theta, a)

    def _numerical_integration(self, integrand_name: str, theta: float, a: float) -> float:
        """Looks up the integrand as function and performs numerical integration from 0 to theta.

        Uses the the parametrization specified in the class instance.

        Args:
            integrand_name (str): Name of the integrand.
            theta (float): Upper limit of the integration. Total area of the pulse waveform.
            a (float): Scaling parameter.

        Returns:
            Result of the integration as float.
        """
        integrand = self._INTEGRAL_LOOKUP[integrand_name]

        # The parametrization is a monotone function with param(t=0) == 0 and param(t=1) == 1.
        param = self.pulse_parametrization

        # We scale this parametrization such that scaled_param(t=0) == 0 and scaled_param(t=1) == theta.
        scaled_param = lambda t: param(t) * theta

        # We parametrize the integrand and integrate it from 0 to a. Integral should go from 0 to a.
        integrand_p = lambda t: integrand(scaled_param(t), a)
        y, abserr = scipy.integrate.quad(integrand_p, 0, a)

        return y

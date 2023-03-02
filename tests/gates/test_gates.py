import pytest
import time
import numpy as np
import pickle

from src.quantum_gates._legacy.gates import X, SX, CNOT, CNOT_inv
from src.quantum_gates.gates import standard_gates, noise_free_gates
from src.quantum_gates._gates.gates import numerical_gates, almost_noise_free_gates
from src.quantum_gates.gates import Gates
from src.quantum_gates.pulses import GaussianPulse
import tests.helpers.device_parameters as helper_dev_param


""" Constants """


# Gates
# Original gates before refactoring
original_gates_list = [X, SX, CNOT, CNOT_inv]

# Gates without numerical integration after refactoring
refactored_gates_list = [
    standard_gates.X, standard_gates.SX, standard_gates.CNOT, standard_gates.CNOT_inv
]

# Gates with numerical integration after refactoring
numerical_gates_list = [
    numerical_gates.X, numerical_gates.SX, numerical_gates.CNOT, numerical_gates.CNOT_inv
]

# Args
single_qubit_args = {
    "phi": np.pi/2,
    "p": helper_dev_param.p[0],
    "T1": helper_dev_param.T1[0],
    "T2": helper_dev_param.T2[0]
}
two_qubit_args = {
    "phi_ctr": np.pi/2,
    "phi_trg": np.pi/2,
    "t_cnot": helper_dev_param.t_cnot[0][1],
    "p_cnot": helper_dev_param.p_cnot[0][1],
    "p_single_ctr": helper_dev_param.p[0],
    "p_single_trg": helper_dev_param.p[1],
    "T1_ctr": helper_dev_param.T1[0],
    "T2_ctr": helper_dev_param.T2[0],
    "T1_trg": helper_dev_param.T1[1],
    "T2_trg": helper_dev_param.T2[1]
}
args = [single_qubit_args, single_qubit_args, two_qubit_args, two_qubit_args]


""" Helper functions """


def _speed_test_helper(old_gate, new_gate, arg, n=100, factor=1.05):
    """
    Times the old gate against the new gate for n creations. If the new time is not more than factor of the old time,
    the test is passed.

    Note: We switch between the two in each iteration. This way, external influences (CPU load, memory) are averaged
    away.
    """

    # Test old gates
    time_old = 0
    time_new = 0

    for i in range(n):
        # Test old gate
        time_old -= time.time()
        old_gate(**arg)
        time_old += time.time()

        # Test new gate
        time_new -= time.time()
        new_gate(**arg)
        time_new += time.time()

    print(f"Time old {time_old} s vs time new {time_new} s.")
    assert(time_new < factor * time_old), f"Found that old version took {time_old} s < {time_new} * {factor}."


def _almost_equal(m1, m2, nqubit: int, abs_tol: float=1e-9):
    """ Check if m1 and m2 are close. """
    diff = m1 - m2
    return all((abs(diff[i,j]) < abs_tol) for i in range(2**nqubit) for j in range(2**nqubit))


"""  Tests """


@pytest.mark.parametrize(
    "old_gate,new_gate,arg",
    [(old_g, new_g, arg) for old_g, new_g, arg in zip(original_gates_list, refactored_gates_list, args)]
)
def test_gates_speed_standard_gates(old_gate, new_gate, arg):
    """ Check that creating the new standard gates is not slower than creating the initially used gates.
    """
    _speed_test_helper(old_gate, new_gate, arg, n=1000, factor=1.2)


@pytest.mark.parametrize(
    "old_gate,new_gate,arg",
    [(old_g, new_g, arg) for old_g, new_g, arg in zip(original_gates_list, numerical_gates_list, args)]
)
def test_gates_speed_numerical_gates(old_gate, new_gate, arg):
    """ Check that creating the new numerical gates is not more than 50% slower than creating the initially used gates.
    """
    _speed_test_helper(old_gate, new_gate, arg, n=1000, factor=1.5)


def test_gates_noiseless_relaxation():
    # Dt is the idle time, T1 qubit's amplitude damping time (ns), T2 qubit's dephasing time (ns)
    tg = 35 * 10**(-9)
    args = {"Dt": tg, "T1": 1e12, "T2": 1e12}
    res_exp = noise_free_gates.relaxation(**args)    # Harcoded noiseless
    res = numerical_gates.relaxation(**args)         # Simulated noiseless
    assert _almost_equal(res_exp, res, nqubit=1, abs_tol=1e-9), f"Found almost noiseless relaxation {res} instead of {res_exp}."


def test_gates_noiseless_bitflip():
    # Dt is the idle time, p readout error
    tg = 35 * 10**(-9)
    args = {"Dt": tg, "p": 0.0}
    res_exp = noise_free_gates.bitflip(**args)    # Harcoded noiseless
    res = numerical_gates.bitflip(**args)         # Simulated noiseless
    assert _almost_equal(res_exp, res, nqubit=1, abs_tol=1e-9), \
        f"Found almost noiseless bitflip {res} instead of {res_exp}."


def test_gates_noiseless_depolarizing():
    # Dt is the idle time, p readout error
    tg = 35 * 10**(-9)
    args = {"Dt": tg, "p": 0.0}
    res_exp = noise_free_gates.depolarizing(**args)    # Harcoded noiseless
    res = numerical_gates.depolarizing(**args)         # Simulated noiseless
    assert _almost_equal(res_exp, res, nqubit=1, abs_tol=1e-9), \
        f"Found almost noiseless depolarizing {res} instead of {res_exp}."


@pytest.mark.parametrize(
    "phi,theta",
    [(phi, theta) for phi in np.linspace(0, np.pi, 4) for theta in np.linspace(0, np.pi, 4)]
)
def test_gates_noiseless_single_qubit_gate(theta: float, phi: float):
    # theta: angle by which the state is rotated (double)
    # phi: phase of the drive defining axis of rotation on the Bloch sphere (double)
    # p: single-qubit depolarizing error probability (double)
    # T1: qubit's amplitude damping time in ns (double)
    # T2: qubit's dephasing time in ns (double)
    args = {"theta": theta, "phi": phi, "p": 0.0, "T1": 1e12, "T2": 1e12}
    res_exp = noise_free_gates.single_qubit_gate(**args)    # Hardcoded noiseless
    res = numerical_gates.single_qubit_gate(**args)         # Simulated noiseless
    assert _almost_equal(res_exp, res, nqubit=1, abs_tol=1e-9), \
        f"Found almost noiseless single qubit gate {res} instead of {res_exp}."


@pytest.mark.parametrize("phi", np.linspace(0, np.pi, 10))
def test_gates_noiseless_x(phi: float):
    # phi: phase of the drive defining axis of rotation on the Bloch sphere (double)
    # p: single-qubit depolarizing error probability (double)
    # T1: qubit's amplitude damping time in ns (double)
    # T2: qubit's dephasing time in ns (double)
    args = {"phi": phi, "p": 0.0, "T1": 1e12, "T2": 1e12}
    res_exp = noise_free_gates.X(**args)    # Hardcoded noiseless
    res = numerical_gates.X(**args)         # Simulated noiseless
    assert _almost_equal(res_exp, res, nqubit=1, abs_tol=1e-9), \
        f"Found almost noiseless X {res} instead of {res_exp}."


@pytest.mark.parametrize("phi", np.linspace(0, np.pi, 10))
def test_gates_noiseless_sx(phi):
    # For parameters see test_noiseless_x()
    args = {"phi": phi, "p": 0.0, "T1": 1e12, "T2": 1e12}
    res_exp = noise_free_gates.SX(**args)    # Hardcoded noiseless
    res = numerical_gates.SX(**args)         # Simulated noiseless
    assert _almost_equal(res_exp, res, nqubit=1, abs_tol=1e-9), \
        f"Found almost noiseless SX {res} instead of {res_exp}."


@pytest.mark.parametrize(
    "phi,theta",
    [(phi, theta) for phi in np.linspace(np.pi/2, np.pi, 4) for theta in np.linspace(np.pi/2, np.pi, 4)]
)
def test_gates_noiseless_cr(phi: float, theta: float):
    # theta: angle of rotation on the Bloch sphere (double)
    # phi: phase of the drive defining axis of rotation on the Bloch sphere (double)
    # t_cr: CR gate time in ns (double)
    # p_cr: CR depolarizing error probability (double)
    # T1_ctr: control qubit's amplitude damping time in ns (double)
    # T2_ctr: control qubit's dephasing time in ns (double)
    # T1_trg: target qubit's amplitude damping time in ns (double)
    # T2_trg: target qubit's dephasing time in ns (double)
    args = {
        "theta": theta,
        "phi": phi,
        "t_cr": 3.3422e-07,
        "p_cr": 0.0,
        "T1_ctr": 1e12,
        "T2_ctr": 1e12,
        "T1_trg": 1e12,
        "T2_trg": 1e12
    }
    res_exp = noise_free_gates.CR(**args)    # Hardcoded noiseless
    res = numerical_gates.CR(**args)         # Simulated noiseless
    assert _almost_equal(res_exp, res, nqubit=2, abs_tol=1e-6), \
        f"Found almost noiseless CR {res} instead of {res_exp}."


@pytest.mark.parametrize(
    "phi_ctr,phi_trg",
    [(phi1, phi2) for phi1 in np.linspace(0, np.pi, 4) for phi2 in np.linspace(0, np.pi, 4)]
)
def test_gates_noiseless_cnot(phi_ctr: float, phi_trg: float):
    # phi_ctr: control qubit phase of the drive defining axis of rotation on the Bloch sphere (double)
    # phi_trg: target qubit phase of the drive defining axis of rotation on the Bloch sphere (double)
    # t_cnot: CNOT gate time in ns (double)
    # p_cnot: CNOT depolarizing error probability (double)
    # p_single_ctr: control qubit depolarizing error probability (double)
    # p_single_trg: target qubit depolarizing error probability (double)
    # T1_ctr: control qubit's amplitude damping time in ns (double)
    # T2_ctr: control qubit's dephasing time in ns (double)
    # T1_trg: target qubit's amplitude damping time in ns (double)
    # T2_trg: target qubit's dephasing time in ns (double)
    args = {
        "phi_ctr": phi_ctr,
        "phi_trg": phi_trg,
        "t_cnot": 3.3422e-07,
        "p_cnot": 0.0,
        "p_single_ctr": 0.0,
        "p_single_trg": 0.0,
        "T1_ctr": 1e12,
        "T2_ctr": 1e12,
        "T1_trg": 1e12,
        "T2_trg": 1e12
    }
    res_exp = noise_free_gates.CNOT(**args)    # Hardcoded noiseless
    res = numerical_gates.CNOT(**args)         # Simulated noiseless
    assert _almost_equal(res_exp, res, nqubit=2, abs_tol=1e-6), \
        f"Found almost noiseless CNOT {res} instead of {res_exp}."


@pytest.mark.parametrize(
    "phi_ctr,phi_trg",
    [(phi1, phi2) for phi1 in np.linspace(0, np.pi, 4) for phi2 in np.linspace(0, np.pi, 4)]
)
def test_gates_noiseless_cnot_inv(phi_ctr: float, phi_trg: float):
    # Parameters see test_noiseless_cnot()
    args = {
        "phi_ctr": phi_ctr,
        "phi_trg": phi_trg,
        "t_cnot": 3.3422e-07,
        "p_cnot": 0.0,
        "p_single_ctr": 0.0,
        "p_single_trg": 0.0,
        "T1_ctr": 1e12,
        "T2_ctr": 1e12,
        "T1_trg": 1e12,
        "T2_trg": 1e12
    }
    res_exp = noise_free_gates.CNOT_inv(**args)    # Hardcoded noiseless
    res = numerical_gates.CNOT_inv(**args)         # Simulated noiseless
    assert _almost_equal(res_exp, res, nqubit=2, abs_tol=1e-9), \
        f"Found almost noiseless CNOT {res} instead of {res_exp}."


def test_gates_pickle_standard_gates():
    pickle.dumps(standard_gates)


def test_gates_pickle_numerical_gates():
    pickle.dumps(numerical_gates)


def test_gates_pickle_noise_free_gates():
    pickle.dumps(noise_free_gates)


def test_gates_pickle_almost_noise_free_gates():
    pickle.dumps(almost_noise_free_gates)


def test_gates_pickle_gates_with_gaussian_pulse():
    pulse = GaussianPulse(loc=0.5, scale=0.3)
    gates = Gates(pulse=pulse)
    pickle.dumps(gates)

import pytest
import numpy as np
import time

from src.quantum_gates.backends import EfficientBackend
from src.quantum_gates._simulation.backend import StandardBackend, BackendForOnes
import tests.helpers.gates as helper_gates
import tests.helpers.functions as helper_functions


backends = [StandardBackend, EfficientBackend, BackendForOnes]
efficient_backend = [EfficientBackend, BackendForOnes]


""" Tests """


@pytest.mark.parametrize("nqubit,Backend", [(nqubit, Backend) for nqubit in [2, 4, 6] for Backend in backends])
def test_backend_init(nqubit, Backend):
    tb = Backend(nqubit=nqubit)


@pytest.mark.parametrize("nqubits,Backend", [(nqubit, Backend) for nqubit in [2, 3, 4, 6, 8, 10, 12] for Backend in backends])
def test_backend_eye(nqubits, Backend):
    tb = Backend(nqubit=nqubits)
    mp = [np.eye(2) for i in range(nqubits)]
    mp_list = [mp]
    psi0 = np.random.rand(2**nqubits)
    psi1 = tb.statevector(mp_list, psi0)

    assert helper_functions.vector_almost_equal(psi0, psi1, nqubits), \
        f"Assumed that applying identities will lead to a trivial circuit, but found {psi1} != 1 {psi0}."


@pytest.mark.parametrize("nqubits,Backend", [(nqubit, Backend) for nqubit in [2, 3, 4, 6, 8, 10, 12] for Backend in backends])
def test_backend_x(nqubits, Backend):
    tb = Backend(nqubit=nqubits)
    mp = [np.fliplr(np.eye(2)) for i in range(nqubits)]
    mp_list = [mp]
    psi0 = np.zeros(2**nqubits)
    psi0[0] = 1
    psi1 = tb.statevector(mp_list, psi0)
    psi1_exp = np.zeros(2**nqubits)
    psi1_exp[-1] = 1
    assert helper_functions.vector_almost_equal(psi1, psi1_exp, nqubits), \
        f"Assumed that applying (X ... X) on |0..0> produces |1...1>, but found {psi1} != (X...X) {psi0}."


@pytest.mark.parametrize("nqubits,Backend", [(nqubit, Backend) for nqubit in [2, 3, 4, 6, 8, 10, 12] for Backend in backends])
def test_backend_result_x_and_many_cnot(nqubits, Backend):

    # Create circuit that maps |0..0> to |1...1> with one X and many CNOTs.
    mp_list = [[helper_gates.X] + [helper_gates.identity for i in range(nqubits - 1)]]
    for j in range(nqubits - 1):
        mp_list.append([helper_gates.identity for i in range(j)] + [helper_gates.CNOT, 1] + [helper_gates.identity for i in range(nqubits - 2 - j)])

    tb = Backend(nqubit=nqubits)

    # Check that it does the correct mapping
    psi0 = np.zeros(2**nqubits)
    psi0[0] = 1.0

    psi1 = tb.statevector(mp_list, psi0)
    psi1_exp = np.zeros(2**nqubits)
    psi1_exp[-1] = 1

    assert all((psi1[i] == psi1_exp[i] for i in range(2**nqubits))), f"Expected psi1 {psi1_exp} but found {psi1}."


@pytest.mark.parametrize(
    "nqubits,steps,gate",
    [(n, s, gate) for n in [2, 3, 4, 6, 10] for s in [1, 10] for gate in helper_gates.single_qubit_gate_list]
)
def test_backends_get_same_result_with_single_qubit_gates(nqubits: int, steps: int, gate: np.array):

    psi0 = np.random.rand(2**nqubits)

    # Setup mp list
    mp = [gate for i in range(nqubits)]
    mp_list = [mp for step in range(steps)]

    # Setup backends
    one_tb = BackendForOnes(nqubits)
    efficient_tb = EfficientBackend(nqubits)
    standard_tb = StandardBackend(nqubits)

    # Compute
    psi_one = one_tb.statevector(mp_list, psi0)
    psi_efficient = efficient_tb.statevector(mp_list, psi0)
    psi_standard = standard_tb.statevector(mp_list, psi0)

    # Evaluate
    assert helper_functions.vector_almost_equal(psi_efficient, psi_standard, nqubits), \
        f"The efficient backend did not generate the same result as the standard backend. Found {psi_efficient} and {psi_standard}."

    assert helper_functions.vector_almost_equal(psi_one, psi_standard, nqubits), \
        f"The one backend did not generate the same result as the standard backend. Found {psi_one} and {psi_standard}."


@pytest.mark.parametrize("nqubits, steps", [(n, s) for n in [2, 3, 4, 6, 8, 9, 10] for s in [5]])
def test_backends_get_same_result_with_random_matrix_products(nqubits, steps):
    mp_list = helper_functions.generate_random_matrix_products(nqubits, steps=steps)

    # Backends
    one_tb = BackendForOnes(nqubits)
    efficient_tb = EfficientBackend(nqubits)
    trivial_tb = StandardBackend(nqubits)

    # Compute
    psi0 = np.zeros(2**nqubits)
    psi0[0] = 1.0
    one_psi1 = one_tb.statevector(mp_list, psi0)
    efficient_psi1 = efficient_tb.statevector(mp_list, psi0)
    trivial_psi1 = trivial_tb.statevector(mp_list, psi0)

    # Evaluate
    print("one_psi1", one_psi1)
    print("efficient_psi1", efficient_psi1)
    print("trivial_psi1", trivial_psi1)

    assert helper_functions.vector_almost_equal(efficient_psi1, trivial_psi1, nqubits), \
        "The efficient backend did not generate the same result as the standard backend."

    assert helper_gates.vector_almost_equal(one_psi1, trivial_psi1, nqubits), \
        "The one backend did not generate the same result as the standard backend."


def test_backends_hard_against_each_other():
    # Setup
    nqubit = 2
    mp1 = [helper_gates.identity, helper_gates.X]
    mp2 = [helper_gates.identity, helper_gates.Z]
    mp_list = [mp1, mp2]

    # Apply
    one_tb = BackendForOnes(nqubit)
    efficient_tb = EfficientBackend(nqubit)
    trivial_tb = StandardBackend(nqubit)

    one_psi = one_tb.statevector(mp_list, np.array([1.0, 0.0, 0.0, 0.0]))
    efficient_psi = efficient_tb.statevector(mp_list, np.array([1.0, 0.0, 0.0, 0.0]))
    trivial_psi = trivial_tb.statevector(mp_list, np.array([1.0, 0.0, 0.0, 0.0]))

    # Evaluate
    print("one_psi", one_psi)
    print("efficient_psi", efficient_psi)
    print("trivial_psi", trivial_psi)

    assert helper_functions.vector_almost_equal(efficient_psi, trivial_psi, nqubit), \
        "The efficient backend did not generate the same result as the standard backend."

    assert helper_functions.vector_almost_equal(one_psi, trivial_psi, nqubit), \
        "The one backend did not generate the same result as the standard backend."


@pytest.mark.parametrize("nqubits, steps", [(n,s) for n in [7, 8, 10] for s in [5, 100]])
def test_backend_is_faster_than_standard_backend(nqubits: int, steps: int):
    mp_list = helper_functions.generate_random_matrix_products(nqubits, steps=steps)

    # Time Backend
    start = time.time()
    tb = EfficientBackend(nqubits)
    psi = np.zeros(2**nqubits)
    psi[0] = 1
    tb.statevector(mp_list, psi)
    time_tb = time.time() - start

    # Time StandardBackend
    start = time.time()
    tb = StandardBackend(nqubits)
    psi = np.zeros(2**nqubits)
    psi[0] = 1
    tb.statevector(mp_list, psi)
    time_triv_tb = time.time() - start

    assert time_triv_tb > time_tb, \
        f"Found that the trivial tb uses less time ({time_triv_tb} s) than the tb ({time_tb} s)."


@pytest.mark.parametrize("nqubits, steps", [(n,s) for n in range(6, 18) for s in [500]])
def test_one_backend_is_faster_than_efficient_backend(nqubits: int, steps: int):
    mp_list = helper_functions.generate_random_matrix_products(nqubits, steps=steps, prob_cnot=1/nqubits, many_identites=True)

    # Time EfficientBackend
    start = time.time()
    tb = EfficientBackend(nqubits)
    psi = np.zeros(2**nqubits)
    psi[0] = 1
    psi_eff = tb.statevector(mp_list, psi)
    time_eff = time.time() - start

    # Time BackendForOnes
    start = time.time()
    tb = BackendForOnes(nqubits)
    psi = np.zeros(2**nqubits)
    psi[0] = 1
    psi_one = tb.statevector(mp_list, psi)
    time_one = time.time() - start

    # Check result is the same
    assert helper_functions.vector_almost_equal(psi_one, psi_eff, nqubits), \
        "The one backend did not generate the same result as the efficient backend."

    # Check runtime
    assert time_one < time_eff, \
        f"Found that the one backend uses more time ({time_one:.4f} s) than the efficient backend tb ({time_eff:.4f} s)."


@pytest.mark.skip(reason="We fail this test on purpose to get the time and print statements.")
@pytest.mark.parametrize(
    "nqubits, steps, prob_cnot",
    [(n,s, prob_cnot) for n in [7, 8, 9, 10, 11, 12, 13, 14] for s in [100] for prob_cnot in [0.0, 0.5]]
)
def test_backend_performance_just_fail_and_print(nqubits: int, steps: int, prob_cnot):
    mp_list = helper_functions.generate_random_matrix_products(nqubits, steps=steps, prob_cnot=prob_cnot)

    start = time.time()
    tb = EfficientBackend(nqubits)
    psi = np.zeros(2**nqubits)
    psi[0] = 1
    tb.statevector(mp_list, psi)
    time_tb = time.time() - start

    assert False, f"Found that the tb needs {time_tb} for {nqubits} nqubits and {steps} steps."

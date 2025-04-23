import pytest
import numpy as np
import os
from dotenv import load_dotenv

from qiskit import QuantumCircuit, transpile
from quantum_gates.qiskit_provider import NoisyGatesProvider
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

load_dotenv()
IBM_TOKEN = os.getenv("IBM_TOKEN", "")
assert IBM_TOKEN != "", "Expected to find 'IBM_TOKEN' in .env file, but it was not found."


@pytest.mark.integration
def test_bell_state_distribution():
    """Check that the Bell state distribution from NoisyGates ibm_brisbane
    has ~40% or more in 00, 11 and less than 10% in 01, 10."""
    # Arrange
    ng_provider = NoisyGatesProvider()
    ng_backend = ng_provider.get_ibm_backend('fake_brisbane')
    shots = 1000

    # Build a simple 2-qubit Bell circuit
    n_qubit = 2
    circ = QuantumCircuit(n_qubit, n_qubit)
    circ.h(0)
    circ.cx(0, 1)
    circ.measure(range(n_qubit), range(n_qubit))

    # Act
    transpiled_ng = ng_backend.ng_transpile(
        circ, init_layout=[0, 1], seed=42
    )
    job_ng = ng_backend.run(transpiled_ng, shots=shots)
    counts_ng = job_ng.result().get_counts()
    print("Raw provider counts:", counts_ng)

    # Normalize if necessary
    freq_ng = {}
    total = sum(counts_ng.values())
    for state in ["00", "01", "10", "11"]:
        freq_ng[state] = counts_ng.get(state, 0) / total

    # Assert that the distribution makes sense
    assert freq_ng["00"] >= 0.40, f"Unexpectedly low freq(00) {freq_ng['00']:.3f}"
    assert freq_ng["11"] >= 0.40, f"Unexpectedly low freq(11) {freq_ng['11']:.3f}"
    assert freq_ng["01"] <= 0.10, f"Unexpectedly high freq(01) {freq_ng['01']:.3f}"
    assert freq_ng["10"] <= 0.10, f"Unexpectedly high freq(10) {freq_ng['10']:.3f}"

    # Assert that the output is normalized
    assert abs(total - 1.0) < 1e-5, f"Expected output to be normalized but found sum={total}."

@pytest.mark.integration
def test_ng_provider_vs_standard_qiskit():
    """Compare the Noisy Gates backend with the standard Qiskit provider
    for a Bell state circuit. Frequencies should be close."""
    # Arrange
    provider = NoisyGatesProvider()
    backend = provider.get_ibm_backend('fake_brisbane')

    # IBM fake backend
    fake_provider = FakeProviderForBackendV2()
    fake_backend = fake_provider.backend('fake_brisbane')

    shots = 1000
    n_qubit = 2

    # Create Bell state circuit
    circ = QuantumCircuit(n_qubit, n_qubit)
    circ.h(0)
    circ.cx(0, 1)
    circ.barrier()
    circ.measure(range(n_qubit), range(n_qubit))

    # Transpile and run with NoisyGates backend
    transpiled_ng = backend.ng_transpile(circ, init_layout=[0, 1], seed=10)
    job_ng = backend.run(transpiled_ng, shots=shots)
    count = job_ng.result().get_counts()

    # Transpile and run with fake IBM backend
    transpiled_ibm = transpile(
        circ,
        backend=fake_backend,
        initial_layout=[0, 1],
        seed_transpiler=10
    )
    ibm_job = fake_backend.run(transpiled_ibm, shots=shots)
    ibm_count = ibm_job.result().get_counts()

    # Compute absolute differences
    c_00 = np.abs(ibm_count['00']/shots - count['00'])
    c_01 = np.abs(ibm_count['01']/shots - count['01'])
    c_10 = np.abs(ibm_count['10']/shots - count['10'])
    c_11 = np.abs(ibm_count['11']/shots - count['11'])

    # Assert that the distribution is close
    threshold = 0.1  # maximum acceptable difference

    assert c_00 <= threshold, f"Mismatch on 00: diff={c_00:.3f}"
    assert c_01 <= threshold, f"Mismatch on 01: diff={c_01:.3f}"
    assert c_10 <= threshold, f"Mismatch on 10: diff={c_10:.3f}"
    assert c_11 <= threshold, f"Mismatch on 11: diff={c_11:.3f}"





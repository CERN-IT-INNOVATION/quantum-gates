import pytest
import os
from dotenv import load_dotenv

from qiskit import QuantumCircuit
from quantum_gates.qiskit_provider import NoisyGatesProvider

load_dotenv()
IBM_TOKEN = os.getenv("IBM_TOKEN", "")
assert IBM_TOKEN != "", "Expected to find 'IBM_TOKEN' in .env file, but it was not found."


@pytest.mark.integration
def test_bell_state_distribution():
    """Check that the Bell state distribution from NoisyGates ibm_brisbane
    has ~40% or more in 00, 11 and less than 10% in 01, 10."""
    # Arrange
    ng_provider = NoisyGatesProvider(token=IBM_TOKEN)
    ng_backend = ng_provider.get_ibm_backend('ibm_brisbane')
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

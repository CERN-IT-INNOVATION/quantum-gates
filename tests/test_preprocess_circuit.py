import pytest
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeKyiv
from quantum_gates._simulation.simulator import MrAndersonSimulator
from qiskit.circuit.instruction import Instruction  # Ensure compatibility


@pytest.fixture
def simulator():
    return MrAndersonSimulator()


@pytest.mark.filterwarnings("ignore:The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.duration`` is deprecated")
@pytest.mark.filterwarnings("ignore:The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.unit`` is deprecated")
def test_preprocess_circuit_with_simple_circuit(simulator):
    """Test _preprocess_circuit with a basic 2-qubit circuit."""
    # Arrange
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    qubits_layout = [0, 1]
    nqubit = 2

    
    n_rz, swap_detector, data = simulator._preprocess_circuit(
        t_qiskit_circ=circuit,
        qubits_layout=qubits_layout,
        nqubit=nqubit
    )

    
    assert n_rz == 0, "Expected no RZ gates in the circuit."
    assert isinstance(data, list) and len(data) > 0, "Expected non-empty data as a list."
    for instruction in data:
        assert hasattr(instruction, "operation"), "Expected instruction to have 'operation' attribute."
        assert hasattr(instruction, "qubits"), "Expected instruction to have 'qubits' attribute."
        assert isinstance(instruction.operation, Instruction), f"Operation should be of type 'Instruction', got {type(instruction.operation)}."


def test_preprocess_circuit_handles_complex_circuit(simulator):
    """Test _preprocess_circuit with a more complex circuit."""
    # Arrange
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.rz(0.5, 1)
    circuit.measure([0, 1, 2], [0, 1, 2])

    qubits_layout = [0, 1, 2]
    nqubit = 3

    
    n_rz, swap_detector, data = simulator._preprocess_circuit(
        t_qiskit_circ=circuit,
        qubits_layout=qubits_layout,
        nqubit=nqubit
    )

    
    assert n_rz == 1, "Expected 1 RZ gate in the circuit."
    assert isinstance(swap_detector, list), "Swap detector should be a list."
    assert len(data) > 0, "Expected non-empty data."


def test_preprocess_circuit_does_not_modify_original_circuit(simulator):
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    qubits_layout = [0, 1]
    nqubit = 2

    # Copy original circuit data
    original_data = list(circuit.data)

    
    simulator._preprocess_circuit(
        t_qiskit_circ=circuit,
        qubits_layout=qubits_layout,
        nqubit=nqubit
    )

    
    assert circuit.data == original_data, "Input circuit data should remain unchanged."

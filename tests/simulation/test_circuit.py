import pytest
import numpy as np

from src.quantum_gates.circuits import EfficientCircuit
from src.quantum_gates._simulation.circuit import Circuit, AlternativeCircuit, StandardCircuit, BinaryCircuit
from src.quantum_gates.backends import EfficientBackend, BinaryBackend
from src.quantum_gates._simulation.backend import StandardBackend
from src.quantum_gates.gates import standard_gates
import tests.helpers.gates as helper_gates
import tests.helpers.functions as helper_functions
import tests.helpers.device_parameters as helper_device_param


# Helper functions
convert_to_prob = lambda x: np.abs(x)**2
convertToProb = np.vectorize(convert_to_prob)

# Backend
backend_classes = [StandardBackend, EfficientBackend]

# Circuits
circuit_classes = [Circuit, StandardCircuit, EfficientCircuit, BinaryCircuit]


@pytest.mark.parametrize("CircuitClass", circuit_classes)
def test_circuit_init(CircuitClass):
    CircuitClass(nqubit=2, depth=2, gates=standard_gates)


@pytest.mark.parametrize("CircuitClass", circuit_classes)
def test_circuit_apply(CircuitClass):
    circ = CircuitClass(nqubit=2, depth=1, gates=standard_gates)
    if isinstance(circ, BinaryCircuit):
        circ.update_circuit_list(gate=helper_gates.X, qubit = [0])
        circ.update_circuit_list(gate=helper_gates.X, qubit = [1])
        psi1 = circ.statevector(np.array([1, 0, 0, 0]), 0, [0,1])
    else:
        circ.apply(gate=helper_gates.X, i=0)
        circ.apply(gate=helper_gates.X, i=1)
        psi1 = circ.statevector(np.array([1, 0, 0, 0]))
    expected_psi = np.array([0, 0, 0, 1])
    assert all(psi1[i] == expected_psi[i] for i in range(4))


@pytest.mark.parametrize("BackendClass", backend_classes)
def test_alternative_circuit_apply(BackendClass):
    circ = AlternativeCircuit(nqubit=2, gates=standard_gates, BackendClass=BackendClass)
    circ.apply(gate=helper_gates.X, i=0)
    circ.apply(gate=helper_gates.X, i=1)
    psi1 = circ.statevector(np.array([1, 0, 0, 0]))
    expected_psi = np.array([0, 0, 0, 1])
    assert all((psi1[i] == expected_psi[i] for i in range(4)))


@pytest.mark.parametrize("CircuitClass", circuit_classes)
def test_circuit_apply_None_raises_ValueError(CircuitClass):
    with pytest.raises(ValueError):
        circ = CircuitClass(nqubit=2, depth=1, gates=standard_gates)
        if isinstance(circ, BinaryCircuit):
            circ.update_circuit_list(gate=None, qubit = [0])
        else:
            circ.apply(gate=None, i=0)


@pytest.mark.parametrize("CircuitClass", circuit_classes)
def test_circuit_apply_2qubit_gate_raises_ValueError(CircuitClass):
    with pytest.raises(ValueError):
        circ = CircuitClass(nqubit=2, depth=1, gates=standard_gates)
        if isinstance(circ, BinaryCircuit):
            circ.update_circuit_list(gate=helper_gates.CNOT, qubit = [0])
        else:
            circ.apply(gate=helper_gates.CNOT, i=0) # Use apply for 2 qubit (CNOT) gate.


@pytest.mark.parametrize("BackendClass", backend_classes)
def test_alternative_no_gates_applied(BackendClass):
    circ = AlternativeCircuit(nqubit=2, gates=standard_gates, BackendClass=BackendClass)
    psi1 = circ.statevector(np.array([1, 0, 0, 0]))
    expected_psi = np.array([1, 0, 0, 0])
    assert helper_functions.vector_almost_equal(psi1, expected_psi, 2), "Did not output the input when no gates are applied."


@pytest.mark.parametrize("BackendClass", backend_classes)
def test_alternative_apply_cnot_two_qubits(BackendClass):
    circ = AlternativeCircuit(nqubit=2, gates=standard_gates, BackendClass=BackendClass)
    circ.apply(gate=helper_gates.X, i=0)
    circ.I(i=1)
    circ.CNOT(i=0, k=1, **helper_device_param.INT_args)
    psi1 = circ.statevector(np.array([1, 0, 0, 0]))
    prop_found = convertToProb(psi1)
    prop_exp = np.array([0, 0, 0, 1**2])
    assert all(abs(abs(prop_found[i]) - abs(prop_exp[i])) < 0.2 for i in range(4)), \
        f"Propagation failed. Expected prop {prop_exp} vs. found {prop_found}."


@pytest.mark.parametrize("BackendClass", backend_classes)
def test_alternative_apply_cnot_three_qubits(BackendClass):
    circ = AlternativeCircuit(nqubit=3, gates=standard_gates, BackendClass=BackendClass)
    circ.apply(gate=helper_gates.X, i=0)
    circ.I(i=1)
    circ.apply(gate=helper_gates.X, i=2)
    circ.CNOT(i=0, k=1, **helper_device_param.INT_args)
    circ.I(i=2)
    psi1 = circ.statevector(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    prop_found = convertToProb(psi1)
    prop_exp = np.array([0, 0, 0, 0, 0, 0, 0, 1**2])
    assert all(abs(abs(prop_found[i]) - abs(prop_exp[i])) < 0.2 for i in range(8)), \
        f"Propagation failed. Expected prop {prop_exp} vs. found {prop_found}."
    
import uuid
import time
import numpy as np

from qiskit.providers import BackendV2 as Backend
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2 as FakeBackend
from qiskit.providers.options import Options
from qiskit.transpiler import Target
from qiskit import QuantumCircuit, transpile
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.circuit.library import (
    HGate, XGate, SXGate, RZGate, RXGate, RYGate, UGate
)
from qiskit.circuit.library import CXGate, ECRGate, RZXGate
from qiskit.circuit import Measure, Delay, Barrier

from .._simulation.simulator import MrAndersonSimulator
from .._gates.pulse import ConstantPulse, ConstantPulseNumerical
from .._gates.gates import Gates
from .._simulation.circuit import EfficientCircuit, AlternativeCircuit, BinaryCircuit
from .._utility.device_parameters import DeviceParameters
from .._utility.simulations_utility import fix_counts

from .ng_job import NoisyGatesJob


# Pulse
constant_pulse = ConstantPulse()
constant_pulse_numerical = ConstantPulseNumerical()


# Gates with constant Pulse
standard_gates = Gates(pulse=constant_pulse)
numerical_gates = Gates(pulse=constant_pulse_numerical)


class NoisyGatesBackend(Backend):
    """
    Qiskit backend for the Noisy Gates model. It works as a FakeBackend for a real or fake device but using the Noisy Gates formalism.
    """

    def __init__(self, device = None,  **fields):
        """""Initialize the backend.

        Args:
            device : IBM device used to import the information for the target and the parameters. It can be either a real or a fake backend
                        depending if a ibm token is provided
                        
        Attributes:
            gates_ng : An object of the class Gates of the Noisy Gates model
            circuit_class_ng : The class of circuit model from the Noisy Gates model
            parallel_ng [Bool] : If true run in parallel the simulation, otherwise not
            simulator : MrAndersonSimulator object that represent the simulator of the Noisy gates model
        """""
        if device is not None and hasattr(device, "configuration"):
            n_qubits = device.configuration().num_qubits
        else:
            n_qubits = fields.pop("num_qubits", 2)
        super().__init__(
            name="Giotto",
            description="A python noise simulator for quantum experiments that implements the Noisy Gates formalism",
            backend_version="1.0",
            **fields,
        )

        self.device : FakeBackend | Backend | None = device
        self.gates_ng = standard_gates
        self.circuit_class_ng = BinaryCircuit
        self.parallel_ng : bool = False
        self.simulator : MrAndersonSimulator = MrAndersonSimulator(gates=self.gates_ng, CircuitClass=self.circuit_class_ng, parallel= self.parallel_ng)

        self._target = Target(num_qubits=n_qubits)
        for gate in (
            HGate(), XGate(), SXGate(),
            RZGate(0), RXGate(0), RYGate(0),
            UGate(0, 0, 0),
            CXGate(),
            RZXGate(0),
            ECRGate()):
            self._target.add_instruction(gate)
        self._target.add_instruction(Measure())
        self._target.add_instruction(Delay(0))
        self._target.add_instruction(Barrier(1))
        self._options = self._default_options()

    @property
    def target(self) -> Target:
        return self._target
    
    @property
    def max_circuits(self) -> None:
        return None
 
    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots = 1024,
            seed_simulator=None
        )
    
    def set_gates_ng(self, gate : Gates) -> Gates:
        self.gates_ng = gate
        self.simulator.gates = gate
        return gate
    
    def set_circuit_class_ng(self, circuit_class : str):
        if circuit_class == 'EfficientCircuit':
            self.circuit_class_ng = EfficientCircuit
            self.simulator.CircuitClass = self.circuit_class_ng
        elif circuit_class == 'AlternativeCircuit':
            self.circuit_class_ng = AlternativeCircuit
            self.simulator.CircuitClass = self.circuit_class_ng
        elif circuit_class == 'BinaryCircuit':
            self.circuit_class_ng = BinaryCircuit
            self.simulator.CircuitClass = self.circuit_class_ng
        else: 
            raise ValueError("Class circuit provided doesn't exist or is not available")
        
    def ng_transpile(self, circ : QuantumCircuit, init_layout: list, seed: int) -> QuantumCircuit:
        """Function to transpile a circuit using the optimal option for this backend.

        Args:
            circ (QuantumCircuit): Quantum circuit to be transpiled
            init_layout (list): initial layout of the qubit
            seed (int): seed for the transpilation

        Returns:
            QuantumCircuit: Transpiled circuit ready for the run
        """

        self.options.seed_simulator = seed

        t_circ = transpile(
        circuits=circ,
        backend=self,
        initial_layout=init_layout,
        seed_transpiler=self.options.seed_simulator
        )

        return t_circ

    def set_parallel(self, paral: bool):
        self.parallel_ng = paral
        self.simulator.parallel = paral
    
    def gates(self):
        """Give a list of the possible Gates set based on the pulse

        Returns:
            list: list of the Gates
        """
        gates_type = ['standard_gates', 'numerical_gates']
        return gates_type
    
    def circuit_classes(self):
        """Give a list of the possible Circuit class based on the backend of the model

        Returns:
            list: list of the circuit class
        """
        classes = ['EfficientCircuit', 'AlternativeCircuit', 'BinaryCircuit']
        return classes
    
    def parameter_from_device(self, qubits_layout : list) -> dict:
        """From a given configuration of the qubit_layout is return a dictionary with the information of the real device

        Args:
            qubits_layout (list): Layout of the physical qubit

        Returns:
            dict: dictionary with the information of the real device loaded from the real backend
        """
        device_param = DeviceParameters(list(np.arange(max(qubits_layout)+1)))
        device_param.load_from_backend(self.device)
        device_param_lookup = device_param.__dict__()
        return device_param_lookup

    def process_layout(self, circ: QuantumCircuit):
        qubit_map = {bit: idx for idx, bit in enumerate(circ.qubits)}
        clbit_map = {bit: idx for idx, bit in enumerate(circ.clbits)}

        used_q: list[int] = []
        measure_qc: list[tuple[int, int]] = []

        for x in circ.data:
            op = x.operation
            qargs = x.qubits
            cargs = x.clbits

            if op.name != "delay":
                for qbit in qargs:
                    idx = qubit_map[qbit]
                    if idx not in used_q:
                        used_q.append(idx)

            if op.name == "measure" and qargs and cargs:
                measure_qc.append((qubit_map[qargs[0]], clbit_map[cargs[0]]))

        return used_q, measure_qc, len(used_q)

    def run(self, circuits: QuantumCircuit, shots : int = 1024, **fields) -> NoisyGatesJob:
        """Standard run function of a qiskit backend that submit a circuit to the simulator and return a job object

        Args:
            circuits (QuantumCircuit): Quantum circuit to execute by the simulator after the transpilation
            shots (int, optional): number of repetition of the experiment. Defaults to 1024.

        Returns:
            NoisyGatesJob: Result of the run of the circuit 
        """

        self.options.shots = shots
        
        # Preprocess the qubit_layout after the transpilation
        qubits_layout_t, qubit_bit, n_qubit_t = self.process_layout(circuits)

        n_measured_qubit = len(qubit_bit)  # Number of measured qubit
        if n_measured_qubit == 0:
            raise ValueError("None qubit measured")

        if n_qubit_t > 40:
            raise ValueError("Memory Error -> Too many qubits used in the circuit.")

        psi0 = [1] + [0] * (2**n_qubit_t-1) # starting state
        device_parameters = self.parameter_from_device(qubits_layout_t)
        
        # Define the information for the job
        job_id = str(uuid.uuid4())  # job_id

        # Run the experiment
        start = time.time()

        counts_ng = self.simulator.run(
            t_qiskit_circ=circuits, 
            qubits_layout=qubits_layout_t, 
            psi0=np.array(psi0), 
            shots= self.options.shots, 
            device_param=device_parameters,  # device parameters
            nqubit=n_qubit_t)
        
        end = time.time()

        # Measurament and post-process the result
        counts_ng = fix_counts(counts_ng, n_measured_qubit)  # convert in qiskit notation

        # Convert the result compatible with Qiskit
        exp_data = ExperimentResultData(counts=counts_ng) 
        result_1 = ExperimentResult(shots=self.options.shots , success=True, data = exp_data)
        data = []
        data.append(result_1)

        result = Result(
            backend_name= self.name,
            backend_version= self.backend_version,
            qobj_id=id(circuits),
            job_id=job_id,
            results=data,
            status="COMPLETED",
            success=True,
            time_taken=(end - start),
        )
                
        job = NoisyGatesJob(self, job_id, result)
        return job 

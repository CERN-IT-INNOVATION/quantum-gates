from dotenv import load_dotenv
import os

from qiskit_ibm_runtime import QiskitRuntimeService
from quantum_gates._utility.device_parameters import DeviceParameters


# Environment variables
load_dotenv()
HUB = os.environ["HUB"]
GROUP = os.environ["GROUP"]
PROJECT = os.environ["PROJECT"]
TOKEN = os.environ["IBM_TOKEN"]

# Settings
BACKEND = "ibmq_brisbane"
NUM_QUBITS = 15


service = QiskitRuntimeService(
    channel="ibm_quantum",
    token=TOKEN,
    instance=f"{HUB}/{GROUP}/{PROJECT}",
)

backend = service.backend(BACKEND)

device_param = DeviceParameters(qubits_layout=list(range(NUM_QUBITS)))
device_param.load_from_backend(backend)

device_param.save_to_json("")
device_param.save_to_texts(location="")

print("Device parameters saved.")

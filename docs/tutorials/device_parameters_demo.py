"""
Get the device parameters of an IBM backend and save them as json or in text files.

Usage:
- Do pip install quantum-gates.
- Go to IBM Quantum Lab and get your token.
- Copy paste your IBM token into configuration/token.py as variable IBM_TOKEN. This file should not be versioned.
- Set the hub, group, project and device name in this script.
- Run the script with path on top level of the repository.

Note:
- The files are saved in the outputs folder.
"""

import os

from quantum_gates.utilities import setup_backend
# from quantum_gates.utilities import DeviceParameters
from src.quantum_gates._utility.device_parameters import DeviceParameters
from configuration.token import IBM_TOKEN


# Setup backend
backend_config = {
    "hub": "ibm-q",
    "group": "open",
    "project": "main",
    "device_name": "ibmq_manila"
}
backend = setup_backend(Token=IBM_TOKEN, **backend_config)

# Load device parameters from IBM backend
qubits_layout = [0, 1, 2, 3, 4]
device_param = DeviceParameters(qubits_layout)
device_param.load_from_backend(backend)
location = "docs/tutorials/outputs/device_parameters/"
if not os.path.exists(location[:-1]):
    os.makedirs(location[:-1])

# Save device parameters as json
device_param.save_to_json(location)

# Save device parameters as text
device_param.save_to_texts(location)

# Load device param from text
device_param_from_text = DeviceParameters(qubits_layout)
device_param_from_text.load_from_texts(location)
print(device_param_from_text)

# Load device param from json
device_param_from_json = DeviceParameters(qubits_layout)
device_param_from_json.load_from_texts(location)
print(device_param_from_json)

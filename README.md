# Noisy Quantum Gates [![Made at QMTS!](https://img.shields.io/badge/University%20of%20Trieste-Bassi%20Group-brightgreen)](http://www.qmts.it/) [![Made at CERN!](https://img.shields.io/badge/CERN-CERN%20openlab-brightgreen)](https://openlab.cern/) [![Made at CERN!](https://img.shields.io/badge/CERN-Open%20Source-%232980b9.svg)](https://home.cern) [![Made at CERN!](https://img.shields.io/badge/CERN-QTI-blue)](https://quantum.cern/our-governance)

Implementation of the Noisy Quantum Gates model, published in [Di Bartolomeo, 2023](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043210). It is a novel method to simulate the noisy behaviour of quantum devices by incorporating the noise directly in the gates, which become stochastic matrices. 


## Documentations
The documentation for Noisy Quantum Gates can be accessed on the website 
<a href="https://quantum-gates.readthedocs.io/en/latest/index.html" target="_blank"> Read the Docs</a>.


## How to install
### Requirements
The Python version should be 3.9 or later. Find your Python version by typing `python` or `python3` in the CLI. 
We recommend using the repo together with an [IBM Quantum Lab](https://quantum-computing.ibm.com/lab) account, as it necessary for circuit compilation with Qiskit in many cases. 


### Installation as a user
The library is available on the Python Package Index (PyPI) with `pip install quantum-gates`. 


### Installation as a contributor
For users who want to have control over the source code, we recommend the following installation. Clone the repository 
from [Github](https://github.com/CERN-IT-INNOVATION/quantum-gates), create a new virtual environment, and activate the 
environment. Then you can build the wheel and install it with the package manager of your choice as described in the 
section [How to contribute](#how-to-contribute). This will install all dependencies in your virtual environment, 
and install a working version of the library. 


## Quickstart 
You can find this quickstart implemented in the tutorial notebook [here](docs/tutorials/notebooks/tutorial_quantum_gates.ipynb).

Execute the following code in a script or notebook. Add your IBM token to by defining it as the variable IBM_TOKEN = "your_token". Optimally, you save your token in a separate file that is not in your version control system, so you are not at risk of accidentally revealing your access token. 


```python
# Standard libraries
import numpy as np
import json

# Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram

# Own library
from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import standard_gates
from quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from quantum_gates.utilities import DeviceParameters
from quantum_gates.utilities import setup_backend
IBM_TOKEN = "<your_token>"
```
We create a quantum circuit with Qiskit. 

```python
circ = QuantumCircuit(2,2)
circ.h(0)
circ.cx(0,1)
circ.barrier(range(2))
circ.measure(range(2),range(2))
circ.draw('mpl')
```

We load the configuration from a json file or from code with
```python
config = {
    "backend": {
        "hub": "ibm-q",
        "group": "open",
        "project": "main",
        "device_name": "ibm_kyiv"
    },
    "run": {
        "shots": 1000,
        "qubits_layout": [0, 1],
        "psi0": [1, 0, 0, 0]
    }
}
```
... and setup the Qiskit backend used for the circuit transpilation.

```python
backend_config = config["backend"]
backend = setup_backend(Token=IBM_TOKEN, **backend_config)
run_config = config["run"]
```

This allows us to load the device parameters, which represent the noise of the quantum hardware. 
```python
qubits_layout = run_config["qubits_layout"]
device_param = DeviceParameters(qubits_layout)
device_param.load_from_backend(backend)
device_param_lookup = device_param.__dict__()
```

Last, we perform the simulation ... 
```python
sim = MrAndersonSimulator(gates=standard_gates, CircuitClass=EfficientCircuit)

t_circ = transpile(
    circ,
    backend,
    scheduling_method='asap',
    initial_layout=qubits_layout,
    seed_transpiler=42
)

probs = sim.run(
    t_qiskit_circ=t_circ, 
    qubits_layout=qubits_layout, 
    psi0=np.array(run_config["psi0"]), 
    shots=run_config["shots"], 
    device_param=device_param_lookup,
    nqubit=2,
    level_opt = 0)

```
... and analyse the result. 

```python
plot_histogram(probs, bar_labels=False, legend=['Noisy Gates simulation'])
```

If you want to use a non-linear topology you must use the BinaryCircuit and slightly modify the code.
First of all we modify the qubit layout to match the topology of the device.

```python
config = {
    "run": {
        "shots": 1000,
        "qubits_layout": [0, 14],
        "psi0": [1, 0, 0, 0]
    }
}
```
Then also the command to import the parameter device has to change in order to import all the information of all the qubits up to the one with the max indices.

```python
device_param = DeviceParameters(list(np.arange(max(qubits_layout)+1)))
device_param.load_from_backend(backend)
device_param_lookup = device_param.__dict__()
```

Last, we perform the simulation ... 
```python
sim = MrAndersonSimulator(gates=standard_gates, CircuitClass=BinaryCircuit)

t_circ = transpile(
    circ,
    backend,
    scheduling_method='asap',
    initial_layout=qubits_layout,
    seed_transpiler=42
)

probs = sim.run(
    t_qiskit_circ=t_circ, 
    qubits_layout=qubits_layout, 
    psi0=np.array(run_config["psi0"]), 
    shots=run_config["shots"], 
    device_param=device_param_lookup,
    nqubit=2,
    level_opt = 4)

```
... and analyse the result. 

```python
plot_histogram(probs, bar_labels=False, legend=['Noisy Gates simulation'])
```

Remember that the ouput of the simulator follows the Big-Endian order, if you want to switch to Little-Endian order, which is the standard for Qiskit, you can use the command

```python
from quantum_gates.utilities import fix_counts

n_measured_qubit = 2
probs_little_endian = fix_counts(probs, n_measured_qubit)
plot_histogram(probs_little_endian, bar_labels=False, legend=['Noisy Gates simulation'])
```


# Usage
We recommend to read the [overview](https://quantum-gates.readthedocs.io/en/latest/index.html) of the documentation as a 2-minute preparation. 


## Imports
There are two ways of importing the package. 1) If you installed the code with pip, then the imports are simply of the form seen in the [Quickstart](<#quickstart>). 

```python
from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import standard_gates
from quantum_gates.circuits import EfficientCircuit
from quantum_gates.utilities import DeviceParameters, setup_backend
```

2) If you use the source code directly and develop within the repository, then the imports become

```python
from src.quantum_gates._simulation.simulator import MrAndersonSimulator
from src.quantum_gates._gates.gates import standard_gates
from src.quantum_gates._simulation.circuit import EfficientCircuit
from src.quantum_gates._utility.device_parameters import (
    DeviceParameters, 
    setup_backend
)
``` 


# Functionality
The main components are the [gates](https://quantum-gates.readthedocs.io/en/latest/gates.html), 
and the [simulator](https://quantum-gates.readthedocs.io/en/latest/simulators.html). 
One can configure the gates with different [pulse shapes](https://quantum-gates.readthedocs.io/en/latest/pulses.html>), 
and the simulator with different [circuit classes](https://quantum-gates.readthedocs.io/en/latest/circuits.html>) and 
[backends](https://quantum-gates.readthedocs.io/en/latest/backends.html). The circuit classes use a specific 
backend for the statevector simulation. 
The [EfficientBackend](https://quantum-gates.readthedocs.io/en/latest/backends.html) has the same functionality as 
the [StandardBackend](https://quantum-gates.readthedocs.io/en/latest/backends.html), but is much more performant 
thanks to optimized tensor contraction algorithms. We also provide various
[quantum algorithms](https://quantum-gates.readthedocs.io/en/latest/quantum_algorithms.html) as circuits, and 
scripts to run the circuits with the simulator, the IBM simulator, and a real IBM backend. Last, all functionality is 
unit tested and one can get sample code from the unit tests.


# Unit Tests
We recommend running the unit tests once you are finished with the setup of your environment. As some tests need access
to IBM devices, you have to follow the steps outlined in token.py. Make sure that your token is active and you have 
accepted all license agreement with IBM in your IBM account. Before you run the tests, make sure that the device you are 
testing with (the on set in tests/simulation/test_anderson_simulator.py) is available in your account and the device 
parameters are prepared in the tests/utility/device_parameters folder. In the future we might upgrade the tests and mock 
these dependencies.


# How to contribute
Contributions are welcomed and should apply the usual git-flow: fork this repo, create a local branch named 
'feature-...'. Commit often to ensure that each commit is easy to understand. Name your commits 
'[feature-...] Commit message.', such that it possible to differentiate the commits of different features in the 
main line. Request a merge to the mainline often. Contribute to the test suite and verify the functionality with the unit tests when using a different Python version or dependency versions. Please remember to follow the 
[PEP 8 style guide](https://peps.python.org/pep-0008/), and add comments whenever it helps. The corresponding 
[authors](<#authors>) are happy to support you. 


## Build 
You may also want to create your own distribution and test it. Navigate to the repository in your CLI of choice. 
Build the wheel with the command `python3 -m build --sdist --wheel .` and navigate to the distribution with `cd dist`. 
Use `ls` to display the name of the wheel, and run `pip install <filename>.whl` with the correct filename. 
Now you can use your version of the library. 


# Credits
Please cite the work using the following BibTex entry:

```text
@article{PhysRevResearch.5.043210,
  title = {Noisy gates for simulating quantum computers},
  author = {Di Bartolomeo, Giovanni and Vischi, Michele and Cesa, Francesco and Wixinger, Roman and Grossi, Michele and Donadi, Sandro and Bassi, Angelo},
  journal = {Phys. Rev. Res.},
  volume = {5},
  issue = {4},
  pages = {043210},
  numpages = {19},
  year = {2023},
  month = {Dec},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.5.043210},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.5.043210}
}
```


# Authors
This project has been developed thanks to the effort of the following people:

* Giovanni Di Bartolomeo (dibartolomeo.giov@gmail.com)
* Michele Vischi (vischimichele@gmail.com)
* Francesco Cesa
* Michele Grossi (michele.grossi@cern.ch) 
* Sandro Donadi
* Angelo Bassi 
* Paolo Da Rold 
* Roman Wixinger (roman.wixinger@gmail.com)

"""
Class for loading, storing, validating and passing device parameters. These parameter represent the noise level of the
device.
"""

import os
from datetime import datetime
import json
import numpy as np


class DeviceParameters(object):
    """Snapshot of the noise of the IBM backend. Can load and save the properties.

    Args:
        qubits_layout (list[int]): Layout of the qubits.

    Attributes:
        qubits_layout (list[int]): Layout of the qubits.
        nr_of_qubits (int): Number of qubits to be used.
        T1 (np.array): T1 time.
        T2 (np.array): T2 time.
        p (np.array): To be added.
        rout (np.array): To be added.
        p_cnot (np.array): Error probabilites in the CNOT gate.
        p_cnot (np.array): Gate time to implement controlled not operations in the CNOT gate.
        tm (np.array): To be added.
        dt (np.array): To be added.
        
    """

    # Filename when storing the data in text files or a single json file
    f_T1 = "T1.txt"
    f_T2 = "T2.txt"
    f_p = "p.txt"
    f_rout = "rout.txt"
    f_p_cnot = "p_cnot.txt"
    f_t_cnot = "t_cnot.txt"
    f_tm = "tm.txt"
    f_dt = "dt.txt"
    f_json = "device_parameters.json"
    f_metadata = "metadata.json"

    def __init__(self, qubits_layout: list):
        self.qubits_layout = qubits_layout
        self.nr_of_qubits = len(qubits_layout)
        self.T1 = None
        self.T2 = None
        self.p = None
        self.rout = None
        self.p_cnot = None
        self.t_cnot = None
        self.tm = None
        self.dt = None
        self.metadata = None
        self._names = ["T1", "T2", "p", "rout", "p_cnot", "t_cnot", "tm", "dt", "metadata"]
        self._f_txt = ["T1.txt", "T2.txt", "p.txt", "rout.txt", "p_cnot.txt", "t_cnot.txt", "tm.txt", "dt.txt",
                       "metadata.json"]

    def load_from_json(self, location: str):
        """ Load device parameters from single json file at the location.
        """
        # Verify that it exists
        self._json_exists_at_location(location)

        # Load
        f = open(location + self.f_json)
        data_dict = json.load(f)

        # Check json keys
        if any((name not in data_dict for name in self._names)):
            raise Exception("Loading of device parameters from json not successful: At least one quantity is missing.")

        # Add lists to instance as arrays
        self.T1 = np.array(data_dict["T1"])
        self.T2 = np.array(data_dict["T2"])
        self.p = np.array(data_dict["p"])
        self.rout = np.array(data_dict["rout"])
        self.p_cnot = np.array(data_dict["p_cnot"])
        self.t_cnot = np.array(data_dict["t_cnot"])
        self.tm = np.array(data_dict["tm"])
        self.dt = np.array(data_dict["dt"])
        self.metadata = data_dict["metadata"]

        # Verify
        if not self.is_complete():
            raise Exception("Loading of device parameters from json was not successful: Did not pass verification.")

        return

    def load_from_texts(self, location: str):
        """ Load device parameters from many text files at the location.
        """

        # Verify that exists
        self._texts_exist_at_location(location)

        # Load -> If the text has only one line, we have to make it into an 1x1 array explicitely.
        if self.nr_of_qubits == 1:
            # Here we use 'array' because with only one qubit 'loadtxt' doesn't load an array
            self.T1 = np.array([np.loadtxt(location + self.f_T1)])
            self.T2 = np.array([np.loadtxt(location + self.f_T2)])
            self.p = np.array([np.loadtxt(location + self.f_p)])
            self.rout = np.array([np.loadtxt(location + self.f_rout)])
            self.p_cnot = np.array([np.loadtxt(location + self.f_p_cnot)])
            self.t_cnot = np.array([np.loadtxt(location + self.f_t_cnot)])
            self.tm = np.array([np.loadtxt(location + self.f_tm)])
        else:
            self.T1 = np.loadtxt(location + self.f_T1)
            self.T2 = np.loadtxt(location + self.f_T2)
            self.p = np.loadtxt(location + self.f_p)
            self.rout = np.loadtxt(location + self.f_rout)
            self.p_cnot = np.loadtxt(location + self.f_p_cnot)
            self.t_cnot = np.loadtxt(location + self.f_t_cnot)
            self.tm = np.loadtxt(location + self.f_tm)
        self.dt = np.array([np.loadtxt(location + self.f_dt)])
        with open(location + self.f_metadata, "r") as metadata_file:
            self.metadata = json.load(metadata_file)

        # Verify
        if not self.is_complete():
            raise Exception("Loading of device parameters from text files was not successful: Did not pass verification.")

        return

    def load_from_backend(self, backend):
        """ Load device parameters from the IBM backend. """

        # Load
        prop = backend.properties()
        config = backend.configuration()
        defaults = backend.defaults()

        self.T1 = [prop.t1(j) for j in self.qubits_layout]
        self.T2 = [prop.t2(j) for j in self.qubits_layout]
        self.p = [prop.gate_error('x', [j]) for j in self.qubits_layout]
        self.rout = [prop.readout_error(j) for j in self.qubits_layout]
        self.dt = [backend.configuration().dt]
        self.tm = [prop.readout_length(j) for j in self.qubits_layout]
        self.metadata = {
            "version": datetime.today().strftime('%Y%m%d'),
            "device": config.backend_name,
            "qubits": config.n_qubits,
            "qubits_layout": self.qubits_layout,
            "config": config.to_dict()
        }

        t_cnot = np.zeros((self.nr_of_qubits, self.nr_of_qubits))
        p_cnot = np.zeros((self.nr_of_qubits, self.nr_of_qubits))

        if self.nr_of_qubits > 1:
            for i in range(self.nr_of_qubits):
                if i == 0:
                    t_cnot[0][1] = prop.gate_length('cx', [self.qubits_layout[0], self.qubits_layout[1]])
                    p_cnot[0][1] = prop.gate_error('cx', [self.qubits_layout[0], self.qubits_layout[1]])
                if i != 0 and i != self.nr_of_qubits-1:
                    t_cnot[i][i-1] = prop.gate_length('cx', [self.qubits_layout[i], self.qubits_layout[i-1]])
                    p_cnot[i][i-1] = prop.gate_error('cx', [self.qubits_layout[i], self.qubits_layout[i-1]])
                    t_cnot[i][i+1] = prop.gate_length('cx', [self.qubits_layout[i], self.qubits_layout[i+1]])
                    p_cnot[i][i+1] = prop.gate_error('cx', [self.qubits_layout[i], self.qubits_layout[i+1]])
                if i == self.nr_of_qubits-1:
                    t_cnot[i][i-1] = prop.gate_length('cx', [self.qubits_layout[i], self.qubits_layout[i-1]])
                    p_cnot[i][i-1] = prop.gate_error('cx', [self.qubits_layout[i], self.qubits_layout[i-1]])
        self.t_cnot = t_cnot
        self.p_cnot = p_cnot

        # Verify
        if not self.is_complete():
            raise Exception("Loading of device parameters from the backend was not successful: Did not pass verification.")

        return

    def save_to_texts(self, location: str):
        """ Save device parameters to text files at a specific location.
        """
        # Verify parameters
        if not self.is_complete():
            raise Exception("Saving the device parameter was not successful: Did not pass verification.")
        print("Device parameters are valid. We can save them. ")

        # Save parameters
        np.savetxt(location + self.f_T1, self.T1)
        np.savetxt(location + self.f_T2, self.T2)
        np.savetxt(location + self.f_p, self.p)
        np.savetxt(location + self.f_rout, self.rout)
        np.savetxt(location + self.f_p_cnot, self.p_cnot)
        np.savetxt(location + self.f_t_cnot, self.t_cnot)
        np.savetxt(location + self.f_dt, self.dt)
        np.savetxt(location + self.f_tm, self.tm)
        with open(location + self.f_metadata, 'w') as fp:
            json.dump(self.metadata, fp, indent=4, sort_keys=False, default=default_serializer)
        print("Device parameters saved successfully.")
        return

    def save_to_json(self, location: str):
        """ Save device parameters to a json file at the location. The arrays are converted to lists in the process.
        """
        # Verify
        if not self.is_complete():
            raise Exception("Saving the device parameter was not successful: Did not pass verification.")
        print("Device parameters are valid. We can save them. ")

        # Build dict and convert arrays to list
        device_parameter_dict = self.__dict__()
        for key in device_parameter_dict:
            # Convert array to list
            if isinstance(device_parameter_dict[key], np.ndarray):
                device_parameter_dict[key] = device_parameter_dict[key].tolist()

        # Save
        with open(location + self.f_json, 'w') as fp:
            json.dump(device_parameter_dict, fp, indent=4, sort_keys=False, default=default_serializer)
        print("Device parameters saved successfully.")
        return

    def get_as_tuple(self) -> tuple:
        """ Get the parameters as a tuple. The parameters have to be already loaded.
        """
        if not self.is_complete():
            raise Exception("Exception in DeviceParameters.get_as_tuble(): At least one of the parameters is None.")
        return self.T1, self.T2, self.p, self.rout, self.p_cnot, self.t_cnot, self.tm, self.dt, self.metadata

    def is_complete(self) -> bool:
        """ Returns whether all device parameters have been successfully initialized.
        """
        # Check not None
        if any((
                self.T1 is None,
                self.T2 is None,
                self.p is None,
                self.rout is None,
                self.p_cnot is None,
                self.t_cnot is None,
                self.tm is None,
                self.dt is None,
                self.metadata is None)):
            return False

        return True

    def check_T1_and_T2_times(self, do_raise_exception: bool) -> bool:
        """ Checks the T1 and T2 times. Raises an exception in case of invalid T1, T2 times if the flag is set. Returns
            whether or not all qubits are flawless.
        """

        print("Verifying the T1 and T2 times of the device: ")
        nr_bad_qubits = 0
        for i, (T1, T2) in enumerate(zip(self.T1, self.T2)):
            if T1 >= 2*T2:
                nr_bad_qubits += 1
                print('The qubit n.', self.qubits_layout[i], 'is bad.')
                print('Delete the affected qubit from qubits_layout and change the layout.')

        if nr_bad_qubits:
            print(f'Attention, there are {nr_bad_qubits} bad qubits.')
            print('In case of side effects contact Jay Gambetta.')
        else:
            print('All right!')

        if nr_bad_qubits and do_raise_exception:
            raise Exception(f'Stop simulation: The DeviceParameters class found {nr_bad_qubits} bad qubits.')

        return nr_bad_qubits == 0

    def _texts_exist_at_location(self, location):
        """ Checks if the text files with the device parameters exist at the expected location. Raises an exception
            if more than one text is missing.
        """
        missing = [f for f in self._f_txt if not os.path.exists(location + f)]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"DeviceParameter found that at {location} the files {missing} are missing."
            )
        return

    def _json_exists_at_location(self, location):
        """ Checks if the json files with the device parameters exist, otherwise raises an exception.
        """
        if not os.path.exists(location + self.f_json):
            raise FileNotFoundError(
                f"DeviceParameter found that at {location} the file {self.f_json} is missing."
            )
        return

    def __dict__(self):
        """ Get dict representation. """
        return {
            "T1": self.T1,
            "T2": self.T2,
            "p": self.p,
            "rout": self.rout,
            "p_cnot": self.p_cnot,
            "t_cnot": self.t_cnot,
            "tm": self.tm,
            "dt": self.dt,
            "metadata": self.metadata
        }

    def __str__(self):
        """ Representation as str. """
        return json.dumps(self.__dict__(), indent=4, default=default_serializer)

    def __eq__(self, other):
        """ Allows us to compare instances. """
        return self.__str__() == other.__str__()


def default_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

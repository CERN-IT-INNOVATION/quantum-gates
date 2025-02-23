from qiskit.providers import ProviderV1
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

from .ng_backend import NoisyGatesBackend


class NoisyGatesProvider(ProviderV1):
    """Provider for backends for Noisy Gates model.
    At the moment is possible to use a backend that work alongside with IBM real device
    """

    def __init__(self, token=None):
        """

        Args:
            token (str): Token for the availability of real device from IBM. Defaults to None.
        """
        super().__init__()
        self.token = token

        if token is not None:
            self.qiskit_provider = QiskitRuntimeService(channel= 'ibm_quantum', token=self.token) 
        else:
            self.qiskit_provider = FakeProviderForBackendV2()

    def backends(self):
        return 

    def ibm_backends(self, **kwargs):
        """Show the possible usable device from IBM

        Raises:
            ValueError: If a token is not provided, it's not possibile to access to IBM device

        Returns:
            List: List of available device from IBM
        """
        if self.token is not None:
            return self.qiskit_provider.backends()
        else:
            return self.qiskit_provider.backends()

    def get_ibm_backend(self, name_backend, **kwargs) -> NoisyGatesBackend:
        """Choose one of the real device to simulare the features during the running process

        Args:
            name_backend (str): Name of the device from IBM, real if it's provided a token, fake if there isn't a token
                                example: - if the device is real the name is 'ibm_sherbrooke'
                                         - if the devise is fake the name is 'fake_sherbrooke'

        Raises:
            ValueError: If a token is not provided, it's not possibile to access to IBM device 

        Returns:
            NoisyGatesBackend: Return a noisy gates backend that use ibm_device as real device from IBM
        """
        if self.token is not None:
            ibm_device = self.qiskit_provider.get_backend(name_backend)
            backend = NoisyGatesBackend(device=ibm_device)
            return backend
        else:
            ibm_device = self.qiskit_provider.backend(name_backend)
            backend = NoisyGatesBackend(device=ibm_device)
            return backend

from qiskit.providers import JobV1 as Job
from qiskit.providers import JobStatus


class NoisyGatesJob(Job):
    """Representation of a Job that will run on an Noisy Gates backend.

    It is not recommended to create Job instances directly, but rather use the
    :meth:`run <NoisyGatesBackend.run>`

    """

    _async = False

    def __init__(self, backend, job_id, result):
        
        self._backend = backend
        self.job_id = job_id
        self._result = result

    def submit(self):
        return
 
    def result(self):
        return self._result
 
    def status(self):
        return JobStatus.DONE



    
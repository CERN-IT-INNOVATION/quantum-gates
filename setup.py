from setuptools import setup

setup(
    name='quantum-gates',
    version='1.0.2',
    author='M. Grossi, G. D. Bartolomeo, M. Vischi, R. Wixinger',
    author_email='michele.grossi@cern.ch, dibartolomeo.giov@gmail.com, vischimichele@gmail.com, roman.wixinger@gmail.com',
    packages=['quantum_gates',
              'quantum_gates._gates',
              'quantum_gates._legacy',
              'quantum_gates._simulation',
              'quantum_gates._utility'],
    license='MIT',
    python_requires='>=3.9'
)

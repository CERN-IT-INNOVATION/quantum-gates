import numpy as np
import os

from src.quantum_gates.utilities import post_process_split


location = 'tests/helpers/result_samples'


def test_post_process_split_mean():
    source_filenames = [f"{location}/file{i}.txt" for i in range(1, 5)]
    target_filenames = [f"{location}/target.txt"]
    post_process_split(source_filenames, target_filenames, 4)
    mean_array = np.loadtxt(target_filenames[0])
    mean_expected = np.array([0.8, 0.05, 0.05, 0.1])
    assert all((abs(mean_array[i] - mean_expected[i]) < 1e-9 for i in range(len(source_filenames)))), \
        f"Expected {mean_expected} but found {mean_array}."
    for f in target_filenames:
        os.remove(f)

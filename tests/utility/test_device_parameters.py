import pytest
import numpy as np

from src.quantum_gates.utilities import DeviceParameters


location = 'tests/helpers/device_parameters/ibmq_kyoto/'
invalid_location = 'invalid_location'


def test_device_parameters_load_from_json():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    device_param.load_from_json(location=location)


def test_device_parameters_load_from_texts():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    device_param.load_from_texts(location=location)


def test_load_device_parameters_is_equal():
    device_param_text = DeviceParameters([0, 1, 2, 3, 4])
    device_param_text.load_from_texts(location=location)
    device_param_json = DeviceParameters([0, 1, 2, 3, 4])
    device_param_json.load_from_json(location=location)
    assert device_param_text == device_param_json


def test_device_parameters_load_from_texts_with_invalid_input():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    with pytest.raises(FileNotFoundError):
        device_param.load_from_texts(location=invalid_location)


def test_device_parameters_get_as_tuple():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    device_param.load_from_texts(location=location)
    T1, T2, p, rout, p_int, t_int, tm, dt, metadata = device_param.get_as_tuple()


def test_device_parameters_get_as_tuple_type():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    device_param.load_from_texts(location=location)
    items = device_param.get_as_tuple()
    assert all((isinstance(item, np.ndarray) or isinstance(item, dict) for item in items)), \
        f"Expected to get a tuple of numpy arrays but found {(type(item) for item in items)}."


def test_device_parameters_get_as_tuple_sizes():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    device_param.load_from_texts(location=location)
    T1, T2, p, rout, p_int, t_int, tm, dt, metadata = device_param.get_as_tuple()
    assert T1.shape == (5,), f"T1 had invalid shape {T1.shape} instead of (5)."
    assert T2.shape == (5,), f"T2 had invalid shape {T2.shape} instead of (5)."
    assert p.shape == (5,), f"p had invalid shape {p.shape} instead of (5)."
    assert rout.shape == (5,), f"rout had invalid shape {rout.shape} instead of (5)."
    assert p_int.shape == (5, 5), f"p_int had invalid shape {p_int.shape} instead of (5, 5)."
    assert t_int.shape == (5, 5), f"t_int had invalid shape {t_int.shape} instead of (5, 5)."
    assert tm.shape == (5,), f"tm had invalid shape {tm.shape} instead of (5)"
    assert dt.shape == (1,), f"dt had invalid shape {dt.shape} instead of (5)"
    assert isinstance(metadata, dict), f"Expected metadata to be of type dict, but found {type(metadata)}."


def test_device_parameters_is_complete():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    device_param.load_from_texts(location=location)
    assert device_param.is_complete()


def test_device_parameters_is_complete_while_it_is_not():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    assert not device_param.is_complete()


@pytest.mark.skip(reason="There are actually bad qubits in the dataset.")
def test_device_parameters_check_T1_and_T2_times():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    device_param.load_from_json(location=location)
    assert device_param.check_T1_and_T2_times(do_raise_exception=False)


def test_device_parameters_check_T1_and_T2_times_bad():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    device_param.load_from_json(location=location)
    device_param.T2 = np.zeros_like(device_param.T2)
    assert not device_param.check_T1_and_T2_times(do_raise_exception=False)


def test_device_parameters_check_T1_and_T2_times_bad_with_exception():
    device_param = DeviceParameters([0, 1, 2, 3, 4])
    device_param.load_from_json(location=location)
    device_param.T2 = np.zeros_like(device_param.T2)
    with pytest.raises(Exception):
        device_param.check_T1_and_T2_times(do_raise_exception=True)

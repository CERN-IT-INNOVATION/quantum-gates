import pytest
import numpy as np

from src.quantum_gates.utilities import DeviceParameters


location = 'tests/helpers/device_parameters/'
invalid_location = 'invalid_location'


def test_device_parameters_load_from_json():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param.load_from_json(location=location)


def test_device_parameters_load_from_texts():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param.load_from_texts(location=location)


def test_load_device_parameters_is_equal():
    device_param_text = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param_text.load_from_texts(location=location)
    device_param_json = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param_json.load_from_json(location=location)
    assert device_param_text == device_param_json


def test_device_parameters_load_from_texts_with_invalid_input():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    with pytest.raises(FileNotFoundError):
        device_param.load_from_texts(location=invalid_location)


def test_device_parameters_get_as_tuple():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param.load_from_texts(location=location)
    T1, T2, p, rout, p_cnot, t_cnot, tm, dt = device_param.get_as_tuple()


def test_device_parameters_get_as_tuple_type():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param.load_from_texts(location=location)
    items = device_param.get_as_tuple()
    assert all((isinstance(item, np.ndarray) for item in items)), \
        f"Expected to get a tuple of numpy arrays but found {(type(item) for item in items)}."


def test_device_parameters_get_as_tuple_sizes():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param.load_from_texts(location=location)
    T1, T2, p, rout, p_cnot, t_cnot, tm, dt = device_param.get_as_tuple()
    assert T1.shape == (21,), f"T1 had invalid shape {T1.shape} instead of (21)."
    assert T2.shape == (21,), f"T2 had invalid shape {T2.shape} instead of (21)."
    assert p.shape == (21,), f"p had invalid shape {p.shape} instead of (21)."
    assert rout.shape == (21,), f"rout had invalid shape {rout.shape} instead of (21)."
    assert p_cnot.shape == (21, 21), f"p_cnot had invalid shape {p_cnot.shape} instead of (21, 21)."
    assert t_cnot.shape == (21, 21), f"t_cnot had invalid shape {t_cnot.shape} instead of (21, 21)."
    assert tm.shape == (21,), f"tm had invalid shape {tm.shape} instead of (21)"
    assert dt.shape == (1,), f"dt had invalid shape {dt.shape} instead of (21)"


def test_device_parameters_is_complete():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param.load_from_texts(location=location)
    assert device_param.is_complete()


def test_device_parameters_is_complete_while_it_is_not():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    assert not device_param.is_complete()


@pytest.mark.skip(reason="There are actually bad qubits in the dataset.")
def test_device_parameters_check_T1_and_T2_times():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param.load_from_json(location=location)
    assert device_param.check_T1_and_T2_times(do_raise_exception=False)


def test_device_parameters_check_T1_and_T2_times_bad():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param.load_from_json(location=location)
    device_param.T2 = np.zeros_like(device_param.T2)
    assert not device_param.check_T1_and_T2_times(do_raise_exception=False)


def test_device_parameters_check_T1_and_T2_times_bad_with_exception():
    device_param = DeviceParameters([0,1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2])
    device_param.load_from_json(location=location)
    device_param.T2 = np.zeros_like(device_param.T2)
    with pytest.raises(Exception):
        device_param.check_T1_and_T2_times(do_raise_exception=True)







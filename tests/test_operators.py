import numpy as np

from tinyedgebench import operators


def test_int8_matmul_shape_and_error() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 5)).astype(np.float32)
    b = rng.normal(size=(5, 3)).astype(np.float32)

    fp32 = operators.matmul_fp32(a, b)
    int8 = operators.matmul_int8_sim(a, b)

    assert int8.shape == fp32.shape
    assert np.mean(np.abs(int8 - fp32)) < 0.1


def test_conv2d_shapes_match() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 2, 8, 8)).astype(np.float32)
    weight = rng.normal(size=(3, 2, 3, 3)).astype(np.float32)
    bias = rng.normal(size=(3,)).astype(np.float32)

    fp32 = operators.conv2d_fp32(x, weight, bias, stride=1, padding=1)
    int8 = operators.conv2d_int8_sim(x, weight, bias, stride=1, padding=1)
    shift = operators.conv2d_shift_only(x, weight, bias, stride=1, padding=1)

    assert fp32.shape == (1, 3, 8, 8)
    assert int8.shape == fp32.shape
    assert shift.shape == fp32.shape

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv2d_fp32(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    windows = _conv_windows(x.astype(np.float32, copy=False), weight.shape[-2:], stride, padding)
    out = np.einsum("nchwkl,ockl->nohw", windows, weight.astype(np.float32, copy=False), optimize=True)
    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)
    return out.astype(np.float32, copy=False)


def depthwise_conv2d_fp32(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    windows = _conv_windows(x.astype(np.float32, copy=False), weight.shape[-2:], stride, padding)
    out = np.einsum("nchwkl,ckl->nchw", windows, weight.astype(np.float32, copy=False), optimize=True)
    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)
    return out.astype(np.float32, copy=False)


def matmul_fp32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32, copy=False) @ b.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def batch_matmul_fp32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def linear_fp32(a: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    out = a.astype(np.float32, copy=False) @ weight.astype(np.float32, copy=False)
    if bias is not None:
        out += bias
    return out.astype(np.float32, copy=False)


def conv2d_int8_sim(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    x_q, x_scale = quantize_symmetric_int8(x)
    w_q, w_scale = quantize_symmetric_int8(weight)
    windows = _conv_windows(x_q.astype(np.int32), w_q.shape[-2:], stride, padding)
    acc = np.einsum("nchwkl,ockl->nohw", windows, w_q.astype(np.int32), optimize=True)
    out = acc.astype(np.float32) * (x_scale * w_scale)
    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)
    return out.astype(np.float32, copy=False)


def depthwise_conv2d_int8_sim(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    x_q, x_scale = quantize_symmetric_int8(x)
    w_q, w_scale = quantize_symmetric_int8(weight)
    windows = _conv_windows(x_q.astype(np.int32), w_q.shape[-2:], stride, padding)
    acc = np.einsum("nchwkl,ckl->nchw", windows, w_q.astype(np.int32), optimize=True)
    out = acc.astype(np.float32) * (x_scale * w_scale)
    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)
    return out.astype(np.float32, copy=False)


def matmul_int8_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_q, a_scale = quantize_symmetric_int8(a)
    b_q, b_scale = quantize_symmetric_int8(b)
    acc = a_q.astype(np.int32) @ b_q.astype(np.int32)
    return (acc.astype(np.float32) * (a_scale * b_scale)).astype(np.float32, copy=False)


def batch_matmul_int8_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_q, a_scale = quantize_symmetric_int8(a)
    b_q, b_scale = quantize_symmetric_int8(b)
    acc = np.matmul(a_q.astype(np.int32), b_q.astype(np.int32))
    return (acc.astype(np.float32) * (a_scale * b_scale)).astype(np.float32, copy=False)


def linear_int8_sim(a: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    a_q, a_scale = quantize_symmetric_int8(a)
    w_q, w_scale = quantize_symmetric_int8(weight)
    acc = a_q.astype(np.int32) @ w_q.astype(np.int32)
    out = acc.astype(np.float32) * (a_scale * w_scale)
    if bias is not None:
        out += bias
    return out.astype(np.float32, copy=False)


def matmul_shift_only(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return matmul_fp32(quantize_to_powers_of_two(a), quantize_to_powers_of_two(b))


def batch_matmul_shift_only(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return batch_matmul_fp32(quantize_to_powers_of_two(a), quantize_to_powers_of_two(b))


def linear_shift_only(a: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    return linear_fp32(quantize_to_powers_of_two(a), quantize_to_powers_of_two(weight), bias)


def conv2d_shift_only(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    return conv2d_fp32(quantize_to_powers_of_two(x), quantize_to_powers_of_two(weight), bias, stride, padding)


def depthwise_conv2d_shift_only(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    return depthwise_conv2d_fp32(
        quantize_to_powers_of_two(x),
        quantize_to_powers_of_two(weight),
        bias,
        stride,
        padding,
    )


def dequantize_symmetric_int8(values: np.ndarray) -> np.ndarray:
    quantized, scale = quantize_symmetric_int8(values)
    return (quantized.astype(np.float32) * scale).astype(np.float32, copy=False)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0).astype(np.float32, copy=False)


def relu6(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 6).astype(np.float32, copy=False)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x).astype(np.float32, copy=False)


def gelu(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))).astype(
        np.float32,
        copy=False,
    )


def silu(x: np.ndarray) -> np.ndarray:
    return (x * sigmoid(x)).astype(np.float32, copy=False)


def leaky_relu(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, negative_slope * x).astype(np.float32, copy=False)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32, copy=False)


def selu(x: np.ndarray) -> np.ndarray:
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return (scale * elu(x, alpha=alpha)).astype(np.float32, copy=False)


def celu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return (np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1.0))).astype(np.float32, copy=False)


def softplus(x: np.ndarray) -> np.ndarray:
    return (np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)).astype(np.float32, copy=False)


def softsign(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.abs(x))).astype(np.float32, copy=False)


def hard_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.clip((x + 3.0) / 6.0, 0.0, 1.0).astype(np.float32, copy=False)


def hard_swish(x: np.ndarray) -> np.ndarray:
    return (x * hard_sigmoid(x)).astype(np.float32, copy=False)


def mish(x: np.ndarray) -> np.ndarray:
    return (x * np.tanh(softplus(x))).astype(np.float32, copy=False)


def prelu(x: np.ndarray, slope: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, slope * x).astype(np.float32, copy=False)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=axis, keepdims=True)).astype(np.float32, copy=False)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return (shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))).astype(np.float32, copy=False)


def maxpool2d(x: np.ndarray, kernel_size: tuple[int, int], stride: int = 2) -> np.ndarray:
    windows = _pool_windows(x.astype(np.float32, copy=False), kernel_size, stride)
    return np.max(windows, axis=(-2, -1)).astype(np.float32, copy=False)


def avgpool2d(x: np.ndarray, kernel_size: tuple[int, int], stride: int = 2) -> np.ndarray:
    windows = _pool_windows(x.astype(np.float32, copy=False), kernel_size, stride)
    return np.mean(windows, axis=(-2, -1)).astype(np.float32, copy=False)


def global_avgpool2d(x: np.ndarray) -> np.ndarray:
    return np.mean(x.astype(np.float32, copy=False), axis=(2, 3), keepdims=True).astype(np.float32, copy=False)


def batchnorm2d(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return (
        (x - running_mean.reshape(1, -1, 1, 1))
        / np.sqrt(running_var.reshape(1, -1, 1, 1) + eps)
        * gamma.reshape(1, -1, 1, 1)
        + beta.reshape(1, -1, 1, 1)
    ).astype(np.float32, copy=False)


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * gamma + beta).astype(np.float32, copy=False)


def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    return (x / rms * gamma).astype(np.float32, copy=False)


def groupnorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, groups: int = 1, eps: float = 1e-5) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n, c, h, w = x.shape
    groups = max(1, min(groups, c))
    if c % groups != 0:
        groups = 1
    reshaped = x.reshape(n, groups, c // groups, h, w)
    mean = np.mean(reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(reshaped, axis=(2, 3, 4), keepdims=True)
    normed = ((reshaped - mean) / np.sqrt(var + eps)).reshape(n, c, h, w)
    return (normed * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)).astype(np.float32, copy=False)


def instance_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2, 3), keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)).astype(
        np.float32,
        copy=False,
    )


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True) + eps)
    return (x / denom).astype(np.float32, copy=False)


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32, copy=False) + b.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32, copy=False) * b.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32, copy=False) / (b.astype(np.float32, copy=False) + 1e-3)).astype(np.float32, copy=False)


def maximum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def minimum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def bias_add(x: np.ndarray, bias: np.ndarray) -> np.ndarray:
    if x.ndim == 4:
        return (x + bias.reshape(1, -1, 1, 1)).astype(np.float32, copy=False)
    return (x + bias).astype(np.float32, copy=False)


def concat(a: np.ndarray, b: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.concatenate([a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)], axis=axis).astype(
        np.float32,
        copy=False,
    )


def transpose(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4:
        return np.transpose(x, (0, 2, 3, 1)).astype(np.float32, copy=False)
    return np.swapaxes(x, -1, -2).astype(np.float32, copy=False)


def reshape(x: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    return np.reshape(x, target_shape).astype(np.float32, copy=False)


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32, copy=False)


def squeeze(x: np.ndarray) -> np.ndarray:
    return np.squeeze(x).astype(np.float32, copy=False)


def expand_dims(x: np.ndarray, axis: int = 1) -> np.ndarray:
    safe_axis = max(0, min(axis, x.ndim))
    return np.expand_dims(x, axis=safe_axis).astype(np.float32, copy=False)


def tile(x: np.ndarray, repeats: int = 2) -> np.ndarray:
    reps = (1,) * (x.ndim - 1) + (repeats,)
    return np.tile(x, reps).astype(np.float32, copy=False)


def slice_tensor(x: np.ndarray) -> np.ndarray:
    slices = tuple(slice(0, max(1, dim // 2)) for dim in x.shape)
    return x[slices].astype(np.float32, copy=False)


def gather(x: np.ndarray, indices: np.ndarray, axis: int = 0) -> np.ndarray:
    safe_axis = axis if -x.ndim <= axis < x.ndim else 0
    return np.take(x, indices, axis=safe_axis).astype(np.float32, copy=False)


def one_hot(indices: np.ndarray, depth: int) -> np.ndarray:
    return np.eye(depth, dtype=np.float32)[indices]


def upsample_nearest2d(x: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    return np.repeat(np.repeat(x.astype(np.float32, copy=False), scale_factor, axis=2), scale_factor, axis=3).astype(
        np.float32,
        copy=False,
    )


def pad(x: np.ndarray, padding: int = 1) -> np.ndarray:
    if x.ndim == 4:
        pads = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    else:
        pads = tuple((padding, padding) for _ in range(x.ndim))
    return np.pad(x.astype(np.float32, copy=False), pads, mode="constant").astype(np.float32, copy=False)


def channel_shuffle(x: np.ndarray, groups: int = 2) -> np.ndarray:
    n, c, h, w = x.shape
    groups = groups if c % groups == 0 else 1
    return x.reshape(n, groups, c // groups, h, w).transpose(0, 2, 1, 3, 4).reshape(n, c, h, w).astype(
        np.float32,
        copy=False,
    )


def space_to_depth(x: np.ndarray, block_size: int = 2) -> np.ndarray:
    n, c, h, w = x.shape
    block_size = min(block_size, h, w)
    h_trim = h - (h % block_size)
    w_trim = w - (w % block_size)
    x = x[:, :, :h_trim, :w_trim]
    return x.reshape(n, c, h_trim // block_size, block_size, w_trim // block_size, block_size).transpose(
        0, 1, 3, 5, 2, 4
    ).reshape(n, c * block_size * block_size, h_trim // block_size, w_trim // block_size).astype(np.float32, copy=False)


def depth_to_space(x: np.ndarray, block_size: int = 2) -> np.ndarray:
    n, c, h, w = x.shape
    block_area = block_size * block_size
    if c % block_area != 0:
        block_size = 1
        block_area = 1
    return x.reshape(n, c // block_area, block_size, block_size, h, w).transpose(0, 1, 4, 2, 5, 3).reshape(
        n,
        c // block_area,
        h * block_size,
        w * block_size,
    ).astype(np.float32, copy=False)


def reduce_mean(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.mean(x.astype(np.float32, copy=False), axis=axis).astype(np.float32, copy=False)


def reduce_sum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sum(x.astype(np.float32, copy=False), axis=axis).astype(np.float32, copy=False)


def reduce_max(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.max(x.astype(np.float32, copy=False), axis=axis).astype(np.float32, copy=False)


def reduce_min(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.min(x.astype(np.float32, copy=False), axis=axis).astype(np.float32, copy=False)


def reduce_prod(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.prod(x.astype(np.float32, copy=False), axis=axis).astype(np.float32, copy=False)


def identity(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=True)


def abs_tensor(x: np.ndarray) -> np.ndarray:
    return np.abs(x).astype(np.float32, copy=False)


def neg(x: np.ndarray) -> np.ndarray:
    return (-x).astype(np.float32, copy=False)


def square(x: np.ndarray) -> np.ndarray:
    return np.square(x).astype(np.float32, copy=False)


def sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(x) + 1e-6).astype(np.float32, copy=False)


def rsqrt(x: np.ndarray) -> np.ndarray:
    return (1.0 / sqrt(x)).astype(np.float32, copy=False)


def exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, -20, 20)).astype(np.float32, copy=False)


def log(x: np.ndarray) -> np.ndarray:
    return np.log(np.abs(x) + 1e-6).astype(np.float32, copy=False)


def log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(x)).astype(np.float32, copy=False)


def pow_tensor(x: np.ndarray, exponent: float = 2.0) -> np.ndarray:
    return np.power(np.abs(x) + 1e-6, exponent).astype(np.float32, copy=False)


def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x).astype(np.float32, copy=False)


def cos(x: np.ndarray) -> np.ndarray:
    return np.cos(x).astype(np.float32, copy=False)


def reciprocal(x: np.ndarray) -> np.ndarray:
    return (1.0 / (x + np.sign(x) * 1e-3 + (x == 0) * 1e-3)).astype(np.float32, copy=False)


def floor(x: np.ndarray) -> np.ndarray:
    return np.floor(x).astype(np.float32, copy=False)


def ceil(x: np.ndarray) -> np.ndarray:
    return np.ceil(x).astype(np.float32, copy=False)


def round_tensor(x: np.ndarray) -> np.ndarray:
    return np.round(x).astype(np.float32, copy=False)


def clip(x: np.ndarray, min_value: float = -1.0, max_value: float = 1.0) -> np.ndarray:
    return np.clip(x, min_value, max_value).astype(np.float32, copy=False)


def sign(x: np.ndarray) -> np.ndarray:
    return np.sign(x).astype(np.float32, copy=False)


def standardize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32, copy=False)


def minmax_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    min_value = np.min(x, axis=axis, keepdims=True)
    max_value = np.max(x, axis=axis, keepdims=True)
    return ((x - min_value) / (max_value - min_value + eps)).astype(np.float32, copy=False)


def pixel_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    axis = 1 if x.ndim == 4 else -1
    return (x / np.sqrt(np.mean(np.square(x), axis=axis, keepdims=True) + eps)).astype(np.float32, copy=False)


def dropout_inference(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=True)


def where(mask: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(mask, a, b).astype(np.float32, copy=False)


def masked_fill(x: np.ndarray, mask: np.ndarray, value: float = 0.0) -> np.ndarray:
    return np.where(mask, value, x).astype(np.float32, copy=False)


def greater(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a > b).astype(np.float32)


def less(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a < b).astype(np.float32)


def equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isclose(a, b, atol=1e-3).astype(np.float32)


def not_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.logical_not(np.isclose(a, b, atol=1e-3)).astype(np.float32)


def argmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.argmax(x, axis=axis).astype(np.float32)


def argmin(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.argmin(x, axis=axis).astype(np.float32)


def topk(x: np.ndarray, k: int = 3, axis: int = -1) -> np.ndarray:
    axis_len = x.shape[axis]
    k = max(1, min(k, axis_len))
    partitioned = np.partition(x, axis_len - k, axis=axis)
    top = np.take(partitioned, indices=range(axis_len - k, axis_len), axis=axis)
    return np.flip(np.sort(top, axis=axis), axis=axis).astype(np.float32, copy=False)


def sort(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sort(x, axis=axis).astype(np.float32, copy=False)


def cumsum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.cumsum(x, axis=axis).astype(np.float32, copy=False)


def cumprod(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.cumprod(np.clip(x, -2.0, 2.0), axis=axis).astype(np.float32, copy=False)


def adaptive_avgpool2d(x: np.ndarray, output_size: int = 2) -> np.ndarray:
    return _adaptive_pool2d(x, output_size, reducer=np.mean)


def adaptive_maxpool2d(x: np.ndarray, output_size: int = 2) -> np.ndarray:
    return _adaptive_pool2d(x, output_size, reducer=np.max)


def cosine_similarity(a: np.ndarray, b: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    numerator = np.sum(a * b, axis=axis)
    denominator = np.sqrt(np.sum(a * a, axis=axis) * np.sum(b * b, axis=axis)) + eps
    return (numerator / denominator).astype(np.float32, copy=False)


def pairwise_distance(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.sum(np.square(a - b), axis=axis) + 1e-8).astype(np.float32, copy=False)


def glu(x: np.ndarray) -> np.ndarray:
    a, b = _split_last_dim(x)
    return (a * sigmoid(b)).astype(np.float32, copy=False)


def swiglu(x: np.ndarray) -> np.ndarray:
    a, b = _split_last_dim(x)
    return (a * silu(b)).astype(np.float32, copy=False)


def geglu(x: np.ndarray) -> np.ndarray:
    a, b = _split_last_dim(x)
    return (a * gelu(b)).astype(np.float32, copy=False)


def embedding(indices: np.ndarray, table: np.ndarray) -> np.ndarray:
    return table[indices].astype(np.float32, copy=False)


def scaled_dot_product_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scale = np.sqrt(q.shape[-1]).astype(np.float32) if hasattr(np.sqrt(q.shape[-1]), "astype") else np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / scale
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v).astype(np.float32, copy=False)


def causal_self_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scale = np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / scale
    seq = scores.shape[-1]
    mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v).astype(np.float32, copy=False)


def rotary_embedding(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    dim = x.shape[-1]
    if dim < 2:
        return x.copy()
    even = x[..., 0::2]
    odd = x[..., 1::2]
    pair_dim = min(even.shape[-1], odd.shape[-1])
    positions = np.arange(x.shape[-2], dtype=np.float32)[..., None]
    inv_freq = 1.0 / (10000 ** (np.arange(pair_dim, dtype=np.float32) / max(1, pair_dim)))
    angles = positions * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)
    out = x.copy()
    out[..., 0 : pair_dim * 2 : 2] = even[..., :pair_dim] * cos - odd[..., :pair_dim] * sin
    out[..., 1 : pair_dim * 2 : 2] = even[..., :pair_dim] * sin + odd[..., :pair_dim] * cos
    return out.astype(np.float32, copy=False)


def quantize_symmetric_int8(values: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs == 0.0:
        return np.zeros_like(values, dtype=np.int8), 1.0
    scale = max_abs / 127.0
    quantized = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    return quantized, scale


def quantize_to_powers_of_two(values: np.ndarray, min_exp: int = -8, max_exp: int = 7) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    out = np.zeros_like(values, dtype=np.float32)
    mask = values != 0
    exponents = np.clip(np.round(np.log2(np.abs(values[mask]))), min_exp, max_exp)
    out[mask] = np.sign(values[mask]) * np.power(2.0, exponents)
    return out


def _conv_windows(x: np.ndarray, kernel_size: tuple[int, int], stride: int, padding: int) -> np.ndarray:
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    windows = sliding_window_view(x, kernel_size, axis=(2, 3))
    return windows[:, :, ::stride, ::stride, :, :]


def _pool_windows(x: np.ndarray, kernel_size: tuple[int, int], stride: int) -> np.ndarray:
    windows = sliding_window_view(x, kernel_size, axis=(2, 3))
    return windows[:, :, ::stride, ::stride, :, :]


def _adaptive_pool2d(x: np.ndarray, output_size: int, reducer) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n, c, h, w = x.shape
    output_size = max(1, min(output_size, h, w))
    out = np.empty((n, c, output_size, output_size), dtype=np.float32)
    for i in range(output_size):
        h0 = int(np.floor(i * h / output_size))
        h1 = int(np.ceil((i + 1) * h / output_size))
        for j in range(output_size):
            w0 = int(np.floor(j * w / output_size))
            w1 = int(np.ceil((j + 1) * w / output_size))
            out[:, :, i, j] = reducer(x[:, :, h0:h1, w0:w1], axis=(2, 3))
    return out


def _split_last_dim(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    split = max(1, x.shape[-1] // 2)
    if x.shape[-1] < 2:
        return x, x
    return x[..., :split], x[..., split : split * 2]

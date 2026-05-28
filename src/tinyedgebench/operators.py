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


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32, copy=False) + b.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32, copy=False) * b.astype(np.float32, copy=False)).astype(np.float32, copy=False)


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


def reduce_mean(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.mean(x.astype(np.float32, copy=False), axis=axis).astype(np.float32, copy=False)


def reduce_sum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sum(x.astype(np.float32, copy=False), axis=axis).astype(np.float32, copy=False)


def embedding(indices: np.ndarray, table: np.ndarray) -> np.ndarray:
    return table[indices].astype(np.float32, copy=False)


def scaled_dot_product_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scale = np.sqrt(q.shape[-1]).astype(np.float32) if hasattr(np.sqrt(q.shape[-1]), "astype") else np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / scale
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v).astype(np.float32, copy=False)


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

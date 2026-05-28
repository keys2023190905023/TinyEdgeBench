from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from tinyedgebench.config import CONV_OPERATORS, MATRIX_OPERATORS, BenchmarkCase, BenchmarkConfig
from tinyedgebench import operators


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    operator: str
    precision: str
    backend: str
    input_description: str
    latency_ms: float
    throughput_ops_per_s: float
    mean_abs_error: float
    max_abs_error: float


def run_benchmarks(config: BenchmarkConfig) -> list[BenchmarkResult]:
    rng = np.random.default_rng(config.seed)
    results: list[BenchmarkResult] = []
    for case in config.benchmarks:
        inputs = _make_inputs(case, rng)
        reference = _run_case(case, "fp32", inputs)
        estimated_ops = _estimate_ops(case)
        for precision in case.precision_modes:
            for _ in range(config.warmup):
                _run_case(case, precision, inputs)
            timings = []
            output = reference
            for _ in range(config.runs):
                start = time.perf_counter()
                output = _run_case(case, precision, inputs)
                timings.append((time.perf_counter() - start) * 1000.0)
            latency_ms = float(np.median(timings))
            error = np.abs(output - reference)
            throughput = estimated_ops / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
            results.append(
                BenchmarkResult(
                    name=case.name,
                    operator=case.operator,
                    precision=precision,
                    backend=config.backend,
                    input_description=_describe_case(case),
                    latency_ms=latency_ms,
                    throughput_ops_per_s=float(throughput),
                    mean_abs_error=float(error.mean()),
                    max_abs_error=float(error.max()),
                )
            )
    return results


def _make_inputs(case: BenchmarkCase, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if case.operator in {"matmul", "linear"}:
        assert case.matrix_m and case.matrix_k and case.matrix_n
        inputs = {
            "a": rng.normal(0, 0.5, size=(case.matrix_m, case.matrix_k)).astype(np.float32),
            "b": rng.normal(0, 0.5, size=(case.matrix_k, case.matrix_n)).astype(np.float32),
        }
        if case.operator == "linear":
            inputs["bias"] = rng.normal(0, 0.05, size=(case.matrix_n,)).astype(np.float32)
        return inputs

    if case.operator == "batch_matmul":
        assert case.batch_size and case.matrix_m and case.matrix_k and case.matrix_n
        return {
            "a": rng.normal(0, 0.5, size=(case.batch_size, case.matrix_m, case.matrix_k)).astype(np.float32),
            "b": rng.normal(0, 0.5, size=(case.batch_size, case.matrix_k, case.matrix_n)).astype(np.float32),
        }

    if case.operator in CONV_OPERATORS:
        assert case.input_shape and case.kernel_size
        x = rng.normal(0, 0.5, size=case.input_shape).astype(np.float32)
        channels = case.input_shape[1]
        if case.operator == "depthwise_conv2d":
            weight_shape = (channels, *case.kernel_size)
            bias_shape = (channels,)
        else:
            assert case.output_channels
            weight_shape = (case.output_channels, channels, *case.kernel_size)
            bias_shape = (case.output_channels,)
        return {
            "x": x,
            "weight": rng.normal(0, 0.5, size=weight_shape).astype(np.float32),
            "bias": rng.normal(0, 0.05, size=bias_shape).astype(np.float32),
        }

    if case.operator == "embedding":
        assert case.batch_size and case.sequence_length and case.vocab_size and case.embedding_dim
        return {
            "indices": rng.integers(0, case.vocab_size, size=(case.batch_size, case.sequence_length)),
            "table": rng.normal(0, 0.5, size=(case.vocab_size, case.embedding_dim)).astype(np.float32),
        }

    if case.operator in {"scaled_dot_product_attention", "causal_self_attention"}:
        assert case.batch_size and case.sequence_length and case.embedding_dim and case.num_heads
        head_dim = max(1, case.embedding_dim // case.num_heads)
        shape = (case.batch_size, case.num_heads, case.sequence_length, head_dim)
        return {
            "q": rng.normal(0, 0.5, size=shape).astype(np.float32),
            "k": rng.normal(0, 0.5, size=shape).astype(np.float32),
            "v": rng.normal(0, 0.5, size=shape).astype(np.float32),
        }

    assert case.input_shape_generic
    x = rng.normal(0, 0.5, size=case.input_shape_generic).astype(np.float32)
    inputs = {"x": x}
    if case.operator in {"add", "sub", "mul", "div", "maximum", "minimum", "concat"}:
        inputs["y"] = rng.normal(0, 0.5, size=case.input_shape_generic).astype(np.float32)
    if case.operator in {"bias_add", "prelu"}:
        if len(case.input_shape_generic) == 4:
            channels = case.input_shape_generic[1]
            shape = (channels,)
        else:
            shape = (case.input_shape_generic[-1],)
        inputs["bias"] = rng.normal(0.05, 0.02, size=shape).astype(np.float32)
    if case.operator == "gather":
        axis_len = case.input_shape_generic[0]
        inputs["indices"] = rng.integers(0, axis_len, size=(max(1, axis_len // 2),))
    if case.operator == "one_hot":
        inputs["indices"] = rng.integers(0, case.input_shape_generic[-1], size=case.input_shape_generic[:-1] or (1,))
    if case.operator == "batchnorm2d":
        channels = case.input_shape_generic[1]
        inputs.update(
            {
                "gamma": rng.normal(1, 0.05, size=(channels,)).astype(np.float32),
                "beta": rng.normal(0, 0.05, size=(channels,)).astype(np.float32),
                "running_mean": rng.normal(0, 0.1, size=(channels,)).astype(np.float32),
                "running_var": np.abs(rng.normal(1, 0.1, size=(channels,))).astype(np.float32),
            }
        )
    if case.operator in {"groupnorm", "instance_norm"}:
        channels = case.input_shape_generic[1]
        inputs["gamma"] = rng.normal(1, 0.05, size=(channels,)).astype(np.float32)
        inputs["beta"] = rng.normal(0, 0.05, size=(channels,)).astype(np.float32)
    if case.operator in {"layernorm", "rmsnorm"}:
        features = case.input_shape_generic[-1]
        inputs["gamma"] = rng.normal(1, 0.05, size=(features,)).astype(np.float32)
        if case.operator == "layernorm":
            inputs["beta"] = rng.normal(0, 0.05, size=(features,)).astype(np.float32)
    return inputs


def _run_case(case: BenchmarkCase, precision: str, inputs: dict[str, np.ndarray]) -> np.ndarray:
    if case.operator == "matmul":
        if precision == "fp32":
            return operators.matmul_fp32(inputs["a"], inputs["b"])
        if precision == "int8_sim":
            return operators.matmul_int8_sim(inputs["a"], inputs["b"])
        if precision == "shift_only":
            return operators.matmul_shift_only(inputs["a"], inputs["b"])

    if case.operator == "batch_matmul":
        if precision == "fp32":
            return operators.batch_matmul_fp32(inputs["a"], inputs["b"])
        if precision == "int8_sim":
            return operators.batch_matmul_int8_sim(inputs["a"], inputs["b"])
        if precision == "shift_only":
            return operators.batch_matmul_shift_only(inputs["a"], inputs["b"])

    if case.operator == "linear":
        if precision == "fp32":
            return operators.linear_fp32(inputs["a"], inputs["b"], inputs["bias"])
        if precision == "int8_sim":
            return operators.linear_int8_sim(inputs["a"], inputs["b"], inputs["bias"])
        if precision == "shift_only":
            return operators.linear_shift_only(inputs["a"], inputs["b"], inputs["bias"])

    if case.operator == "conv2d":
        kwargs = {"stride": case.stride, "padding": case.padding}
        if precision == "fp32":
            return operators.conv2d_fp32(inputs["x"], inputs["weight"], inputs["bias"], **kwargs)
        if precision == "int8_sim":
            return operators.conv2d_int8_sim(inputs["x"], inputs["weight"], inputs["bias"], **kwargs)
        if precision == "shift_only":
            return operators.conv2d_shift_only(inputs["x"], inputs["weight"], inputs["bias"], **kwargs)

    if case.operator == "pointwise_conv2d":
        kwargs = {"stride": case.stride, "padding": case.padding}
        if precision == "fp32":
            return operators.conv2d_fp32(inputs["x"], inputs["weight"], inputs["bias"], **kwargs)
        if precision == "int8_sim":
            return operators.conv2d_int8_sim(inputs["x"], inputs["weight"], inputs["bias"], **kwargs)
        if precision == "shift_only":
            return operators.conv2d_shift_only(inputs["x"], inputs["weight"], inputs["bias"], **kwargs)

    if case.operator == "depthwise_conv2d":
        kwargs = {"stride": case.stride, "padding": case.padding}
        if precision == "fp32":
            return operators.depthwise_conv2d_fp32(inputs["x"], inputs["weight"], inputs["bias"], **kwargs)
        if precision == "int8_sim":
            return operators.depthwise_conv2d_int8_sim(inputs["x"], inputs["weight"], inputs["bias"], **kwargs)
        if precision == "shift_only":
            return operators.depthwise_conv2d_shift_only(inputs["x"], inputs["weight"], inputs["bias"], **kwargs)

    return _run_generic_case(case, _precision_inputs(inputs, precision))


def _estimate_ops(case: BenchmarkCase) -> int:
    if case.operator in {"matmul", "linear"}:
        assert case.matrix_m and case.matrix_k and case.matrix_n
        return 2 * case.matrix_m * case.matrix_k * case.matrix_n
    if case.operator == "batch_matmul":
        assert case.batch_size and case.matrix_m and case.matrix_k and case.matrix_n
        return 2 * case.batch_size * case.matrix_m * case.matrix_k * case.matrix_n

    if case.operator in CONV_OPERATORS:
        assert case.input_shape and case.kernel_size
        batch, channels, height, width = case.input_shape
        kernel_h, kernel_w = case.kernel_size
        out_h = ((height + 2 * case.padding - kernel_h) // case.stride) + 1
        out_w = ((width + 2 * case.padding - kernel_w) // case.stride) + 1
        if case.operator == "depthwise_conv2d":
            return 2 * batch * channels * out_h * out_w * kernel_h * kernel_w
        assert case.output_channels
        return 2 * batch * case.output_channels * out_h * out_w * channels * kernel_h * kernel_w

    if case.operator == "embedding":
        assert case.batch_size and case.sequence_length and case.embedding_dim
        return case.batch_size * case.sequence_length * case.embedding_dim
    if case.operator in {"scaled_dot_product_attention", "causal_self_attention"}:
        assert case.batch_size and case.sequence_length and case.embedding_dim and case.num_heads
        head_dim = max(1, case.embedding_dim // case.num_heads)
        return 4 * case.batch_size * case.num_heads * case.sequence_length * case.sequence_length * head_dim

    assert case.input_shape_generic
    return max(1, _numel(case.input_shape_generic))


def _describe_case(case: BenchmarkCase) -> str:
    if case.operator in {"matmul", "linear"}:
        return f"{case.matrix_m}x{case.matrix_k} @ {case.matrix_k}x{case.matrix_n}"
    if case.operator == "batch_matmul":
        return f"batch={case.batch_size}, {case.matrix_m}x{case.matrix_k} @ {case.matrix_k}x{case.matrix_n}"
    if case.operator in CONV_OPERATORS:
        assert case.input_shape and case.kernel_size
        if case.operator == "depthwise_conv2d":
            return f"NCHW={case.input_shape}, kernel={case.kernel_size}, stride={case.stride}, padding={case.padding}"
        return (
            f"NCHW={case.input_shape}, out_channels={case.output_channels}, "
            f"kernel={case.kernel_size}, stride={case.stride}, padding={case.padding}"
        )
    if case.operator == "embedding":
        return (
            f"batch={case.batch_size}, seq={case.sequence_length}, "
            f"vocab={case.vocab_size}, dim={case.embedding_dim}"
        )
    if case.operator in {"scaled_dot_product_attention", "causal_self_attention"}:
        return (
            f"batch={case.batch_size}, heads={case.num_heads}, "
            f"seq={case.sequence_length}, dim={case.embedding_dim}"
        )
    return f"shape={case.input_shape_generic}"


def _run_generic_case(case: BenchmarkCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = inputs.get("x")
    if case.operator == "relu":
        return operators.relu(x)
    if case.operator == "relu6":
        return operators.relu6(x)
    if case.operator == "sigmoid":
        return operators.sigmoid(x)
    if case.operator == "tanh":
        return operators.tanh(x)
    if case.operator == "gelu":
        return operators.gelu(x)
    if case.operator == "silu":
        return operators.silu(x)
    if case.operator == "leaky_relu":
        return operators.leaky_relu(x)
    if case.operator == "elu":
        return operators.elu(x)
    if case.operator == "selu":
        return operators.selu(x)
    if case.operator == "celu":
        return operators.celu(x)
    if case.operator == "softplus":
        return operators.softplus(x)
    if case.operator == "softsign":
        return operators.softsign(x)
    if case.operator == "hard_sigmoid":
        return operators.hard_sigmoid(x)
    if case.operator == "hard_swish":
        return operators.hard_swish(x)
    if case.operator == "mish":
        return operators.mish(x)
    if case.operator == "prelu":
        return operators.prelu(inputs["x"], inputs["bias"])
    if case.operator == "softmax":
        return operators.softmax(x, axis=case.axis)
    if case.operator == "log_softmax":
        return operators.log_softmax(x, axis=case.axis)
    if case.operator == "maxpool2d":
        assert case.kernel_size
        return operators.maxpool2d(x, case.kernel_size, case.stride)
    if case.operator == "avgpool2d":
        assert case.kernel_size
        return operators.avgpool2d(x, case.kernel_size, case.stride)
    if case.operator == "global_avgpool2d":
        return operators.global_avgpool2d(x)
    if case.operator == "batchnorm2d":
        return operators.batchnorm2d(inputs["x"], inputs["gamma"], inputs["beta"], inputs["running_mean"], inputs["running_var"])
    if case.operator == "layernorm":
        return operators.layernorm(inputs["x"], inputs["gamma"], inputs["beta"])
    if case.operator == "rmsnorm":
        return operators.rmsnorm(inputs["x"], inputs["gamma"])
    if case.operator == "groupnorm":
        return operators.groupnorm(inputs["x"], inputs["gamma"], inputs["beta"], groups=case.groups)
    if case.operator == "instance_norm":
        return operators.instance_norm(inputs["x"], inputs["gamma"], inputs["beta"])
    if case.operator == "l2_normalize":
        return operators.l2_normalize(x, case.axis)
    if case.operator == "add":
        return operators.add(inputs["x"], inputs["y"])
    if case.operator == "sub":
        return operators.sub(inputs["x"], inputs["y"])
    if case.operator == "mul":
        return operators.mul(inputs["x"], inputs["y"])
    if case.operator == "div":
        return operators.div(inputs["x"], inputs["y"])
    if case.operator == "maximum":
        return operators.maximum(inputs["x"], inputs["y"])
    if case.operator == "minimum":
        return operators.minimum(inputs["x"], inputs["y"])
    if case.operator == "bias_add":
        return operators.bias_add(inputs["x"], inputs["bias"])
    if case.operator == "concat":
        return operators.concat(inputs["x"], inputs["y"], axis=case.axis)
    if case.operator == "transpose":
        return operators.transpose(x)
    if case.operator == "reshape":
        assert case.target_shape
        return operators.reshape(x, case.target_shape)
    if case.operator == "flatten":
        return operators.flatten(x)
    if case.operator == "squeeze":
        return operators.squeeze(x)
    if case.operator == "expand_dims":
        return operators.expand_dims(x, case.axis)
    if case.operator == "tile":
        return operators.tile(x, case.scale_factor)
    if case.operator == "slice":
        return operators.slice_tensor(x)
    if case.operator == "gather":
        return operators.gather(x, inputs["indices"], axis=0)
    if case.operator == "one_hot":
        return operators.one_hot(inputs["indices"], depth=case.input_shape_generic[-1])
    if case.operator == "upsample_nearest2d":
        return operators.upsample_nearest2d(x, case.scale_factor)
    if case.operator == "pad":
        return operators.pad(x, case.padding)
    if case.operator == "channel_shuffle":
        return operators.channel_shuffle(x, case.groups)
    if case.operator == "space_to_depth":
        return operators.space_to_depth(x, case.scale_factor)
    if case.operator == "depth_to_space":
        return operators.depth_to_space(x, case.scale_factor)
    if case.operator == "reduce_mean":
        return operators.reduce_mean(x, case.axis)
    if case.operator == "reduce_sum":
        return operators.reduce_sum(x, case.axis)
    if case.operator == "reduce_max":
        return operators.reduce_max(x, case.axis)
    if case.operator == "reduce_min":
        return operators.reduce_min(x, case.axis)
    if case.operator == "reduce_prod":
        return operators.reduce_prod(x, case.axis)
    if case.operator == "identity":
        return operators.identity(x)
    if case.operator == "abs":
        return operators.abs_tensor(x)
    if case.operator == "neg":
        return operators.neg(x)
    if case.operator == "square":
        return operators.square(x)
    if case.operator == "sqrt":
        return operators.sqrt(x)
    if case.operator == "rsqrt":
        return operators.rsqrt(x)
    if case.operator == "exp":
        return operators.exp(x)
    if case.operator == "log":
        return operators.log(x)
    if case.operator == "reciprocal":
        return operators.reciprocal(x)
    if case.operator == "floor":
        return operators.floor(x)
    if case.operator == "ceil":
        return operators.ceil(x)
    if case.operator == "round":
        return operators.round_tensor(x)
    if case.operator == "clip":
        return operators.clip(x)
    if case.operator == "sign":
        return operators.sign(x)
    if case.operator == "dropout_inference":
        return operators.dropout_inference(x)
    if case.operator == "embedding":
        return operators.embedding(inputs["indices"], inputs["table"])
    if case.operator == "scaled_dot_product_attention":
        return operators.scaled_dot_product_attention(inputs["q"], inputs["k"], inputs["v"])
    if case.operator == "causal_self_attention":
        return operators.causal_self_attention(inputs["q"], inputs["k"], inputs["v"])
    if case.operator == "rotary_embedding":
        return operators.rotary_embedding(x)
    raise ValueError(f"Unsupported benchmark operator: {case.operator}")


def _precision_inputs(inputs: dict[str, np.ndarray], precision: str) -> dict[str, np.ndarray]:
    if precision == "fp32":
        return inputs
    transformed = {}
    for key, value in inputs.items():
        if not np.issubdtype(value.dtype, np.floating):
            transformed[key] = value
        elif precision == "int8_sim":
            transformed[key] = operators.dequantize_symmetric_int8(value)
        elif precision == "shift_only":
            transformed[key] = operators.quantize_to_powers_of_two(value)
        else:
            raise ValueError(f"Unsupported precision: {precision}")
    return transformed


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for item in shape:
        total *= item
    return total

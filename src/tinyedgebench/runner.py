from __future__ import annotations

import time
from dataclasses import dataclass
import subprocess

import numpy as np

from tinyedgebench.config import CONV_OPERATORS, MATRIX_OPERATORS, BenchmarkCase, BenchmarkConfig
from tinyedgebench import operators
from tinyedgebench.real_backends import (
    build_onnxruntime_executor,
    build_torch_executor,
    supports_onnxruntime_backend,
    supports_real_backend,
)


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
    latency_median_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_std_ms: float = 0.0
    valid_runs: int = 0
    failed_runs: int = 0
    oom_runs: int = 0
    peak_memory_mb: float | None = None
    gpu_memory_allocated_mb: float | None = None
    gpu_memory_reserved_mb: float | None = None
    power_w: float | None = None
    energy_mj: float | None = None
    edp_mj_ms: float | None = None
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.latency_median_ms == 0.0:
            object.__setattr__(self, "latency_median_ms", self.latency_ms)
        if self.latency_p90_ms == 0.0:
            object.__setattr__(self, "latency_p90_ms", self.latency_ms)
        if self.valid_runs == 0:
            object.__setattr__(self, "valid_runs", 1)
        if self.inference_ms == 0.0:
            object.__setattr__(self, "inference_ms", self.latency_ms)


def run_benchmarks(config: BenchmarkConfig) -> list[BenchmarkResult]:
    rng = np.random.default_rng(config.seed)
    results: list[BenchmarkResult] = []
    for case in config.benchmarks:
        inputs = _make_inputs(case, rng)
        reference = _run_case(case, "fp32", inputs, backend="cpu")
        estimated_ops = _estimate_ops(case)
        for backend in getattr(config, "backends", (config.backend,)):
            for precision in case.precision_modes:
                executor = _build_executor(case, precision, inputs, backend)
                for _ in range(config.warmup):
                    executor()
                _reset_torch_cuda_stats(backend)
                process = _process()
                peak_memory_mb = _rss_mb(process)
                timings = []
                output = reference
                power_samples = []
                for _ in range(config.runs):
                    power_before = _nvidia_smi_power_w(backend)
                    _synchronize_executor(executor)
                    start = time.perf_counter()
                    output = executor()
                    _synchronize_executor(executor)
                    timings.append((time.perf_counter() - start) * 1000.0)
                    power_after = _nvidia_smi_power_w(backend)
                    peak_memory_mb = _max_optional(peak_memory_mb, _rss_mb(process))
                    if power_before is not None:
                        power_samples.append(power_before)
                    if power_after is not None:
                        power_samples.append(power_after)
                latency_ms = float(np.median(timings))
                latency_p90_ms = float(np.percentile(timings, 90))
                latency_std_ms = float(np.std(timings))
                error = np.abs(output - reference)
                throughput = estimated_ops / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
                power_w = float(np.mean(power_samples)) if power_samples else None
                energy_mj = (power_w * latency_ms) if power_w is not None else None
                results.append(
                    BenchmarkResult(
                        name=case.name,
                        operator=case.operator,
                        precision=precision,
                        backend=backend,
                        input_description=_describe_case(case),
                        latency_ms=latency_ms,
                        throughput_ops_per_s=float(throughput),
                        mean_abs_error=float(error.mean()),
                        max_abs_error=float(error.max()),
                        latency_median_ms=latency_ms,
                        latency_p90_ms=latency_p90_ms,
                        latency_std_ms=latency_std_ms,
                        valid_runs=len(timings),
                        peak_memory_mb=peak_memory_mb,
                        gpu_memory_allocated_mb=_torch_cuda_memory_mb(backend, "allocated"),
                        gpu_memory_reserved_mb=_torch_cuda_memory_mb(backend, "reserved"),
                        power_w=power_w,
                        energy_mj=energy_mj,
                        edp_mj_ms=(energy_mj * latency_ms) if energy_mj is not None else None,
                        inference_ms=latency_ms,
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
    if case.operator in {
        "add",
        "sub",
        "mul",
        "div",
        "maximum",
        "minimum",
        "concat",
        "where",
        "greater",
        "less",
        "equal",
        "not_equal",
        "cosine_similarity",
        "pairwise_distance",
    }:
        inputs["y"] = rng.normal(0, 0.5, size=case.input_shape_generic).astype(np.float32)
    if case.operator in {"where", "masked_fill"}:
        inputs["mask"] = rng.random(size=case.input_shape_generic) > 0.5
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


def _build_executor(case: BenchmarkCase, precision: str, inputs: dict[str, np.ndarray], backend: str):
    if backend in {"cpu", "numpy_cpu"}:
        return lambda: _run_case(case, precision, inputs, backend="cpu")
    if precision != "fp32":
        raise ValueError(
            f"Backend '{backend}' currently reports real backend timings for fp32 only. "
            "Use backend 'cpu' for int8_sim and shift_only simulations."
        )
    if backend == "torch_cpu":
        if not supports_real_backend(case.operator):
            raise ValueError(f"Backend '{backend}' does not support operator '{case.operator}' yet.")
        return build_torch_executor(case, inputs, device="cpu")
    if backend == "torch_cuda":
        if not supports_real_backend(case.operator):
            raise ValueError(f"Backend '{backend}' does not support operator '{case.operator}' yet.")
        return build_torch_executor(case, inputs, device="cuda")
    if backend == "onnxruntime_cpu":
        if not supports_onnxruntime_backend(case.operator):
            raise ValueError(f"Backend '{backend}' does not support operator '{case.operator}' yet.")
        return build_onnxruntime_executor(case, inputs, provider="CPUExecutionProvider")
    if backend == "onnxruntime_cuda":
        if not supports_onnxruntime_backend(case.operator):
            raise ValueError(f"Backend '{backend}' does not support operator '{case.operator}' yet.")
        return build_onnxruntime_executor(case, inputs, provider="CUDAExecutionProvider")
    if backend == "onnxruntime_tensorrt":
        if not supports_onnxruntime_backend(case.operator):
            raise ValueError(f"Backend '{backend}' does not support operator '{case.operator}' yet.")
        return build_onnxruntime_executor(case, inputs, provider="TensorrtExecutionProvider")
    if backend in {"openvino_cpu", "openvino_npu", "tvm_cpu", "tvm_cuda", "tensorrt_cuda"}:
        raise RuntimeError(
            f"Backend '{backend}' is registered for deployment planning but does not have an executor yet. "
            "Use onnxruntime_tensorrt for TensorRT-provider experiments today, or add the runtime-specific executor."
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _run_case(case: BenchmarkCase, precision: str, inputs: dict[str, np.ndarray], backend: str = "cpu") -> np.ndarray:
    if backend not in {"cpu", "numpy_cpu"}:
        return _build_executor(case, precision, inputs, backend)()
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
    if case.operator == "log1p":
        return operators.log1p(x)
    if case.operator == "pow":
        return operators.pow_tensor(x)
    if case.operator == "sin":
        return operators.sin(x)
    if case.operator == "cos":
        return operators.cos(x)
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
    if case.operator == "standardize":
        return operators.standardize(x, case.axis)
    if case.operator == "minmax_normalize":
        return operators.minmax_normalize(x, case.axis)
    if case.operator == "pixel_norm":
        return operators.pixel_norm(x)
    if case.operator == "dropout_inference":
        return operators.dropout_inference(x)
    if case.operator == "where":
        return operators.where(inputs["mask"], inputs["x"], inputs["y"])
    if case.operator == "masked_fill":
        return operators.masked_fill(inputs["x"], inputs["mask"])
    if case.operator == "greater":
        return operators.greater(inputs["x"], inputs["y"])
    if case.operator == "less":
        return operators.less(inputs["x"], inputs["y"])
    if case.operator == "equal":
        return operators.equal(inputs["x"], inputs["y"])
    if case.operator == "not_equal":
        return operators.not_equal(inputs["x"], inputs["y"])
    if case.operator == "argmax":
        return operators.argmax(x, case.axis)
    if case.operator == "argmin":
        return operators.argmin(x, case.axis)
    if case.operator == "topk":
        return operators.topk(x, k=case.scale_factor + 1, axis=case.axis)
    if case.operator == "sort":
        return operators.sort(x, case.axis)
    if case.operator == "cumsum":
        return operators.cumsum(x, case.axis)
    if case.operator == "cumprod":
        return operators.cumprod(x, case.axis)
    if case.operator == "adaptive_avgpool2d":
        return operators.adaptive_avgpool2d(x, output_size=case.scale_factor)
    if case.operator == "adaptive_maxpool2d":
        return operators.adaptive_maxpool2d(x, output_size=case.scale_factor)
    if case.operator == "cosine_similarity":
        return operators.cosine_similarity(inputs["x"], inputs["y"], case.axis)
    if case.operator == "pairwise_distance":
        return operators.pairwise_distance(inputs["x"], inputs["y"], case.axis)
    if case.operator == "glu":
        return operators.glu(x)
    if case.operator == "swiglu":
        return operators.swiglu(x)
    if case.operator == "geglu":
        return operators.geglu(x)
    if case.operator == "embedding":
        return operators.embedding(inputs["indices"], inputs["table"])
    if case.operator == "scaled_dot_product_attention":
        return operators.scaled_dot_product_attention(inputs["q"], inputs["k"], inputs["v"])
    if case.operator == "causal_self_attention":
        return operators.causal_self_attention(inputs["q"], inputs["k"], inputs["v"])
    if case.operator == "rotary_embedding":
        return operators.rotary_embedding(x)
    raise ValueError(f"Unsupported benchmark operator: {case.operator}")


def _synchronize_executor(executor) -> None:
    synchronize = getattr(executor, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _process():
    try:
        import psutil

        return psutil.Process()
    except ImportError:
        return None


def _rss_mb(process) -> float | None:
    if process is None:
        return None
    try:
        return float(process.memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def _max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _reset_torch_cuda_stats(backend: str) -> None:
    if backend != "torch_cuda":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _torch_cuda_memory_mb(backend: str, kind: str) -> float | None:
    if backend != "torch_cuda":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if kind == "allocated":
            value = torch.cuda.max_memory_allocated()
        else:
            value = torch.cuda.max_memory_reserved()
        return float(value / (1024 * 1024))
    except Exception:
        return None


def _nvidia_smi_power_w(backend: str) -> float | None:
    if "cuda" not in backend and "tensorrt" not in backend:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
    try:
        return float(first)
    except ValueError:
        return None


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

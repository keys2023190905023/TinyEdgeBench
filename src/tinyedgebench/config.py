from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CONV_OPERATORS = {"conv2d", "depthwise_conv2d", "pointwise_conv2d"}
MATRIX_OPERATORS = {"matmul", "batch_matmul", "linear"}
TENSOR_OPERATORS = {
    "relu",
    "relu6",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "leaky_relu",
    "elu",
    "selu",
    "celu",
    "softplus",
    "softsign",
    "hard_sigmoid",
    "hard_swish",
    "mish",
    "prelu",
    "softmax",
    "log_softmax",
    "maxpool2d",
    "avgpool2d",
    "global_avgpool2d",
    "batchnorm2d",
    "layernorm",
    "rmsnorm",
    "groupnorm",
    "instance_norm",
    "l2_normalize",
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
    "minimum",
    "bias_add",
    "concat",
    "transpose",
    "reshape",
    "flatten",
    "squeeze",
    "expand_dims",
    "tile",
    "slice",
    "gather",
    "one_hot",
    "upsample_nearest2d",
    "pad",
    "channel_shuffle",
    "space_to_depth",
    "depth_to_space",
    "reduce_mean",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "identity",
    "abs",
    "neg",
    "square",
    "sqrt",
    "rsqrt",
    "exp",
    "log",
    "log1p",
    "pow",
    "sin",
    "cos",
    "reciprocal",
    "floor",
    "ceil",
    "round",
    "clip",
    "sign",
    "standardize",
    "minmax_normalize",
    "pixel_norm",
    "dropout_inference",
    "where",
    "masked_fill",
    "greater",
    "less",
    "equal",
    "not_equal",
    "argmax",
    "argmin",
    "topk",
    "sort",
    "cumsum",
    "cumprod",
    "adaptive_avgpool2d",
    "adaptive_maxpool2d",
    "cosine_similarity",
    "pairwise_distance",
    "glu",
    "swiglu",
    "geglu",
    "embedding",
    "scaled_dot_product_attention",
    "causal_self_attention",
    "rotary_embedding",
}
SUPPORTED_OPERATORS = CONV_OPERATORS | MATRIX_OPERATORS | TENSOR_OPERATORS
SUPPORTED_PRECISIONS = {"fp32", "int8_sim", "shift_only"}
SUPPORTED_BACKENDS = {
    "cpu",
    "numpy_cpu",
    "torch_cpu",
    "torch_cuda",
    "onnxruntime_cpu",
    "onnxruntime_cuda",
    "onnxruntime_tensorrt",
    "openvino_cpu",
    "tvm_cpu",
    "tvm_cuda",
    "tensorrt_cuda",
}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    operator: str
    precision_modes: list[str]
    input_shape: tuple[int, ...] | None = None
    output_channels: int | None = None
    kernel_size: tuple[int, int] | None = None
    stride: int = 1
    padding: int = 0
    matrix_m: int | None = None
    matrix_k: int | None = None
    matrix_n: int | None = None
    batch_size: int | None = None
    input_shape_generic: tuple[int, ...] | None = None
    axis: int = -1
    groups: int = 1
    target_shape: tuple[int, ...] | None = None
    scale_factor: int = 2
    vocab_size: int | None = None
    sequence_length: int | None = None
    embedding_dim: int | None = None
    num_heads: int = 1


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: Path = Path("results")
    warmup: int = 2
    runs: int = 5
    backend: str = "cpu"
    backends: tuple[str, ...] = ("cpu",)
    seed: int = 42
    benchmarks: list[BenchmarkCase] = field(default_factory=list)


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> BenchmarkConfig:
    raw_backends = raw.get("backends", [raw.get("backend", "cpu")])
    if isinstance(raw_backends, str):
        raw_backends = [raw_backends]
    backends = tuple(str(item).lower() for item in raw_backends)
    bad_backends = [backend for backend in backends if backend not in SUPPORTED_BACKENDS]
    if bad_backends:
        raise ValueError(f"Unsupported backend(s): {', '.join(bad_backends)}. CPU is supported by default.")
    backend = backends[0]

    warmup = _positive_int(raw.get("warmup", 2), "warmup", allow_zero=True)
    runs = _positive_int(raw.get("runs", 5), "runs")
    seed = int(raw.get("seed", 42))
    output_dir = Path(raw.get("output_dir", "results"))
    benchmarks = []
    for preset in raw.get("network_presets", []):
        if isinstance(preset, str):
            preset_name = preset
            preset_modes = None
        else:
            preset_name = str(preset.get("name"))
            preset_modes = [str(mode).lower() for mode in preset.get("precision_modes", ["fp32", "int8_sim", "shift_only"])]
        from tinyedgebench.network_presets import build_network_preset

        benchmarks.extend(build_network_preset(preset_name, preset_modes))
    benchmarks.extend(_parse_case(item, index) for index, item in enumerate(raw.get("benchmarks", []), start=1))
    if not benchmarks:
        raise ValueError("At least one benchmark case is required.")

    return BenchmarkConfig(
        output_dir=output_dir,
        warmup=warmup,
        runs=runs,
        backend=backend,
        backends=backends,
        seed=seed,
        benchmarks=benchmarks,
    )


def wizard_case_to_config(case: BenchmarkCase, output_dir: str | Path = "results") -> BenchmarkConfig:
    return BenchmarkConfig(output_dir=Path(output_dir), benchmarks=[case])


def _parse_case(raw: dict[str, Any], index: int) -> BenchmarkCase:
    operator = str(raw.get("operator", "")).lower()
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"Benchmark {index}: unsupported operator '{operator}'.")

    precision_modes = [str(mode).lower() for mode in raw.get("precision_modes", ["fp32"])]
    bad_modes = [mode for mode in precision_modes if mode not in SUPPORTED_PRECISIONS]
    if bad_modes:
        raise ValueError(f"Benchmark {index}: unsupported precision mode(s): {', '.join(bad_modes)}.")

    name = str(raw.get("name", f"{operator}_{index}"))
    stride = _positive_int(raw.get("stride", 1), "stride")
    padding = _non_negative_int(raw.get("padding", 0), "padding")

    if operator in CONV_OPERATORS:
        input_shape = _shape_tuple(raw.get("input_shape"), "input_shape", dims=4)
        default_kernel = [1, 1] if operator == "pointwise_conv2d" else [3, 3]
        kernel_size = _kernel_tuple(raw.get("kernel_size", default_kernel))
        output_channels = None
        if operator in {"conv2d", "pointwise_conv2d"}:
            output_channels = _positive_int(raw.get("output_channels"), "output_channels")
        return BenchmarkCase(
            name=name,
            operator=operator,
            precision_modes=precision_modes,
            input_shape=input_shape,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    if operator in MATRIX_OPERATORS:
        return BenchmarkCase(
            name=name,
            operator=operator,
            precision_modes=precision_modes,
            matrix_m=_positive_int(raw.get("matrix_m"), "matrix_m"),
            matrix_k=_positive_int(raw.get("matrix_k"), "matrix_k"),
            matrix_n=_positive_int(raw.get("matrix_n"), "matrix_n"),
            batch_size=_positive_int(raw.get("batch_size", 1), "batch_size"),
        )

    if operator == "embedding":
        return BenchmarkCase(
            name=name,
            operator=operator,
            precision_modes=precision_modes,
            vocab_size=_positive_int(raw.get("vocab_size", 128), "vocab_size"),
            sequence_length=_positive_int(raw.get("sequence_length", 16), "sequence_length"),
            embedding_dim=_positive_int(raw.get("embedding_dim", 32), "embedding_dim"),
            batch_size=_positive_int(raw.get("batch_size", 1), "batch_size"),
        )

    if operator in {"scaled_dot_product_attention", "causal_self_attention"}:
        return BenchmarkCase(
            name=name,
            operator=operator,
            precision_modes=precision_modes,
            batch_size=_positive_int(raw.get("batch_size", 1), "batch_size"),
            sequence_length=_positive_int(raw.get("sequence_length", 16), "sequence_length"),
            embedding_dim=_positive_int(raw.get("embedding_dim", 32), "embedding_dim"),
            num_heads=_positive_int(raw.get("num_heads", 1), "num_heads"),
        )

    input_shape_generic = _shape_tuple(raw.get("input_shape", [1, 16, 16, 16]), "input_shape", dims=None)
    target_shape = None
    if operator == "reshape":
        target_shape = _shape_tuple(raw.get("target_shape", input_shape_generic), "target_shape", dims=None)
        if _numel(target_shape) != _numel(input_shape_generic):
            raise ValueError("target_shape must have the same number of elements as input_shape.")
    return BenchmarkCase(
        name=name,
        operator=operator,
        precision_modes=precision_modes,
        input_shape_generic=input_shape_generic,
        kernel_size=_kernel_tuple(raw.get("kernel_size", [2, 2])) if operator in {"maxpool2d", "avgpool2d"} else None,
        stride=_positive_int(raw.get("stride", 2), "stride") if operator in {"maxpool2d", "avgpool2d"} else stride,
        padding=_non_negative_int(raw.get("padding", 1), "padding") if operator == "pad" else padding,
        axis=int(raw.get("axis", -1)),
        groups=_positive_int(raw.get("groups", 1), "groups"),
        target_shape=target_shape,
        scale_factor=_positive_int(raw.get("scale_factor", 2), "scale_factor"),
    )


def _shape_tuple(value: Any, label: str, dims: int | None) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or (dims is not None and len(value) != dims) or not value:
        expected = f"{dims} " if dims is not None else ""
        raise ValueError(f"{label} must contain {expected}positive integers.")
    shape = tuple(_positive_int(item, label) for item in value)
    return shape


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for item in shape:
        total *= item
    return total


def _kernel_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (_positive_int(value[0], "kernel_size"), _positive_int(value[1], "kernel_size"))
    raise ValueError("kernel_size must be an integer or two positive integers.")


def _positive_int(value: Any, label: str, allow_zero: bool = False) -> int:
    if value is None:
        raise ValueError(f"{label} is required.")
    converted = int(value)
    if allow_zero and converted == 0:
        return converted
    if converted <= 0:
        raise ValueError(f"{label} must be positive.")
    return converted


def _non_negative_int(value: Any, label: str) -> int:
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{label} must be non-negative.")
    return converted

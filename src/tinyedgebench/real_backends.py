from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np

from tinyedgebench.config import BenchmarkCase


REAL_BACKEND_OPERATORS = {
    "matmul",
    "batch_matmul",
    "linear",
    "conv2d",
    "pointwise_conv2d",
    "depthwise_conv2d",
    "relu",
    "relu6",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "softmax",
    "add",
    "sub",
    "mul",
    "div",
    "maxpool2d",
    "avgpool2d",
    "global_avgpool2d",
    "flatten",
}
ONNXRUNTIME_BACKEND_OPERATORS = {
    "matmul",
    "batch_matmul",
    "linear",
    "conv2d",
    "pointwise_conv2d",
    "depthwise_conv2d",
    "relu",
    "sigmoid",
    "tanh",
    "add",
    "sub",
    "mul",
    "div",
    "flatten",
}


def supports_real_backend(operator: str) -> bool:
    return operator in REAL_BACKEND_OPERATORS


def supports_onnxruntime_backend(operator: str) -> bool:
    return operator in ONNXRUNTIME_BACKEND_OPERATORS


def run_torch_cpu(case: BenchmarkCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    return build_torch_executor(case, inputs, device="cpu")()


def build_torch_executor(case: BenchmarkCase, inputs: dict[str, np.ndarray], device: str = "cpu") -> Callable[[], np.ndarray]:
    if not supports_real_backend(case.operator):
        raise ValueError(f"torch_{device} does not support operator '{case.operator}' yet.")
    torch = _torch()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "torch_cuda backend requires a CUDA-capable PyTorch install on this machine. "
            "Install a CUDA PyTorch build and run TinyEdgeBench locally."
        )

    tensors = {
        key: torch.from_numpy(value.astype(np.float32, copy=False)).to(device)
        for key, value in inputs.items()
        if np.issubdtype(value.dtype, np.floating)
    }

    def execute() -> np.ndarray:
        with torch.no_grad():
            output = _run_torch_case(case, tensors)
        return _to_numpy(output)

    if device == "cuda":
        execute.synchronize = torch.cuda.synchronize  # type: ignore[attr-defined]
    return execute


def _run_torch_case(case: BenchmarkCase, tensors: dict[str, Any]) -> Any:
    torch = _torch()
    if case.operator == "matmul":
        return torch.matmul(tensors["a"], tensors["b"])
    if case.operator == "batch_matmul":
        return torch.matmul(tensors["a"], tensors["b"])
    if case.operator == "linear":
        return torch.matmul(tensors["a"], tensors["b"]) + tensors["bias"]
    if case.operator in {"conv2d", "pointwise_conv2d"}:
        return torch.nn.functional.conv2d(
            tensors["x"],
            tensors["weight"],
            tensors["bias"],
            stride=case.stride,
            padding=case.padding,
        )
    if case.operator == "depthwise_conv2d":
        channels = tensors["x"].shape[1]
        return torch.nn.functional.conv2d(
            tensors["x"],
            tensors["weight"].reshape(channels, 1, *tensors["weight"].shape[-2:]),
            tensors["bias"],
            stride=case.stride,
            padding=case.padding,
            groups=channels,
        )
    if case.operator == "relu":
        return torch.relu(tensors["x"])
    if case.operator == "relu6":
        return torch.clamp(tensors["x"], 0, 6)
    if case.operator == "sigmoid":
        return torch.sigmoid(tensors["x"])
    if case.operator == "tanh":
        return torch.tanh(tensors["x"])
    if case.operator == "gelu":
        return torch.nn.functional.gelu(tensors["x"])
    if case.operator == "silu":
        return torch.nn.functional.silu(tensors["x"])
    if case.operator == "softmax":
        return torch.softmax(tensors["x"], dim=case.axis)
    if case.operator == "add":
        return tensors["x"] + tensors["y"]
    if case.operator == "sub":
        return tensors["x"] - tensors["y"]
    if case.operator == "mul":
        return tensors["x"] * tensors["y"]
    if case.operator == "div":
        return tensors["x"] / (tensors["y"] + 1e-3)
    if case.operator == "maxpool2d":
        return torch.nn.functional.max_pool2d(tensors["x"], case.kernel_size, stride=case.stride)
    if case.operator == "avgpool2d":
        return torch.nn.functional.avg_pool2d(tensors["x"], case.kernel_size, stride=case.stride)
    if case.operator == "global_avgpool2d":
        return torch.mean(tensors["x"], dim=(2, 3), keepdim=True)
    if case.operator == "flatten":
        return torch.flatten(tensors["x"], start_dim=1)
    raise ValueError(f"torch backend does not support operator '{case.operator}' yet.")


def build_onnxruntime_cpu(case: BenchmarkCase, inputs: dict[str, np.ndarray]) -> Callable[[], np.ndarray]:
    return build_onnxruntime_executor(case, inputs, provider="CPUExecutionProvider")


def build_onnxruntime_executor(
    case: BenchmarkCase,
    inputs: dict[str, np.ndarray],
    provider: str = "CPUExecutionProvider",
) -> Callable[[], np.ndarray]:
    if not supports_onnxruntime_backend(case.operator):
        raise ValueError(f"onnxruntime backend does not support operator '{case.operator}' yet.")
    onnx = _onnx()
    ort = _onnxruntime()
    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise RuntimeError(
            f"ONNX Runtime provider '{provider}' is not available on this machine. "
            f"Available providers: {', '.join(available_providers)}"
        )
    helper = onnx.helper
    TensorProto = onnx.TensorProto

    input_names, output_name, nodes, initializers, shapes = _onnx_graph_spec(case, inputs, helper, TensorProto)
    graph_inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shapes[name]))
        for name in input_names
    ]
    graph_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)
    graph = helper.make_graph(nodes, f"tinyedgebench_{case.operator}", graph_inputs, [graph_output], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = min(model.ir_version, 10)
    providers = [provider]
    if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available_providers:
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    active_providers = session.get_providers()
    if provider != "CPUExecutionProvider" and (not active_providers or active_providers[0] != provider):
        raise RuntimeError(
            f"ONNX Runtime did not activate provider '{provider}'. Active providers: {', '.join(active_providers)}. "
            "Install the matching runtime libraries locally or remove this backend from the config."
        )
    feed = {name: inputs[name].astype(np.float32, copy=False) for name in input_names}

    def execute() -> np.ndarray:
        return session.run([output_name], feed)[0].astype(np.float32, copy=False)

    return execute


def _onnx_graph_spec(case: BenchmarkCase, inputs: dict[str, np.ndarray], helper: Any, TensorProto: Any) -> tuple[list[str], str, list[Any], list[Any], dict[str, tuple[int, ...]]]:
    nodes = []
    initializers = []
    shapes = {key: value.shape for key, value in inputs.items() if np.issubdtype(value.dtype, np.floating)}

    if case.operator == "matmul":
        initializers.append(_tensor_proto(helper, TensorProto, "b_weight", inputs["b"].astype(np.float32)))
        nodes.append(helper.make_node("MatMul", ["a", "b"], ["out"]))
        nodes[-1].input[1] = "b_weight"
        return ["a"], "out", nodes, initializers, shapes
    if case.operator == "batch_matmul":
        initializers.append(_tensor_proto(helper, TensorProto, "b_weight", inputs["b"].astype(np.float32)))
        nodes.append(helper.make_node("MatMul", ["a", "b"], ["out"]))
        nodes[-1].input[1] = "b_weight"
        return ["a"], "out", nodes, initializers, shapes
    if case.operator == "linear":
        initializers.append(_tensor_proto(helper, TensorProto, "linear_weight", inputs["b"].astype(np.float32)))
        initializers.append(_tensor_proto(helper, TensorProto, "linear_bias", inputs["bias"].astype(np.float32)))
        nodes.extend(
            [
                helper.make_node("MatMul", ["a", "linear_weight"], ["mm"]),
                helper.make_node("Add", ["mm", "linear_bias"], ["out"]),
            ]
        )
        return ["a"], "out", nodes, initializers, shapes
    if case.operator in {"conv2d", "pointwise_conv2d"}:
        initializers.append(_tensor_proto(helper, TensorProto, "conv_weight", inputs["weight"].astype(np.float32)))
        initializers.append(_tensor_proto(helper, TensorProto, "conv_bias", inputs["bias"].astype(np.float32)))
        nodes.append(
            helper.make_node(
                "Conv",
                ["x", "conv_weight", "conv_bias"],
                ["out"],
                strides=[case.stride, case.stride],
                pads=[case.padding, case.padding, case.padding, case.padding],
            )
        )
        return ["x"], "out", nodes, initializers, shapes
    if case.operator == "depthwise_conv2d":
        channels = inputs["x"].shape[1]
        weight = inputs["weight"].reshape(channels, 1, *inputs["weight"].shape[-2:]).astype(np.float32)
        initializers.append(_tensor_proto(helper, TensorProto, "dw_weight", weight))
        initializers.append(_tensor_proto(helper, TensorProto, "dw_bias", inputs["bias"].astype(np.float32)))
        nodes.append(
            helper.make_node(
                "Conv",
                ["x", "dw_weight", "dw_bias"],
                ["out"],
                strides=[case.stride, case.stride],
                pads=[case.padding, case.padding, case.padding, case.padding],
                group=channels,
            )
        )
        return ["x"], "out", nodes, initializers, shapes
    if case.operator == "relu":
        nodes.append(helper.make_node("Relu", ["x"], ["out"]))
        return ["x"], "out", nodes, initializers, shapes
    if case.operator == "sigmoid":
        nodes.append(helper.make_node("Sigmoid", ["x"], ["out"]))
        return ["x"], "out", nodes, initializers, shapes
    if case.operator == "tanh":
        nodes.append(helper.make_node("Tanh", ["x"], ["out"]))
        return ["x"], "out", nodes, initializers, shapes
    if case.operator in {"add", "sub", "mul", "div"}:
        op = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div"}[case.operator]
        nodes.append(helper.make_node(op, ["x", "y"], ["out"]))
        return ["x", "y"], "out", nodes, initializers, shapes
    if case.operator == "flatten":
        nodes.append(helper.make_node("Flatten", ["x"], ["out"], axis=1))
        return ["x"], "out", nodes, initializers, shapes
    raise ValueError(f"onnxruntime backend does not support operator '{case.operator}' yet.")


def _tensor_proto(helper: Any, TensorProto: Any, name: str, value: np.ndarray) -> Any:
    return helper.make_tensor(name, TensorProto.FLOAT, list(value.shape), value.flatten().tolist())


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def backend_availability() -> dict[str, str]:
    availability = {
        "cpu": "available",
        "numpy_cpu": "available",
        "torch_cpu": "missing PyTorch",
        "torch_cuda": "missing CUDA-capable PyTorch",
        "onnxruntime_cpu": "missing ONNX Runtime",
        "onnxruntime_cuda": "missing ONNX Runtime CUDAExecutionProvider",
        "onnxruntime_tensorrt": "missing ONNX Runtime TensorrtExecutionProvider",
        "openvino_cpu": "missing OpenVINO runtime",
        "tvm_cpu": "missing Apache TVM runtime",
        "tvm_cuda": "missing Apache TVM CUDA runtime",
        "tensorrt_cuda": "missing TensorRT Python runtime",
    }
    try:
        torch = _torch()
        availability["torch_cpu"] = "available"
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "CUDA device"
            availability["torch_cuda"] = f"available ({device_name})"
    except RuntimeError:
        pass
    try:
        ort = _onnxruntime()
        providers = ort.get_available_providers()
        if "CPUExecutionProvider" in providers:
            availability["onnxruntime_cpu"] = "available"
        if "CUDAExecutionProvider" in providers:
            availability["onnxruntime_cuda"] = "available (CUDAExecutionProvider)"
        if "TensorrtExecutionProvider" in providers:
            availability["onnxruntime_tensorrt"] = "provider listed (TensorRT libraries must load locally)"
    except RuntimeError:
        pass
    try:
        import openvino as _openvino  # noqa: F401

        availability["openvino_cpu"] = "installed (executor pending)"
    except ImportError:
        pass
    try:
        import tvm as _tvm  # noqa: F401

        availability["tvm_cpu"] = "installed (executor pending)"
        availability["tvm_cuda"] = "installed (executor pending)"
    except ImportError:
        pass
    try:
        import tensorrt as _tensorrt  # noqa: F401

        availability["tensorrt_cuda"] = "installed (executor pending)"
    except ImportError:
        pass
    return availability


@lru_cache(maxsize=1)
def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch_cpu backend requires PyTorch. Install with: python -m pip install -e \".[torch]\"") from exc
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return torch


@lru_cache(maxsize=1)
def _onnxruntime() -> Any:
    try:
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime_cpu backend requires ONNX Runtime. Install with: python -m pip install -e \".[onnx]\""
        ) from exc
    return onnxruntime


@lru_cache(maxsize=1)
def _onnx() -> Any:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnxruntime_cpu backend requires onnx. Install with: python -m pip install -e \".[onnx]\"") from exc
    return onnx

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


def supports_real_backend(operator: str) -> bool:
    return operator in REAL_BACKEND_OPERATORS


def run_torch_cpu(case: BenchmarkCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = _torch()
    with torch.no_grad():
        if case.operator == "matmul":
            return _to_numpy(torch.matmul(_tensor(inputs["a"]), _tensor(inputs["b"])))
        if case.operator == "batch_matmul":
            return _to_numpy(torch.matmul(_tensor(inputs["a"]), _tensor(inputs["b"])))
        if case.operator == "linear":
            return _to_numpy(torch.matmul(_tensor(inputs["a"]), _tensor(inputs["b"])) + _tensor(inputs["bias"]))
        if case.operator in {"conv2d", "pointwise_conv2d"}:
            return _to_numpy(
                torch.nn.functional.conv2d(
                    _tensor(inputs["x"]),
                    _tensor(inputs["weight"]),
                    _tensor(inputs["bias"]),
                    stride=case.stride,
                    padding=case.padding,
                )
            )
        if case.operator == "depthwise_conv2d":
            channels = inputs["x"].shape[1]
            return _to_numpy(
                torch.nn.functional.conv2d(
                    _tensor(inputs["x"]),
                    _tensor(inputs["weight"]).reshape(channels, 1, *inputs["weight"].shape[-2:]),
                    _tensor(inputs["bias"]),
                    stride=case.stride,
                    padding=case.padding,
                    groups=channels,
                )
            )
        if case.operator == "relu":
            return _to_numpy(torch.relu(_tensor(inputs["x"])))
        if case.operator == "relu6":
            return _to_numpy(torch.clamp(_tensor(inputs["x"]), 0, 6))
        if case.operator == "sigmoid":
            return _to_numpy(torch.sigmoid(_tensor(inputs["x"])))
        if case.operator == "tanh":
            return _to_numpy(torch.tanh(_tensor(inputs["x"])))
        if case.operator == "gelu":
            return _to_numpy(torch.nn.functional.gelu(_tensor(inputs["x"])))
        if case.operator == "silu":
            return _to_numpy(torch.nn.functional.silu(_tensor(inputs["x"])))
        if case.operator == "softmax":
            return _to_numpy(torch.softmax(_tensor(inputs["x"]), dim=case.axis))
        if case.operator == "add":
            return _to_numpy(_tensor(inputs["x"]) + _tensor(inputs["y"]))
        if case.operator == "sub":
            return _to_numpy(_tensor(inputs["x"]) - _tensor(inputs["y"]))
        if case.operator == "mul":
            return _to_numpy(_tensor(inputs["x"]) * _tensor(inputs["y"]))
        if case.operator == "div":
            return _to_numpy(_tensor(inputs["x"]) / (_tensor(inputs["y"]) + 1e-3))
        if case.operator == "maxpool2d":
            return _to_numpy(torch.nn.functional.max_pool2d(_tensor(inputs["x"]), case.kernel_size, stride=case.stride))
        if case.operator == "avgpool2d":
            return _to_numpy(torch.nn.functional.avg_pool2d(_tensor(inputs["x"]), case.kernel_size, stride=case.stride))
        if case.operator == "global_avgpool2d":
            return _to_numpy(torch.mean(_tensor(inputs["x"]), dim=(2, 3), keepdim=True))
        if case.operator == "flatten":
            return _to_numpy(torch.flatten(_tensor(inputs["x"]), start_dim=1))
    raise ValueError(f"torch_cpu does not support operator '{case.operator}' yet.")


def build_onnxruntime_cpu(case: BenchmarkCase, inputs: dict[str, np.ndarray]) -> Callable[[], np.ndarray]:
    if not supports_real_backend(case.operator):
        raise ValueError(f"onnxruntime_cpu does not support operator '{case.operator}' yet.")
    onnx = _onnx()
    ort = _onnxruntime()
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
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
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
    raise ValueError(f"onnxruntime_cpu does not support operator '{case.operator}' yet.")


def _tensor_proto(helper: Any, TensorProto: Any, name: str, value: np.ndarray) -> Any:
    return helper.make_tensor(name, TensorProto.FLOAT, list(value.shape), value.flatten().tolist())


def _tensor(value: np.ndarray) -> Any:
    torch = _torch()
    return torch.from_numpy(value.astype(np.float32, copy=False))


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


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

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Callable

from tinyedgebench.benchmark import run_config
from tinyedgebench.config import (
    SUPPORTED_OPERATORS,
    SUPPORTED_PRECISIONS,
    BenchmarkCase,
    CONV_OPERATORS,
    MATRIX_OPERATORS,
    wizard_case_to_config,
)


InputFunc = Callable[[str], str]


def build_wizard_case(input_func: InputFunc = input, print_func: Callable[[str], None] = print) -> BenchmarkCase:
    print_func("TinyEdgeBench interactive wizard")
    operator = _choice(
        "Operator type (for example conv2d, matmul, relu, softmax)",
        SUPPORTED_OPERATORS,
        input_func,
        default="conv2d",
    )
    precision_modes = _precision_modes(input_func)

    if operator in MATRIX_OPERATORS:
        m = _int_prompt("Matrix M", input_func, default=32)
        k = _int_prompt("Matrix K", input_func, default=64)
        n = _int_prompt("Matrix N", input_func, default=16)
        batch = 1
        if operator == "batch_matmul":
            batch = _int_prompt("Batch size", input_func, default=1)
        return BenchmarkCase(
            name=f"wizard_{operator}",
            operator=operator,
            precision_modes=precision_modes,
            matrix_m=m,
            matrix_k=k,
            matrix_n=n,
            batch_size=batch,
        )

    if operator == "embedding":
        return BenchmarkCase(
            name="wizard_embedding",
            operator=operator,
            precision_modes=precision_modes,
            batch_size=_int_prompt("Batch size", input_func, default=1),
            sequence_length=_int_prompt("Sequence length", input_func, default=16),
            vocab_size=_int_prompt("Vocabulary size", input_func, default=128),
            embedding_dim=_int_prompt("Embedding dimension", input_func, default=32),
        )

    if operator == "scaled_dot_product_attention":
        return BenchmarkCase(
            name="wizard_scaled_dot_product_attention",
            operator=operator,
            precision_modes=precision_modes,
            batch_size=_int_prompt("Batch size", input_func, default=1),
            sequence_length=_int_prompt("Sequence length", input_func, default=16),
            embedding_dim=_int_prompt("Embedding dimension", input_func, default=64),
            num_heads=_int_prompt("Attention heads", input_func, default=4),
        )

    if operator not in CONV_OPERATORS:
        shape = _generic_shape_prompt("Input shape comma-separated", input_func, default=(1, 16, 16, 16))
        return BenchmarkCase(
            name=f"wizard_{operator}",
            operator=operator,
            precision_modes=precision_modes,
            input_shape_generic=shape,
        )

    input_shape = _shape_prompt("Input shape N,C,H,W", input_func, default=(1, 3, 16, 16))
    kernel_size = _kernel_prompt(input_func)
    stride = _int_prompt("Stride", input_func, default=1)
    padding = _int_prompt("Padding", input_func, default=1, allow_zero=True)
    output_channels = None
    if operator in {"conv2d", "pointwise_conv2d"}:
        output_channels = _int_prompt("Output channels", input_func, default=8)

    return BenchmarkCase(
        name=f"wizard_{operator}",
        operator=operator,
        precision_modes=precision_modes,
        input_shape=input_shape,
        output_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def wizard(input_func: InputFunc = input, print_func: Callable[[str], None] = print) -> dict[str, Path]:
    case = build_wizard_case(input_func=input_func, print_func=print_func)
    backend = _choice("Backend", {"cpu"}, input_func, default="cpu")
    output_dir = input_func("Output directory [results]: ").strip() or "results"
    config = wizard_case_to_config(case, output_dir=output_dir)
    if backend != config.backend:
        raise ValueError("Only CPU backend is currently supported.")
    artifacts = run_config(config)
    print_func("TinyEdgeBench completed.")
    for label, path in artifacts.items():
        print_func(f"{label}: {path}")
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TinyEdgeBench local benchmark tool.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("wizard", help="Run the interactive CLI wizard.")
    web_parser = subparsers.add_parser("web", help="Launch the local Streamlit web UI.")
    web_parser.add_argument("streamlit_args", nargs=argparse.REMAINDER, help="Optional arguments passed to Streamlit.")
    args = parser.parse_args(argv)
    if args.command == "wizard":
        wizard()
        return 0
    if args.command == "web":
        return launch_web_app(args.streamlit_args)
    parser.print_help()
    return 1


def launch_web_app(streamlit_args: list[str] | None = None) -> int:
    app_path = Path(resources.files("tinyedgebench").joinpath("web_app.py"))
    extra_args = list(streamlit_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), *extra_args]
    return subprocess.call(command)


def _choice(prompt: str, allowed: set[str], input_func: InputFunc, default: str) -> str:
    allowed_display = ", ".join(sorted(allowed))
    while True:
        value = input_func(f"{prompt} [{default}]: ").strip().lower() or default
        if value in allowed:
            return value
        print(f"Please choose one of: {allowed_display}")


def _precision_modes(input_func: InputFunc) -> list[str]:
    default = "fp32,int8_sim,shift_only"
    while True:
        raw = input_func(f"Precision modes comma-separated [{default}]: ").strip() or default
        modes = [item.strip().lower() for item in raw.split(",") if item.strip()]
        if modes and all(mode in SUPPORTED_PRECISIONS for mode in modes):
            return modes
        print(f"Please choose from: {', '.join(sorted(SUPPORTED_PRECISIONS))}")


def _shape_prompt(prompt: str, input_func: InputFunc, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    default_text = ",".join(str(item) for item in default)
    while True:
        raw = input_func(f"{prompt} [{default_text}]: ").strip() or default_text
        try:
            parts = tuple(int(item.strip()) for item in raw.replace("x", ",").split(",") if item.strip())
            if len(parts) == 4 and all(item > 0 for item in parts):
                return parts  # type: ignore[return-value]
        except ValueError:
            pass
        print("Please enter four positive integers, for example 1,3,16,16.")


def _generic_shape_prompt(prompt: str, input_func: InputFunc, default: tuple[int, ...]) -> tuple[int, ...]:
    default_text = ",".join(str(item) for item in default)
    while True:
        raw = input_func(f"{prompt} [{default_text}]: ").strip() or default_text
        try:
            parts = tuple(int(item.strip()) for item in raw.replace("x", ",").split(",") if item.strip())
            if parts and all(item > 0 for item in parts):
                return parts
        except ValueError:
            pass
        print("Please enter positive integers, for example 1,16,16,16.")


def _kernel_prompt(input_func: InputFunc) -> tuple[int, int]:
    while True:
        raw = input_func("Kernel size H,W [3,3]: ").strip() or "3,3"
        try:
            parts = tuple(int(item.strip()) for item in raw.replace("x", ",").split(",") if item.strip())
            if len(parts) == 1 and parts[0] > 0:
                return (parts[0], parts[0])
            if len(parts) == 2 and all(item > 0 for item in parts):
                return parts  # type: ignore[return-value]
        except ValueError:
            pass
        print("Please enter one positive integer or two positive integers, for example 3 or 3,3.")


def _int_prompt(prompt: str, input_func: InputFunc, default: int, allow_zero: bool = False) -> int:
    while True:
        raw = input_func(f"{prompt} [{default}]: ").strip()
        value = default if not raw else int(raw)
        if value > 0 or (allow_zero and value == 0):
            return value
        print("Please enter a positive integer.")

from __future__ import annotations

import platform
import subprocess
import sys


def get_system_info() -> dict[str, str]:
    return {
        "operating_system": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version.replace("\n", " "),
        "cpu_info": _cpu_info(),
        "cuda_available": str(_cuda_available()),
        "gpu_info": _gpu_info(),
        "torch_version": _module_version("torch"),
        "torch_cuda_available": str(_torch_cuda_available()),
        "torch_cuda_version": _torch_cuda_version(),
        "onnxruntime_version": _module_version("onnxruntime"),
        "onnxruntime_providers": _onnxruntime_providers(),
    }


def _cpu_info() -> str:
    processor = platform.processor()
    if processor:
        return processor
    if platform.system().lower() == "windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip() and line.strip() != "Name"]
            if lines:
                return lines[0]
        except (OSError, subprocess.SubprocessError):
            pass
    return "Unknown"


def _cuda_available() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, timeout=2)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _gpu_info() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode == 0 and lines:
            return "; ".join(lines)
    except (OSError, subprocess.SubprocessError):
        pass
    return "Not detected"


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _torch_cuda_version() -> str:
    try:
        import torch

        return str(torch.version.cuda or "not available")
    except ImportError:
        return "not installed"


def _onnxruntime_providers() -> str:
    try:
        import onnxruntime as ort

        return ", ".join(ort.get_available_providers())
    except ImportError:
        return "not installed"


def _module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "installed"))
    except ImportError:
        return "not installed"

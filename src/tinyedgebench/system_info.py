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
        "torch_version": _module_version("torch"),
        "onnxruntime_version": _module_version("onnxruntime"),
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


def _module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "installed"))
    except ImportError:
        return "not installed"

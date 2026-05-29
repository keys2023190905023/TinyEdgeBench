from pathlib import Path

from tinyedgebench.benchmark import run_config
from tinyedgebench.config import load_config, parse_config
from tinyedgebench.real_backends import backend_availability
from tinyedgebench.runner import run_benchmarks


def test_default_yaml_runs(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml")
    config = type(config)(
        output_dir=tmp_path,
        warmup=0,
        runs=1,
        backend=config.backend,
        seed=config.seed,
        benchmarks=config.benchmarks,
    )

    artifacts = run_config(config)

    assert artifacts["summary"].exists()
    assert artifacts["report"].exists()
    assert artifacts["latency_plot"].exists()
    assert artifacts["error_plot"].exists()
    assert "TinyEdgeBench Report" in artifacts["report"].read_text(encoding="utf-8")


def test_backend_list_config_and_numpy_alias() -> None:
    config = parse_config(
        {
            "backends": ["cpu", "numpy_cpu"],
            "warmup": 0,
            "runs": 1,
            "benchmarks": [
                {
                    "name": "matmul_backend_check",
                    "operator": "matmul",
                    "matrix_m": 2,
                    "matrix_k": 3,
                    "matrix_n": 4,
                    "precision_modes": ["fp32"],
                }
            ],
        }
    )

    results = run_benchmarks(config)

    assert config.backends == ("cpu", "numpy_cpu")
    assert [result.backend for result in results] == ["cpu", "numpy_cpu"]


def test_gpu_backend_names_parse_without_running_hardware() -> None:
    config = parse_config(
        {
            "backends": ["torch_cuda", "onnxruntime_cuda"],
            "warmup": 0,
            "runs": 1,
            "benchmarks": [
                {
                    "name": "cuda_name_check",
                    "operator": "matmul",
                    "matrix_m": 2,
                    "matrix_k": 3,
                    "matrix_n": 4,
                    "precision_modes": ["fp32"],
                }
            ],
        }
    )

    assert config.backends == ("torch_cuda", "onnxruntime_cuda")


def test_backend_availability_reports_default_cpu() -> None:
    availability = backend_availability()

    assert availability["cpu"] == "available"
    assert availability["numpy_cpu"] == "available"
    assert "torch_cuda" in availability
    assert "onnxruntime_cuda" in availability

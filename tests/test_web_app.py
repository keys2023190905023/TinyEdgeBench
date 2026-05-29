from pathlib import Path

from tinyedgebench.cli import launch_web_app
from tinyedgebench.web_app import BACKEND_OPTIONS, build_web_case, build_web_config, results_to_rows
from tinyedgebench.runner import BenchmarkResult


def test_build_web_config_for_matmul(tmp_path: Path) -> None:
    case = build_web_case(
        operator="matmul",
        precision_modes=["fp32", "int8_sim"],
        matrix_m=4,
        matrix_k=5,
        matrix_n=6,
    )

    config = build_web_config(case, warmup=0, runs=1, output_dir=tmp_path)

    assert config.output_dir == tmp_path
    assert config.warmup == 0
    assert config.runs == 1
    assert config.backend == "cpu"
    assert config.benchmarks[0].operator == "matmul"
    assert config.benchmarks[0].matrix_n == 6


def test_web_backend_options_include_local_gpu_backends() -> None:
    assert "torch_cuda" in BACKEND_OPTIONS
    assert "onnxruntime_cuda" in BACKEND_OPTIONS
    assert "onnxruntime_tensorrt" in BACKEND_OPTIONS
    assert "openvino_cpu" in BACKEND_OPTIONS


def test_results_to_rows() -> None:
    rows = results_to_rows(
        [
            BenchmarkResult(
                name="case",
                operator="matmul",
                precision="fp32",
                backend="cpu",
                input_description="2x2 @ 2x2",
                latency_ms=1.0,
                throughput_ops_per_s=100.0,
                mean_abs_error=0.0,
                max_abs_error=0.0,
            )
        ]
    )

    assert rows == [
        {
            "benchmark": "case",
            "operator": "matmul",
            "precision": "fp32",
            "backend": "cpu",
            "input": "2x2 @ 2x2",
            "latency_ms": 1.0,
            "throughput_ops_per_s": 100.0,
            "mean_abs_error": 0.0,
            "max_abs_error": 0.0,
        }
    ]


def test_launch_web_app_invokes_streamlit(monkeypatch) -> None:
    captured = {}

    def fake_call(command):
        captured["command"] = command
        return 0

    monkeypatch.setattr("tinyedgebench.cli.subprocess.call", fake_call)

    exit_code = launch_web_app(["--", "--server.headless", "true"])

    assert exit_code == 0
    assert captured["command"][1:4] == ["-m", "streamlit", "run"]
    assert captured["command"][4].endswith("web_app.py")
    assert captured["command"][-2:] == ["--server.headless", "true"]

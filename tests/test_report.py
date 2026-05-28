from pathlib import Path

from tinyedgebench.artifacts import write_report
from tinyedgebench.runner import BenchmarkResult


def test_report_generation_contains_system_info(tmp_path: Path) -> None:
    result = BenchmarkResult(
        name="case",
        operator="matmul",
        precision="fp32",
        backend="cpu",
        input_description="2x2 @ 2x2",
        latency_ms=1.23,
        throughput_ops_per_s=1000.0,
        mean_abs_error=0.0,
        max_abs_error=0.0,
    )
    report_path = tmp_path / "report.md"

    write_report([result], report_path, tmp_path / "latency_plot.png", tmp_path / "error_plot.png")

    report = report_path.read_text(encoding="utf-8")
    assert "Operating system" in report
    assert "Python version" in report
    assert "CUDA available" in report
    assert "| case | matmul | fp32 | cpu |" in report

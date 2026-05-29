from pathlib import Path

from tinyedgebench.config import BenchmarkCase, BenchmarkConfig
from tinyedgebench.history import compare_runs, record_run, write_comparison, zip_directory
from tinyedgebench.runner import BenchmarkResult
from tinyedgebench.artifacts import write_artifacts


def test_record_and_compare_runs(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        output_dir=tmp_path / "out",
        benchmarks=[BenchmarkCase("case", "matmul", ["fp32"], matrix_m=2, matrix_k=2, matrix_n=2)],
    )
    old_result = BenchmarkResult("case", "matmul", "fp32", "cpu", "2x2 @ 2x2", 2.0, 10.0, 0.0, 0.0)
    new_result = BenchmarkResult("case", "matmul", "fp32", "cpu", "2x2 @ 2x2", 1.0, 20.0, 0.0, 0.0)

    old_artifacts = write_artifacts([old_result], tmp_path / "old")
    new_artifacts = write_artifacts([new_result], tmp_path / "new")
    old_run = record_run(config, [old_result], old_artifacts, runs_dir=tmp_path / "runs")
    new_run = record_run(config, [new_result], new_artifacts, runs_dir=tmp_path / "runs")

    comparison = compare_runs(old_run, new_run)
    outputs = write_comparison(comparison, tmp_path / "compare")
    zip_path = zip_directory(new_run)

    assert comparison[0]["speedup"] == 2.0
    assert outputs["comparison_csv"].exists()
    assert outputs["comparison_report"].exists()
    assert zip_path.exists()

from pathlib import Path

from tinyedgebench.benchmark import run_config
from tinyedgebench.config import load_config


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

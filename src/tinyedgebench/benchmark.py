from __future__ import annotations

import argparse
from pathlib import Path

from tinyedgebench.artifacts import write_artifacts
from tinyedgebench.config import BenchmarkConfig, load_config
from tinyedgebench.history import record_run
from tinyedgebench.runner import run_benchmarks


def run_config(config: BenchmarkConfig, record_history: bool = False) -> dict[str, Path]:
    results = run_benchmarks(config)
    artifacts = write_artifacts(results, config.output_dir)
    if record_history:
        artifacts["history_run"] = record_run(config, results, artifacts)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TinyEdgeBench from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a TinyEdgeBench YAML config.")
    parser.add_argument("--history", action="store_true", help="Copy generated artifacts into results/runs/<timestamp>.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    artifacts = run_config(config, record_history=args.history)
    print("TinyEdgeBench completed.")
    for label, path in artifacts.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

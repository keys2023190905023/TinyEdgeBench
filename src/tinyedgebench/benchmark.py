from __future__ import annotations

import argparse
from pathlib import Path

from tinyedgebench.artifacts import write_artifacts
from tinyedgebench.config import BenchmarkConfig, load_config
from tinyedgebench.runner import run_benchmarks


def run_config(config: BenchmarkConfig) -> dict[str, Path]:
    results = run_benchmarks(config)
    return write_artifacts(results, config.output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TinyEdgeBench from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a TinyEdgeBench YAML config.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    artifacts = run_config(config)
    print("TinyEdgeBench completed.")
    for label, path in artifacts.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

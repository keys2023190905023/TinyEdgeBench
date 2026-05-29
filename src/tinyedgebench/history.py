from __future__ import annotations

import csv
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from tinyedgebench.config import BenchmarkConfig
from tinyedgebench.runner import BenchmarkResult


def record_run(
    config: BenchmarkConfig,
    results: list[BenchmarkResult],
    artifacts: dict[str, Path],
    runs_dir: str | Path = "results/runs",
) -> Path:
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_dir = Path(runs_dir) / run_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = Path(runs_dir) / f"{run_id}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for path in artifacts.values():
        if path.exists():
            shutil.copy2(path, run_dir / path.name)
    metadata = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(config.output_dir),
        "warmup": config.warmup,
        "runs": config.runs,
        "backends": list(getattr(config, "backends", (config.backend,))),
        "benchmarks": len(config.benchmarks),
        "result_rows": len(results),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir


def list_runs(runs_dir: str | Path = "results/runs") -> list[Path]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if (path / "summary.csv").exists()], reverse=True)


def compare_runs(baseline_dir: str | Path, candidate_dir: str | Path) -> list[dict[str, Any]]:
    baseline_rows = _load_summary(Path(baseline_dir) / "summary.csv")
    candidate_rows = _load_summary(Path(candidate_dir) / "summary.csv")
    baseline_by_key = {_row_key(row): row for row in baseline_rows}
    comparisons = []
    for candidate in candidate_rows:
        key = _row_key(candidate)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        old_latency = float(baseline["latency_ms"])
        new_latency = float(candidate["latency_ms"])
        speedup = old_latency / new_latency if new_latency > 0 else 0.0
        delta_pct = ((new_latency - old_latency) / old_latency * 100.0) if old_latency > 0 else 0.0
        comparisons.append(
            {
                "name": candidate["name"],
                "operator": candidate["operator"],
                "precision": candidate["precision"],
                "backend": candidate["backend"],
                "baseline_latency_ms": old_latency,
                "candidate_latency_ms": new_latency,
                "latency_delta_pct": delta_pct,
                "speedup": speedup,
                "baseline_mean_abs_error": float(baseline["mean_abs_error"]),
                "candidate_mean_abs_error": float(candidate["mean_abs_error"]),
            }
        )
    return comparisons


def write_comparison(comparisons: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "comparison.csv"
    md_path = out / "comparison.md"
    columns = [
        "name",
        "operator",
        "precision",
        "backend",
        "baseline_latency_ms",
        "candidate_latency_ms",
        "latency_delta_pct",
        "speedup",
        "baseline_mean_abs_error",
        "candidate_mean_abs_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(comparisons)
    lines = [
        "# TinyEdgeBench Run Comparison",
        "",
        "| Benchmark | Operator | Precision | Backend | Baseline ms | Candidate ms | Delta % | Speedup |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        lines.append(
            "| "
            f"{row['name']} | {row['operator']} | {row['precision']} | {row['backend']} | "
            f"{row['baseline_latency_ms']:.4f} | {row['candidate_latency_ms']:.4f} | "
            f"{row['latency_delta_pct']:.2f} | {row['speedup']:.3f} |"
        )
    if not comparisons:
        lines.append("| No matching rows | - | - | - | - | - | - | - |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"comparison_csv": csv_path, "comparison_report": md_path}


def zip_directory(directory: str | Path, output_path: str | Path | None = None) -> Path:
    source = Path(directory)
    zip_path = Path(output_path) if output_path else source.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source.rglob("*"):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(source))
    return zip_path


def _load_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary.csv: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["name"], row["operator"], row["precision"], row["backend"])

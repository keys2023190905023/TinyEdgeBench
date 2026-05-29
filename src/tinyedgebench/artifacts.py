from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tinyedgebench.runner import BenchmarkResult
from tinyedgebench.system_info import get_system_info


SUMMARY_COLUMNS = [
    "name",
    "operator",
    "precision",
    "backend",
    "input_description",
    "latency_ms",
    "throughput_ops_per_s",
    "mean_abs_error",
    "max_abs_error",
]


def write_artifacts(results: list[BenchmarkResult], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.csv"
    report_path = out / "report.md"
    latency_plot_path = out / "latency_plot.png"
    error_plot_path = out / "error_plot.png"

    write_summary_csv(results, summary_path)
    write_latency_plot(results, latency_plot_path)
    write_error_plot(results, error_plot_path)
    write_report(results, report_path, latency_plot_path, error_plot_path)

    return {
        "summary": summary_path,
        "report": report_path,
        "latency_plot": latency_plot_path,
        "error_plot": error_plot_path,
    }


def write_summary_csv(results: list[BenchmarkResult], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "name": result.name,
                    "operator": result.operator,
                    "precision": result.precision,
                    "backend": result.backend,
                    "input_description": result.input_description,
                    "latency_ms": f"{result.latency_ms:.6f}",
                    "throughput_ops_per_s": f"{result.throughput_ops_per_s:.2f}",
                    "mean_abs_error": f"{result.mean_abs_error:.8f}",
                    "max_abs_error": f"{result.max_abs_error:.8f}",
                }
            )


def write_report(
    results: list[BenchmarkResult],
    path: str | Path,
    latency_plot_path: str | Path,
    error_plot_path: str | Path,
) -> None:
    system_info = get_system_info()
    fastest = min(results, key=lambda item: item.latency_ms)
    slowest = max(results, key=lambda item: item.latency_ms)
    backend_rankings = _backend_rankings(results)
    bottlenecks = sorted(results, key=lambda item: item.latency_ms, reverse=True)[:5]
    path = Path(path)
    latency_rel = Path(latency_plot_path).name
    error_rel = Path(error_plot_path).name
    lines = [
        "# TinyEdgeBench Report",
        "",
        "## System Information",
        "",
        "- Execution location: local machine running this benchmark command or Streamlit app.",
        f"- Operating system: {system_info['operating_system']}",
        f"- Python version: {system_info['python_version']}",
        f"- CPU information: {system_info['cpu_info']}",
        f"- CUDA available: {system_info['cuda_available']}",
        f"- GPU information: {system_info['gpu_info']}",
        f"- PyTorch version: {system_info['torch_version']}",
        f"- PyTorch CUDA available: {system_info['torch_cuda_available']}",
        f"- PyTorch CUDA version: {system_info['torch_cuda_version']}",
        f"- ONNX Runtime version: {system_info['onnxruntime_version']}",
        f"- ONNX Runtime providers: {system_info['onnxruntime_providers']}",
        "",
        "## Executive Summary",
        "",
        f"- Benchmarks executed: {len(results)} result rows.",
        f"- Fastest row: `{fastest.name}` on `{fastest.backend}` / `{fastest.precision}` at {fastest.latency_ms:.4f} ms.",
        f"- Slowest row: `{slowest.name}` on `{slowest.backend}` / `{slowest.precision}` at {slowest.latency_ms:.4f} ms.",
        f"- Highest mean absolute error: {max(result.mean_abs_error for result in results):.6f}.",
        "",
        "## Backend Ranking",
        "",
        "| Backend | Median Latency (ms) | Rows |",
        "| --- | ---: | ---: |",
    ]
    for backend, median_latency, count in backend_rankings:
        lines.append(f"| {backend} | {median_latency:.4f} | {count} |")
    lines.extend(
        [
            "",
            "## Bottleneck Rows",
            "",
            "| Benchmark | Operator | Precision | Backend | Latency (ms) |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for result in bottlenecks:
        lines.append(f"| {result.name} | {result.operator} | {result.precision} | {result.backend} | {result.latency_ms:.4f} |")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "Run the same YAML config on the target machine with:",
            "",
            "```bash",
            "python -m tinyedgebench.benchmark --config path/to/config.yaml --history",
            "```",
            "",
            "Compare two saved runs with:",
            "",
            "```bash",
            "tinyedgebench compare results/runs/<baseline> results/runs/<candidate>",
            "```",
            "",
            "## Results",
            "",
            "| Benchmark | Operator | Precision | Backend | Latency (ms) | Mean Abs Error | Max Abs Error |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        lines.append(
            "| "
            f"{result.name} | {result.operator} | {result.precision} | {result.backend} | "
            f"{result.latency_ms:.4f} | {result.mean_abs_error:.6f} | {result.max_abs_error:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            f"![Latency plot]({latency_rel})",
            "",
            f"![Error plot]({error_rel})",
            "",
            "## Notes",
            "",
            "- `int8_sim` uses symmetric int8 quantization with int32 accumulation and float dequantization.",
            "- `shift_only` rounds operands to signed powers of two to approximate shift-only arithmetic.",
            "- `cpu` is the default NumPy backend and remains available without optional dependencies.",
            "- `torch_cpu`, `torch_cuda`, `onnxruntime_cpu`, and `onnxruntime_cuda` measure real local backend kernels when the matching optional dependencies and hardware are available.",
            "- Real backend comparison currently reports FP32 timings; `int8_sim` and `shift_only` are simulation modes unless a backend-specific quantized kernel is added.",
            "- GitHub Pages can showcase the project, but benchmark data is generated only on the machine where TinyEdgeBench is run.",
            "- FPGA and NPU backends are roadmap items.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _backend_rankings(results: list[BenchmarkResult]) -> list[tuple[str, float, int]]:
    by_backend: dict[str, list[float]] = {}
    for result in results:
        by_backend.setdefault(result.backend, []).append(result.latency_ms)
    rankings = []
    for backend, latencies in by_backend.items():
        sorted_latencies = sorted(latencies)
        mid = len(sorted_latencies) // 2
        if len(sorted_latencies) % 2:
            median_latency = sorted_latencies[mid]
        else:
            median_latency = (sorted_latencies[mid - 1] + sorted_latencies[mid]) / 2
        rankings.append((backend, median_latency, len(latencies)))
    return sorted(rankings, key=lambda item: item[1])


def write_latency_plot(results: list[BenchmarkResult], path: str | Path) -> None:
    labels = [f"{item.name}\n{item.precision}" for item in results]
    values = [item.latency_ms for item in results]
    _write_bar_plot(labels, values, "Latency by benchmark", "Latency (ms)", path)


def write_error_plot(results: list[BenchmarkResult], path: str | Path) -> None:
    labels = [f"{item.name}\n{item.precision}" for item in results]
    values = [item.mean_abs_error for item in results]
    _write_bar_plot(labels, values, "Mean absolute error vs FP32", "Mean absolute error", path)


def _write_bar_plot(labels: list[str], values: list[float], title: str, ylabel: str, path: str | Path) -> None:
    width = max(8.0, len(labels) * 1.15)
    fig, ax = plt.subplots(figsize=(width, 4.8), dpi=140)
    ax.bar(range(len(values)), values, color="#2f6f73")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

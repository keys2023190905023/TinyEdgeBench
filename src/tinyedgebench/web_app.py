from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from tinyedgebench.artifacts import write_artifacts
from tinyedgebench.config import (
    CONV_OPERATORS,
    MATRIX_OPERATORS,
    SUPPORTED_BACKENDS,
    SUPPORTED_OPERATORS,
    BenchmarkCase,
    BenchmarkConfig,
    parse_config,
)
from tinyedgebench.history import compare_runs, list_runs, record_run, write_comparison, zip_directory
from tinyedgebench.network_presets import NETWORK_PRESETS, build_network_preset
from tinyedgebench.real_backends import backend_availability
from tinyedgebench.runner import BenchmarkResult, run_benchmarks


PRECISION_OPTIONS = ["fp32", "int8_sim", "shift_only"]
OPERATOR_OPTIONS = sorted(SUPPORTED_OPERATORS)
BACKEND_OPTIONS = [
    "cpu",
    "torch_cpu",
    "torch_cuda",
    "onnxruntime_cpu",
    "onnxruntime_cuda",
    "onnxruntime_tensorrt",
    "openvino_cpu",
    "tvm_cpu",
    "tvm_cuda",
    "tensorrt_cuda",
]


def build_web_case(
    operator: str,
    precision_modes: list[str],
    input_shape: tuple[int, int, int, int] = (1, 3, 16, 16),
    output_channels: int = 8,
    kernel_size: tuple[int, int] = (3, 3),
    stride: int = 1,
    padding: int = 1,
    matrix_m: int = 32,
    matrix_k: int = 64,
    matrix_n: int = 16,
    batch_size: int = 1,
    generic_shape: tuple[int, ...] = (1, 16, 16, 16),
    axis: int = -1,
    groups: int = 1,
    vocab_size: int = 128,
    sequence_length: int = 16,
    embedding_dim: int = 32,
    num_heads: int = 1,
) -> BenchmarkCase:
    if not precision_modes:
        raise ValueError("Select at least one precision mode.")

    if operator in MATRIX_OPERATORS:
        return BenchmarkCase(
            name=f"web_{operator}",
            operator=operator,
            precision_modes=precision_modes,
            matrix_m=matrix_m,
            matrix_k=matrix_k,
            matrix_n=matrix_n,
            batch_size=batch_size,
        )

    if operator in {"conv2d", "pointwise_conv2d"}:
        return BenchmarkCase(
            name=f"web_{operator}",
            operator=operator,
            precision_modes=precision_modes,
            input_shape=input_shape,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    if operator == "depthwise_conv2d":
        return BenchmarkCase(
            name="web_depthwise_conv2d",
            operator=operator,
            precision_modes=precision_modes,
            input_shape=input_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    if operator == "embedding":
        return BenchmarkCase(
            name="web_embedding",
            operator=operator,
            precision_modes=precision_modes,
            batch_size=batch_size,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
        )

    if operator in {"scaled_dot_product_attention", "causal_self_attention"}:
        return BenchmarkCase(
            name=f"web_{operator}",
            operator=operator,
            precision_modes=precision_modes,
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )

    target_shape = generic_shape if operator == "reshape" else None
    return BenchmarkCase(
        name=f"web_{operator}",
        operator=operator,
        precision_modes=precision_modes,
        input_shape_generic=generic_shape,
        kernel_size=kernel_size if operator in {"maxpool2d", "avgpool2d"} else None,
        stride=stride,
        padding=padding,
        axis=axis,
        groups=groups,
        target_shape=target_shape,
    )


def build_web_config(
    case: BenchmarkCase | list[BenchmarkCase],
    warmup: int,
    runs: int,
    output_dir: str | Path,
    backends: list[str] | None = None,
    seed: int = 42,
) -> BenchmarkConfig:
    selected_backends = tuple(backends or ["cpu"])
    return BenchmarkConfig(
        output_dir=Path(output_dir),
        warmup=warmup,
        runs=runs,
        backend=selected_backends[0],
        backends=selected_backends,
        seed=seed,
        benchmarks=case if isinstance(case, list) else [case],
    )


def results_to_rows(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    return [
        {
            "benchmark": result.name,
            "operator": result.operator,
            "precision": result.precision,
            "backend": result.backend,
            "input": result.input_description,
            "latency_ms": result.latency_ms,
            "latency_p90_ms": result.latency_p90_ms,
            "latency_std_ms": result.latency_std_ms,
            "throughput_ops_per_s": result.throughput_ops_per_s,
            "mean_abs_error": result.mean_abs_error,
            "max_abs_error": result.max_abs_error,
            "peak_memory_mb": result.peak_memory_mb,
            "gpu_memory_allocated_mb": result.gpu_memory_allocated_mb,
            "power_w": result.power_w,
            "energy_mj": result.energy_mj,
            "edp_mj_ms": result.edp_mj_ms,
        }
        for result in results
    ]


def run_web_benchmark(config: BenchmarkConfig, save_history: bool = True) -> tuple[list[BenchmarkResult], dict[str, Path]]:
    results = run_benchmarks(config)
    artifacts = write_artifacts(results, config.output_dir)
    if save_history:
        artifacts["history_run"] = record_run(config, results, artifacts)
    return results, artifacts


def main() -> None:
    st.set_page_config(page_title="TinyEdgeBench", layout="wide")
    _inject_theme()
    availability = backend_availability()
    _render_header(availability)
    _render_backend_deck(availability)
    _render_history_tools()

    with st.sidebar:
        st.header("Benchmark")
        uploaded_yaml = st.file_uploader("Run YAML config", type=["yaml", "yml"])
        if uploaded_yaml is not None and st.button("Run Uploaded YAML", type="secondary"):
            try:
                raw = yaml.safe_load(uploaded_yaml.getvalue().decode("utf-8")) or {}
                config = parse_config(raw)
                with st.spinner("Running uploaded YAML locally..."):
                    results, artifacts = run_web_benchmark(config)
            except Exception as exc:
                st.error(str(exc))
                return
            _render_results(results, artifacts, config.output_dir)
            return
        benchmark_mode = st.radio("Benchmark mode", ["Single operator", "Network preset"], horizontal=True)
        precision_modes = st.multiselect("Precision modes", PRECISION_OPTIONS, default=PRECISION_OPTIONS)
        backends = st.multiselect("Backends", BACKEND_OPTIONS, default=["cpu"])
        st.caption("Backend availability on this local machine:")
        for backend in BACKEND_OPTIONS:
            st.caption(f"{backend}: {availability.get(backend, 'unknown')}")
        warmup = st.number_input("Warmup runs", min_value=0, max_value=100, value=2, step=1)
        runs = st.number_input("Benchmark runs", min_value=1, max_value=1000, value=5, step=1)
        output_dir = st.text_input("Output directory", value="results")
        if any(backend not in {"cpu", "numpy_cpu"} for backend in backends) and any(mode != "fp32" for mode in precision_modes):
            st.warning("Real backend comparison currently supports fp32. Use cpu for int8_sim and shift_only simulations.")
        st.info("The web page is only the control panel. Benchmarks execute in the local Python process that launched Streamlit.")

    if benchmark_mode == "Network preset":
        preset = st.selectbox(
            "Network / block preset",
            list(NETWORK_PRESETS),
            format_func=lambda item: f"{item} - {NETWORK_PRESETS[item]}",
        )
        st.info("Network presets run a suite of local operator benchmarks that approximate common model blocks.")
        run_button = st.button("Run Benchmark", type="primary")
        if run_button:
            try:
                cases = build_network_preset(preset, list(precision_modes))
                config = build_web_config(cases, warmup=int(warmup), runs=int(runs), output_dir=output_dir, backends=backends)
                with st.spinner("Running preset benchmarks locally..."):
                    results, artifacts = run_web_benchmark(config)
            except Exception as exc:
                st.error(str(exc))
                return
            _render_results(results, artifacts, config.output_dir)
        else:
            st.info("Choose a preset, then run it locally from this browser session.")
        return

    operator = st.sidebar.selectbox("Operator type", OPERATOR_OPTIONS, index=OPERATOR_OPTIONS.index("conv2d"))
    st.subheader("Operator Parameters")
    batch_size = 1
    generic_shape = (1, 16, 16, 16)
    axis = -1
    groups = 1
    vocab_size = 128
    sequence_length = 16
    embedding_dim = 32
    num_heads = 1
    if operator in MATRIX_OPERATORS:
        col_m, col_k, col_n = st.columns(3)
        with col_m:
            matrix_m = st.number_input("Matrix M", min_value=1, max_value=8192, value=32, step=1)
        with col_k:
            matrix_k = st.number_input("Matrix K", min_value=1, max_value=8192, value=64, step=1)
        with col_n:
            matrix_n = st.number_input("Matrix N", min_value=1, max_value=8192, value=16, step=1)
        if operator == "batch_matmul":
            batch_size = st.number_input("Batch size", min_value=1, max_value=1024, value=1, step=1)
        input_shape = (1, 3, 16, 16)
        output_channels = 8
        kernel_size = (3, 3)
        stride = 1
        padding = 1
    elif operator in CONV_OPERATORS:
        col_n, col_c, col_h, col_w = st.columns(4)
        with col_n:
            batch = st.number_input("Batch N", min_value=1, max_value=1024, value=1, step=1)
        with col_c:
            channels = st.number_input("Input channels C", min_value=1, max_value=4096, value=3, step=1)
        with col_h:
            height = st.number_input("Height H", min_value=1, max_value=4096, value=16, step=1)
        with col_w:
            width = st.number_input("Width W", min_value=1, max_value=4096, value=16, step=1)

        col_out, col_kh, col_kw, col_stride, col_pad = st.columns(5)
        with col_out:
            output_channels = st.number_input(
                "Output channels",
                min_value=1,
                max_value=4096,
                value=8,
                step=1,
                disabled=operator == "depthwise_conv2d",
            )
        with col_kh:
            kernel_h = st.number_input("Kernel H", min_value=1, max_value=31, value=3, step=1)
        with col_kw:
            kernel_w = st.number_input("Kernel W", min_value=1, max_value=31, value=3, step=1)
        with col_stride:
            stride = st.number_input("Stride", min_value=1, max_value=16, value=1, step=1)
        with col_pad:
            padding = st.number_input("Padding", min_value=0, max_value=64, value=1, step=1)

        matrix_m = 32
        matrix_k = 64
        matrix_n = 16
        input_shape = (int(batch), int(channels), int(height), int(width))
        kernel_size = (int(kernel_h), int(kernel_w))
    elif operator == "embedding":
        col_b, col_s, col_v, col_e = st.columns(4)
        with col_b:
            batch_size = st.number_input("Batch size", min_value=1, max_value=1024, value=1, step=1)
        with col_s:
            sequence_length = st.number_input("Sequence length", min_value=1, max_value=8192, value=16, step=1)
        with col_v:
            vocab_size = st.number_input("Vocabulary size", min_value=1, max_value=100000, value=128, step=1)
        with col_e:
            embedding_dim = st.number_input("Embedding dim", min_value=1, max_value=8192, value=32, step=1)
        matrix_m = 32
        matrix_k = 64
        matrix_n = 16
        input_shape = (1, 3, 16, 16)
        output_channels = 8
        kernel_size = (3, 3)
        stride = 1
        padding = 1
    elif operator in {"scaled_dot_product_attention", "causal_self_attention"}:
        col_b, col_s, col_e, col_h = st.columns(4)
        with col_b:
            batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=1, step=1)
        with col_s:
            sequence_length = st.number_input("Sequence length", min_value=1, max_value=4096, value=16, step=1)
        with col_e:
            embedding_dim = st.number_input("Embedding dim", min_value=1, max_value=8192, value=64, step=1)
        with col_h:
            num_heads = st.number_input("Attention heads", min_value=1, max_value=128, value=4, step=1)
        matrix_m = 32
        matrix_k = 64
        matrix_n = 16
        input_shape = (1, 3, 16, 16)
        output_channels = 8
        kernel_size = (3, 3)
        stride = 1
        padding = 1
    else:
        shape_text = st.text_input("Input shape", value="1,16,16,16")
        generic_shape = _parse_shape_text(shape_text)
        col_axis, col_groups, col_kh, col_kw, col_stride, col_pad = st.columns(6)
        with col_axis:
            axis = st.number_input("Axis", min_value=-8, max_value=8, value=-1, step=1)
        with col_groups:
            groups = st.number_input("Groups", min_value=1, max_value=4096, value=1, step=1)
        with col_kh:
            kernel_h = st.number_input("Kernel H", min_value=1, max_value=31, value=2, step=1)
        with col_kw:
            kernel_w = st.number_input("Kernel W", min_value=1, max_value=31, value=2, step=1)
        with col_stride:
            stride = st.number_input("Stride", min_value=1, max_value=16, value=2, step=1)
        with col_pad:
            padding = st.number_input("Padding", min_value=0, max_value=64, value=1, step=1)
        matrix_m = 32
        matrix_k = 64
        matrix_n = 16
        input_shape = (1, 3, 16, 16)
        output_channels = 8
        kernel_size = (int(kernel_h), int(kernel_w))

    run_button = st.button("Run Benchmark", type="primary")

    if run_button:
        try:
            case = build_web_case(
                operator=operator,
                precision_modes=list(precision_modes),
                input_shape=input_shape,
                output_channels=int(output_channels),
                kernel_size=kernel_size,
                stride=int(stride),
                padding=int(padding),
                matrix_m=int(matrix_m),
                matrix_k=int(matrix_k),
                matrix_n=int(matrix_n),
                batch_size=int(batch_size),
                generic_shape=generic_shape,
                axis=int(axis),
                groups=int(groups),
                vocab_size=int(vocab_size),
                sequence_length=int(sequence_length),
                embedding_dim=int(embedding_dim),
                num_heads=int(num_heads),
            )
            config = build_web_config(case, warmup=int(warmup), runs=int(runs), output_dir=output_dir, backends=backends)
            with st.spinner("Running benchmarks locally..."):
                results, artifacts = run_web_benchmark(config)
        except Exception as exc:
            st.error(str(exc))
            return

        _render_results(results, artifacts, config.output_dir)
    else:
        st.info("Configure a benchmark, then run it locally from this browser session.")


def _render_results(results: list[BenchmarkResult], artifacts: dict[str, Path], output_dir: Path) -> None:
    rows = results_to_rows(results)
    st.success(f"Benchmark complete. Artifacts written to `{output_dir}`.")
    fastest = min(results, key=lambda item: item.latency_ms)
    slowest = max(results, key=lambda item: item.latency_ms)
    avg_error = sum(item.mean_abs_error for item in results) / len(results)
    col_fast, col_slow, col_error = st.columns(3)
    col_fast.metric("Fastest latency", f"{fastest.latency_ms:.4f} ms", fastest.backend)
    col_slow.metric("Slowest latency", f"{slowest.latency_ms:.4f} ms", slowest.backend)
    col_error.metric("Mean numerical error", f"{avg_error:.6f}")
    st.subheader("Summary")
    st.dataframe(rows, use_container_width=True)

    chart_rows = [
        {
            "case": f"{row['benchmark']} / {row['precision']}",
            "backend": row["backend"],
            "latency_ms": row["latency_ms"],
            "mean_abs_error": row["mean_abs_error"],
        }
        for row in rows
    ]
    col_latency, col_error = st.columns(2)
    with col_latency:
        st.subheader("Latency Comparison")
        _bar_chart(chart_rows, x="case", y="latency_ms", color="backend")
    with col_error:
        st.subheader("Numerical Error")
        _bar_chart(chart_rows, x="case", y="mean_abs_error", color="backend")

    report_text = artifacts["report"].read_text(encoding="utf-8")
    st.subheader("Markdown Report Preview")
    st.markdown(report_text)

    st.subheader("Downloads")
    col_csv, col_report, col_latency_png, col_error_png = st.columns(4)
    with col_csv:
        _download_file("summary.csv", artifacts["summary"], "text/csv")
    with col_report:
        _download_file("report.md", artifacts["report"], "text/markdown")
    with col_latency_png:
        _download_file("latency_plot.png", artifacts["latency_plot"], "image/png")
    with col_error_png:
        _download_file("error_plot.png", artifacts["error_plot"], "image/png")
    zip_source = artifacts.get("history_run", output_dir)
    zip_path = zip_directory(zip_source)
    _download_file("artifacts.zip", zip_path, "application/zip")


def _parse_shape_text(text: str) -> tuple[int, ...]:
    shape = tuple(int(part.strip()) for part in text.replace("x", ",").split(",") if part.strip())
    if not shape or any(item <= 0 for item in shape):
        raise ValueError("Input shape must contain positive integers.")
    return shape


def _download_file(label: str, path: Path, mime: str) -> None:
    if path.exists():
        st.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime)
    else:
        st.button(label, disabled=True)


def _bar_chart(rows: list[dict[str, Any]], x: str, y: str, color: str) -> None:
    try:
        import plotly.express as px

        fig = px.bar(rows, x=x, y=y, color=color, template="plotly_white")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(rows, x=x, y=y, color=color)


def _render_history_tools() -> None:
    runs = list_runs()
    with st.expander("History and run comparison", expanded=False):
        if len(runs) < 2:
            st.caption("Run at least two benchmarks with history enabled to compare them here.")
            return
        labels = [path.name for path in runs]
        col_base, col_candidate = st.columns(2)
        with col_base:
            baseline_label = st.selectbox("Baseline run", labels, index=1)
        with col_candidate:
            candidate_label = st.selectbox("Candidate run", labels, index=0)
        if st.button("Compare Selected Runs"):
            baseline = next(path for path in runs if path.name == baseline_label)
            candidate = next(path for path in runs if path.name == candidate_label)
            comparisons = compare_runs(baseline, candidate)
            artifacts = write_comparison(comparisons, "results/compare")
            st.dataframe(comparisons, use_container_width=True)
            col_csv, col_md = st.columns(2)
            with col_csv:
                _download_file("comparison.csv", artifacts["comparison_csv"], "text/csv")
            with col_md:
                _download_file("comparison.md", artifacts["comparison_report"], "text/markdown")


def _render_header(availability: dict[str, str]) -> None:
    available_count = sum(1 for backend in BACKEND_OPTIONS if availability.get(backend, "").startswith("available"))
    st.markdown(
        f"""
        <section class="teb-hero">
          <div>
            <p class="teb-eyebrow">LOCAL HARDWARE BENCHMARK CONSOLE</p>
            <h1>TinyEdgeBench</h1>
            <p class="teb-copy">Measure low-bit edge-AI operators on this machine's CPU/GPU runtime stack.</p>
          </div>
          <div class="teb-status-grid">
            <div><strong>105</strong><span>operators</span></div>
            <div><strong>36</strong><span>presets</span></div>
            <div><strong>{available_count}/{len(BACKEND_OPTIONS)}</strong><span>backends ready</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_backend_deck(availability: dict[str, str]) -> None:
    cards = []
    for backend in BACKEND_OPTIONS:
        status = availability.get(backend, "unknown")
        state = "ready" if status.startswith("available") else "pending"
        cards.append(
            f'<div class="teb-backend-card {state}">'
            f"<span>{state}</span>"
            f"<strong>{backend}</strong>"
            f"<em>{status}</em>"
            "</div>"
        )
    st.markdown(
        f"""
        <section class="teb-backend-deck">
          <div class="teb-section-head">
            <p>LOCAL RUNTIME MATRIX</p>
            <strong>Backend availability</strong>
          </div>
          <div class="teb-backend-grid">
            {''.join(cards)}
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --teb-bg: #f5f5f7;
          --teb-panel: rgba(255, 255, 255, 0.76);
          --teb-line: rgba(29, 29, 31, 0.1);
          --teb-line-strong: rgba(0, 113, 227, 0.34);
          --teb-text: #1d1d1f;
          --teb-muted: #6e6e73;
          --teb-cyan: #0071e3;
          --teb-green: #34c759;
          --teb-magenta: #af52de;
          --teb-radius-card: 20px;
          --teb-radius-panel: 28px;
          --teb-radius-pill: 999px;
          --teb-shadow-soft: 0 18px 42px rgba(0, 0, 0, 0.08);
          --teb-shadow-panel: 0 28px 80px rgba(0, 0, 0, 0.12);
        }

        .stApp {
          background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(245, 245, 247, 0.98) 44%, #ffffff),
            var(--teb-bg);
          color: var(--teb-text);
        }

        [data-testid="stSidebar"] {
          background: rgba(255, 255, 255, 0.72);
          border-right: 1px solid var(--teb-line);
          backdrop-filter: blur(24px) saturate(180%);
        }

        [data-testid="stHeader"] {
          background: rgba(255, 255, 255, 0.62);
          border-bottom: 1px solid rgba(29, 29, 31, 0.08);
          backdrop-filter: blur(22px) saturate(180%);
        }

        .block-container {
          max-width: 1240px;
          padding-top: 2rem;
        }

        .teb-hero {
          display: grid;
          grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
          gap: 24px;
          align-items: stretch;
          margin-bottom: 28px;
          padding: 28px;
          border: 1px solid var(--teb-line);
          border-radius: var(--teb-radius-panel);
          background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.7) 46%, rgba(0, 113, 227, 0.08)),
            rgba(255, 255, 255, 0.78);
          box-shadow: var(--teb-shadow-panel);
          position: relative;
          overflow: hidden;
          backdrop-filter: blur(24px) saturate(180%);
        }

        .teb-hero::before {
          position: absolute;
          top: 0;
          left: 24px;
          right: 24px;
          height: 1px;
          content: "";
          background: linear-gradient(90deg, transparent, rgba(0, 113, 227, 0.32), transparent);
        }

        .teb-eyebrow {
          margin: 0 0 10px;
          color: var(--teb-cyan);
          font-size: 12px;
          font-weight: 800;
          letter-spacing: 0;
        }

        .teb-hero h1 {
          margin: 0;
          color: var(--teb-text);
          font-size: 64px;
          line-height: 0.95;
          letter-spacing: 0;
        }

        .teb-copy {
          max-width: 720px;
          margin: 16px 0 0;
          color: var(--teb-muted);
          font-size: 18px;
          line-height: 1.55;
        }

        .teb-status-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
          align-content: end;
        }

        .teb-status-grid div,
        [data-testid="stMetric"] {
          border: 1px solid rgba(29, 29, 31, 0.08);
          border-radius: var(--teb-radius-card);
          background: rgba(255, 255, 255, 0.76);
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9), var(--teb-shadow-soft);
        }

        .teb-status-grid div {
          min-height: 102px;
          padding: 18px;
        }

        .teb-status-grid strong,
        .teb-status-grid span {
          display: block;
        }

        .teb-status-grid strong {
          font-size: 30px;
          color: var(--teb-text);
        }

        .teb-status-grid span {
          margin-top: 4px;
          color: var(--teb-muted);
          font-size: 13px;
        }

        .teb-backend-deck {
          margin-bottom: 24px;
          padding: 18px;
          border: 1px solid var(--teb-line);
          border-radius: var(--teb-radius-panel);
          background:
            linear-gradient(115deg, rgba(255, 255, 255, 0.94), rgba(255, 255, 255, 0.72)),
            rgba(255, 255, 255, 0.78);
          box-shadow: 0 24px 70px rgba(0, 0, 0, 0.1);
          backdrop-filter: blur(24px) saturate(180%);
        }

        .teb-section-head {
          display: flex;
          align-items: end;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 14px;
        }

        .teb-section-head p {
          margin: 0;
          color: var(--teb-green);
          font-size: 11px;
          font-weight: 800;
          letter-spacing: 0;
        }

        .teb-section-head strong {
          color: var(--teb-text);
          font-size: 18px;
        }

        .teb-backend-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 10px;
        }

        .teb-backend-card {
          min-height: 112px;
          padding: 14px;
          border: 1px solid rgba(29, 29, 31, 0.08);
          border-radius: var(--teb-radius-card);
          background: rgba(255, 255, 255, 0.78);
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
          overflow-wrap: anywhere;
        }

        .teb-backend-card.ready {
          border-color: rgba(52, 199, 89, 0.28);
        }

        .teb-backend-card span,
        .teb-backend-card strong,
        .teb-backend-card em {
          display: block;
        }

        .teb-backend-card span {
          width: fit-content;
          margin-bottom: 12px;
          padding: 4px 8px;
          border-radius: var(--teb-radius-pill);
          background: rgba(175, 82, 222, 0.1);
          color: var(--teb-magenta);
          font-size: 11px;
          font-style: normal;
          font-weight: 800;
          text-transform: uppercase;
        }

        .teb-backend-card.ready span {
          background: rgba(52, 199, 89, 0.1);
          color: var(--teb-green);
        }

        .teb-backend-card strong {
          color: var(--teb-text);
          font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
          font-size: 13px;
        }

        .teb-backend-card em {
          margin-top: 8px;
          color: var(--teb-muted);
          font-size: 12px;
          font-style: normal;
          line-height: 1.35;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
          border: 1px solid var(--teb-line-strong);
          border-radius: var(--teb-radius-pill);
          background: #0071e3;
          color: #ffffff;
          font-weight: 800;
          box-shadow: 0 14px 34px rgba(0, 113, 227, 0.18);
        }

        [data-testid="stDataFrame"],
        [data-testid="stMetric"],
        [data-testid="stAlert"] {
          border-radius: 8px;
        }

        [data-testid="stFileUploader"],
        [data-testid="stExpander"] {
          border: 1px solid rgba(29, 29, 31, 0.08);
          border-radius: var(--teb-radius-card);
          background: rgba(255, 255, 255, 0.68);
          box-shadow: 0 12px 34px rgba(0, 0, 0, 0.06);
        }

        h2, h3 {
          letter-spacing: 0;
        }

        @media (max-width: 900px) {
          .teb-hero,
          .teb-status-grid,
          .teb-backend-grid {
            grid-template-columns: 1fr;
          }

          .teb-hero h1 {
            font-size: 44px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

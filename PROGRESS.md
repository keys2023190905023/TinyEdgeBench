# Progress Log

## 2026-05-28

- Initialized TinyEdgeBench as a local Python package.
- Added YAML config mode and interactive CLI wizard mode.
- Added CPU-only NumPy implementations for FP32, simulated INT8, and shift-only operator experiments.
- Added CSV, Markdown report, latency plot, and error plot artifact generation.
- Added pytest coverage for default config execution, wizard input handling, operators, and report generation.
- Verified `python -m pytest` passes.
- Verified `python -m tinyedgebench.benchmark --config configs/default.yaml` generates the required `results/` artifacts.
- Verified the installed `tinyedgebench wizard` entry point can run from mocked terminal input.
- Verified a clean `.venv` install from the wheel build passes tests and runs both YAML and wizard workflows.

## 2026-05-28 Web UI

- Added `tinyedgebench.web_app` as a local Streamlit interface on top of the existing benchmark runner and artifact writers.
- Added `tinyedgebench web` command to launch the Streamlit app.
- Added browser-configurable operator parameters, precision modes, warmup runs, benchmark runs, and output directory.
- Added web output views for summary rows, latency and error charts, Markdown report preview, and artifact downloads.
- Added tests for web helper functions and Streamlit launch command construction.
- Verified `python -m pytest` passes with 9 tests.
- Verified the YAML benchmark command and mocked wizard workflow still work after adding Streamlit.
- Verified `tinyedgebench web` launches a local Streamlit app and the browser UI can run a benchmark successfully.
- Refreshed the clean `.venv` install from package metadata and verified tests plus YAML execution there.

## 2026-05-28 Operator Expansion

- Expanded supported operators from the original Conv2D/MatMul/depthwise set to 30+ local NumPy benchmark operators.
- Added common activation, pooling, normalization, tensor-shape, reduction, embedding, batch matmul, linear, pointwise convolution, and scaled dot-product attention benchmarks.
- Added lightweight network/block presets: `tiny_cnn`, `mobilenet_block`, `resnet_basic_block`, `transformer_encoder_tiny`, and `mlp_edge`.
- Added `configs/extended_operators.yaml` as a broader YAML example using both presets and individual operators.
- Updated the Streamlit UI with single-operator and network-preset modes.
- Added tests that run every supported operator through the benchmark runner and validate extended YAML/preset execution.
- Verified `python -m pytest` passes with 11 tests.
- Verified both `configs/default.yaml` and `configs/extended_operators.yaml` run successfully.
- Verified the active local Web UI exposes the new single-operator/network-preset selector.

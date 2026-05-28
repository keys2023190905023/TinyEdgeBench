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

## 2026-05-28 Larger Operator And Preset Catalog

- Expanded the catalog to 79 supported operator microbenchmarks.
- Added more activation, unary math, binary elementwise, reduction, normalization, layout, gather/one-hot, causal attention, and rotary embedding operators.
- Expanded network/block presets from 5 to 29 presets, covering CNN, MobileNet/EfficientNet, ConvNeXt, UNet, DeepLab, FPN, YOLO/PAN, ViT, Swin, BERT, GPT, recommendation, speech, autoencoder, generator, super-resolution, recurrent-gate, PointNet, GraphSAGE, and anomaly-detection style blocks.
- Added `configs/model_presets.yaml` for preset-heavy benchmarking.
- Verified `python -m pytest` passes with all supported operators and all presets exercised.
- Verified `configs/default.yaml`, `configs/extended_operators.yaml`, and `configs/model_presets.yaml` run successfully.

## 2026-05-28 Operator 100+ And Website

- Expanded supported operator microbenchmarks from 79 to 105.
- Added comparison, masking, sorting, top-k, cumulative reduction, GLU/SwiGLU/GEGLU, adaptive pooling, similarity, distance, additional unary math, and normalization-style operators.
- Added a GitHub Pages-ready static website under `docs/` with a generated hero image, animated signal canvas, operator catalog section, preset overview, workflow section, and install commands.
- Verified the static website through a local HTTP server and browser DOM inspection.

## 2026-05-28 Real Backend Comparison

- Added optional `torch_cpu` and `onnxruntime_cpu` backend comparison paths alongside the default NumPy `cpu` backend.
- Added `backends: [...]` YAML support for multi-backend benchmark runs.
- Added optional dependency extras: `torch`, `onnx`, and `real-backends`.
- Added `configs/real_backends.yaml` for local deployment-style FP32 backend comparison.
- Updated reports to clearly distinguish real backend FP32 measurements from simulated INT8 and shift-only modes.

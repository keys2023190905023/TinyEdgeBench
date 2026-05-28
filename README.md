# TinyEdgeBench

TinyEdgeBench is a local, CPU-first benchmark tool for low-bit edge-AI operators. It is designed for quick experiments on a developer laptop or edge box, where users want reproducible latency, approximation error, plots, and a small Markdown report without depending on CUDA or cloud services.

## Motivation

Edge-AI deployment often starts with practical questions: how fast is this operator on my machine, how much error does a low-bit approximation introduce, and what shape regime should I optimize next? TinyEdgeBench answers those questions with a compact benchmark loop for FP32, simulated INT8, and shift-only arithmetic experiments.

The current release focuses on NumPy CPU kernels so the tool is easy to install and inspect. It is not a replacement for vendor profilers or production inference runtimes; it is a lightweight baseline for operator-level exploration.

## Installation

```bash
python -m pip install -e ".[dev]"
```

TinyEdgeBench requires Python 3.9 or newer. CPU execution is supported by default. CUDA is checked only for reporting and is not required.

## Quick Start

Run the default benchmark suite:

```bash
python -m tinyedgebench.benchmark --config configs/default.yaml
```

The command writes:

- `results/summary.csv`
- `results/report.md`
- `results/latency_plot.png`
- `results/error_plot.png`

## YAML Usage

Create or edit a YAML file with one or more benchmark cases:

```yaml
output_dir: results
warmup: 2
runs: 5
backend: cpu
seed: 42
benchmarks:
  - name: conv2d_small
    operator: conv2d
    input_shape: [1, 3, 16, 16]
    output_channels: 8
    kernel_size: [3, 3]
    stride: 1
    padding: 1
    precision_modes: [fp32, int8_sim, shift_only]
  - name: matmul_small
    operator: matmul
    matrix_m: 32
    matrix_k: 64
    matrix_n: 16
    precision_modes: [fp32, int8_sim, shift_only]
```

Then run:

```bash
python -m tinyedgebench.benchmark --config path/to/config.yaml
```

Supported operators are:

- `conv2d`
- `matmul`
- `depthwise_conv2d`
- `pointwise_conv2d`
- `batch_matmul`
- `linear`
- activations: `relu`, `relu6`, `sigmoid`, `tanh`, `gelu`, `silu`, `leaky_relu`
- reductions and probabilities: `softmax`, `log_softmax`, `reduce_mean`, `reduce_sum`
- pooling and image layout: `maxpool2d`, `avgpool2d`, `global_avgpool2d`, `upsample_nearest2d`, `pad`
- normalization: `batchnorm2d`, `layernorm`, `rmsnorm`, `groupnorm`
- tensor ops: `add`, `mul`, `concat`, `transpose`, `reshape`, `flatten`
- sequence/model ops: `embedding`, `scaled_dot_product_attention`

Network and block presets can also be included in YAML:

```yaml
network_presets:
  - name: tiny_cnn
    precision_modes: [fp32, int8_sim, shift_only]
  - name: transformer_encoder_tiny
    precision_modes: [fp32, int8_sim]
```

Available presets are `tiny_cnn`, `mobilenet_block`, `resnet_basic_block`, `transformer_encoder_tiny`, and `mlp_edge`. See `configs/extended_operators.yaml` for a broader example.

Supported precision modes are:

- `fp32`
- `int8_sim`
- `shift_only`

## Interactive CLI Usage

After installation, launch the wizard:

```bash
tinyedgebench wizard
```

The wizard asks for the operator type, tensor or matrix shape, kernel size when needed, precision modes, backend, and output directory. The CPU backend is the default and currently supported backend.

## Web UI

TinyEdgeBench also includes a local Streamlit web application:

```bash
tinyedgebench web
```

The web UI runs on the user's own computer and executes benchmarks locally on the same CPU backend as the CLI. From the browser, users can choose either a single operator or a network/block preset, then configure tensor or matrix dimensions, precision modes, warmup runs, benchmark runs, and output directory. After a run, the app shows a summary table, latency and error charts, a Markdown report preview, and download buttons for `summary.csv`, `report.md`, `latency_plot.png`, and `error_plot.png`.

Advanced Streamlit arguments can be passed after `--`:

```bash
tinyedgebench web -- --server.port 8502
```

## Screenshots

Screenshots will be added here as the UI stabilizes.

## Example Output

`results/summary.csv` contains one row per benchmark and precision mode:

```csv
name,operator,precision,backend,input_description,latency_ms,throughput_ops_per_s,mean_abs_error,max_abs_error
conv2d_small,conv2d,fp32,cpu,"NCHW=(1, 3, 16, 16), out_channels=8, kernel=(3, 3), stride=1, padding=1",0.250000,44236800.00,0.00000000,0.00000000
```

`results/report.md` records system information, a Markdown result table, and links to latency and error plots.

## Development

Run tests:

```bash
pytest
```

Run the end-to-end default benchmark:

```bash
python -m tinyedgebench.benchmark --config configs/default.yaml
```

## License

MIT License. See `LICENSE`.

## Roadmap

- Add optional CUDA backends through CuPy or PyTorch when available.
- Add FPGA benchmarking adapters for exported operator traces and board-side timing logs.
- Add NPU benchmarking adapters for vendor SDK command-line runners.
- Add more operator families, including pooling, normalization, activation, and fused kernels.
- Add JSON export and historical comparison reports.

# TinyEdgeBench

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Backend](https://img.shields.io/badge/backend-CPU%20local-lightgrey)](#why-tinyedgebench)

TinyEdgeBench is a local benchmark tool for low-bit edge-AI operators. It lets you configure operators or common model blocks, run benchmarks on your own machine, and generate CSV results, plots, and a Markdown report.

The project is CPU-first, NumPy-based, and designed for fast operator-level exploration before moving to heavier runtime, GPU, FPGA, or NPU profiling stacks.

## Why TinyEdgeBench

Edge-AI work often starts with practical questions:

- How fast is this operator on my laptop or edge box?
- How much error does an INT8-style approximation introduce?
- Which layer family is the likely latency bottleneck?
- Can I get a quick local report without setting up CUDA or vendor SDKs?

TinyEdgeBench gives you a lightweight local baseline for those questions. It is not a production inference runtime; it is a small, inspectable benchmarking harness for early design and optimization work.

## Highlights

| Capability | Status |
| --- | --- |
| Local CPU execution | Supported by default |
| YAML benchmark configs | Supported |
| Interactive CLI wizard | Supported |
| Streamlit Web UI | Supported |
| CSV, Markdown, and PNG outputs | Supported |
| 100+ operator microbenchmarks | Supported |
| 25+ network/block presets | Supported |
| FP32 baseline | Supported |
| Real `torch_cpu` / `onnxruntime_cpu` comparison | Optional |
| Simulated INT8 | Supported |
| Shift-only approximation | Supported |
| CUDA/GPU execution | Future work |

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/keys2023190905023/TinyEdgeBench.git
cd TinyEdgeBench
python -m pip install -e ".[dev]"
```

TinyEdgeBench requires Python 3.9 or newer. CUDA is not required.

## Quick Start

Run the default benchmark suite:

```bash
python -m tinyedgebench.benchmark --config configs/default.yaml
```

Outputs are written to `results/`:

```text
results/
  summary.csv
  report.md
  latency_plot.png
  error_plot.png
```

## Web UI

Launch the local Streamlit application:

```bash
tinyedgebench web
```

Then open:

```text
http://localhost:8501
```

The Web UI runs locally on your own computer. From the browser you can choose:

- single-operator benchmarks
- network or model-block presets
- precision modes
- tensor or matrix shapes
- warmup runs and benchmark runs
- output directory

After a run, the app shows a summary table, latency chart, numerical error chart, Markdown report preview, and download buttons for generated artifacts.

To choose a different Streamlit port:

```bash
tinyedgebench web -- --server.port 8502
```

## Project Website

The repository includes a static, GitHub Pages-ready website in [docs/](docs/):

```text
docs/
  index.html
  styles.css
  app.js
  assets/hero-edge-bench.png
```

To publish it on GitHub, enable Pages in the repository settings and choose `main` plus the `/docs` folder as the source.

## CLI Wizard

Use the interactive terminal wizard:

```bash
tinyedgebench wizard
```

The wizard asks for the operator, shape parameters, precision modes, backend, and output directory. CPU is the default supported backend.

## YAML Usage

Create a benchmark config:

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

Run it:

```bash
python -m tinyedgebench.benchmark --config path/to/config.yaml
```

See [configs/default.yaml](configs/default.yaml), [configs/extended_operators.yaml](configs/extended_operators.yaml), and [configs/model_presets.yaml](configs/model_presets.yaml) for complete examples.

## Real Backend Comparison

By default, `cpu` uses the built-in NumPy benchmark path. TinyEdgeBench can also compare against real local deployment-style CPU kernels through optional backends:

| Backend | What it measures |
| --- | --- |
| `cpu` | Default NumPy CPU implementation |
| `torch_cpu` | PyTorch CPU operator kernels |
| `onnxruntime_cpu` | ONNX Runtime CPUExecutionProvider kernels |

Install optional backend dependencies:

```bash
python -m pip install -e ".[real-backends]"
```

Run a backend comparison suite:

```bash
python -m tinyedgebench.benchmark --config configs/real_backends.yaml
```

Example config:

```yaml
output_dir: results_real_backends
warmup: 2
runs: 10
backends: [cpu, torch_cpu, onnxruntime_cpu]
benchmarks:
  - name: deploy_matmul
    operator: matmul
    matrix_m: 128
    matrix_k: 256
    matrix_n: 128
    precision_modes: [fp32]
```

These backend rows are measured on your local machine and reflect the installed PyTorch or ONNX Runtime CPU kernels. ONNX Runtime benchmark graphs freeze weights as model initializers where practical, which is closer to deployment-style inference than feeding every tensor as an input. `int8_sim` and `shift_only` remain simulation modes unless a backend-specific quantized kernel is added.

## Network Presets

TinyEdgeBench can run lightweight suites that approximate common model blocks:

| Preset | Description |
| --- | --- |
| `tiny_cnn` | Conv/BN/ReLU/Pool/Linear image pipeline |
| `mobilenet_block` | Depthwise separable convolution block |
| `resnet_basic_block` | Residual Conv/BN/ReLU/Add block |
| `transformer_encoder_tiny` | Attention, normalization, MLP, and softmax block |
| `mlp_edge` | Small MLP-style matrix and activation block |
| `efficientnet_mbconv` | Mobile inverted bottleneck convolution block |
| `convnext_block` | ConvNeXt-style depthwise convolution and pointwise MLP block |
| `unet_encoder_block` | UNet downsampling encoder block |
| `unet_decoder_block` | UNet upsampling decoder block |
| `deeplab_aspp_tiny` | Tiny segmentation ASPP-style block |
| `fpn_lateral_block` | Feature pyramid lateral fusion block |
| `yolo_head_tiny` | Tiny detection head block |
| `detection_neck_pan` | PAN-style detection neck fusion block |
| `segmentation_head` | Lightweight semantic segmentation head |
| `vit_patch_embed` | Vision Transformer patch embedding block |
| `swin_window_attention_tiny` | Tiny Swin-style attention and MLP block |
| `bert_ffn_block` | BERT-style feed-forward block |
| `gpt_decoder_tiny` | Tiny causal decoder block |
| `recommender_embedding_mlp` | Embedding plus MLP recommendation block |
| `speech_command_cnn` | Small speech-command CNN block |
| `wav2vec_conv_frontend` | Speech representation frontend approximation |
| `autoencoder_bottleneck` | Encoder bottleneck and decoder projection block |
| `gan_generator_block` | Generator-style upsampling convolution block |
| `super_resolution_block` | Pixel-shuffle-like super-resolution block |
| `lstm_gate_block` | LSTM gate approximation block |
| `gru_gate_block` | GRU gate approximation block |
| `pointnet_mlp_block` | PointNet-style per-point MLP and global reduction block |
| `graphsage_mlp_block` | GraphSAGE-style aggregate and projection block |
| `anomaly_mlp` | Small anomaly-detection MLP block |

Example:

```yaml
network_presets:
  - name: tiny_cnn
    precision_modes: [fp32, int8_sim, shift_only]
  - name: transformer_encoder_tiny
    precision_modes: [fp32, int8_sim]
```

## Supported Operators

| Category | Operators |
| --- | --- |
| Convolution | `conv2d`, `depthwise_conv2d`, `pointwise_conv2d` |
| Matrix and linear | `matmul`, `batch_matmul`, `linear` |
| Activations | `relu`, `relu6`, `sigmoid`, `tanh`, `gelu`, `silu`, `leaky_relu`, `elu`, `selu`, `celu`, `softplus`, `softsign`, `hard_sigmoid`, `hard_swish`, `mish`, `prelu`, `glu`, `swiglu`, `geglu` |
| Pooling and image ops | `maxpool2d`, `avgpool2d`, `global_avgpool2d`, `upsample_nearest2d`, `pad` |
| Normalization | `batchnorm2d`, `layernorm`, `rmsnorm`, `groupnorm`, `instance_norm`, `l2_normalize` |
| Tensor ops | `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `bias_add`, `where`, `masked_fill`, `greater`, `less`, `equal`, `not_equal`, `concat`, `transpose`, `reshape`, `flatten`, `squeeze`, `expand_dims`, `tile`, `slice`, `gather`, `one_hot` |
| Layout/image transforms | `channel_shuffle`, `space_to_depth`, `depth_to_space` |
| Pooling extras | `adaptive_avgpool2d`, `adaptive_maxpool2d` |
| Reductions and probabilities | `softmax`, `log_softmax`, `reduce_mean`, `reduce_sum`, `reduce_max`, `reduce_min`, `reduce_prod`, `argmax`, `argmin`, `topk`, `sort`, `cumsum`, `cumprod` |
| Unary math | `identity`, `abs`, `neg`, `square`, `sqrt`, `rsqrt`, `exp`, `log`, `log1p`, `pow`, `sin`, `cos`, `reciprocal`, `floor`, `ceil`, `round`, `clip`, `sign`, `standardize`, `minmax_normalize`, `pixel_norm`, `dropout_inference` |
| Similarity and distance | `cosine_similarity`, `pairwise_distance` |
| Sequence/model ops | `embedding`, `scaled_dot_product_attention`, `causal_self_attention`, `rotary_embedding` |

## Precision Modes

| Mode | Meaning |
| --- | --- |
| `fp32` | Float32 reference path |
| `int8_sim` | Symmetric INT8-style quantization simulation with float dequantization |
| `shift_only` | Signed power-of-two operand approximation for shift-like experiments |

## Output Files

| File | Purpose |
| --- | --- |
| `summary.csv` | Machine-readable benchmark summary |
| `report.md` | Markdown report with system information and result table |
| `latency_plot.png` | Latency comparison chart |
| `error_plot.png` | Numerical error chart |

The report records operating system, Python version, CPU information when available, and whether CUDA is visible on the machine.

## Example CSV

```csv
name,operator,precision,backend,input_description,latency_ms,throughput_ops_per_s,mean_abs_error,max_abs_error
conv2d_small,conv2d,fp32,cpu,"NCHW=(1, 3, 16, 16), out_channels=8, kernel=(3, 3), stride=1, padding=1",0.250000,44236800.00,0.00000000,0.00000000
```

## Project Layout

```text
TinyEdgeBench/
  configs/                  YAML benchmark examples
  src/tinyedgebench/         package source
    benchmark.py             YAML entry point
    cli.py                   CLI commands
    web_app.py               Streamlit application
    runner.py                benchmark orchestration
    operators.py             NumPy operator implementations
    artifacts.py             CSV, report, and plot generation
    network_presets.py       common model-block presets
  tests/                     pytest suite
```

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

Run end-to-end examples:

```bash
python -m tinyedgebench.benchmark --config configs/default.yaml
python -m tinyedgebench.benchmark --config configs/extended_operators.yaml
python -m tinyedgebench.benchmark --config configs/model_presets.yaml
python -m tinyedgebench.benchmark --config configs/real_backends.yaml
```

## Screenshots

Screenshots will be added as the Web UI stabilizes.

## Roadmap

- Optional CUDA backends through CuPy or PyTorch
- FPGA benchmarking adapters for exported operator traces and board-side timing logs
- NPU benchmarking adapters for vendor SDK command-line runners
- More fused kernels and model-specific operator groups
- JSON export and historical comparison reports
- Richer Web UI presets and saved benchmark sessions

## License

TinyEdgeBench is released under the MIT License. See [LICENSE](LICENSE).

# TinyEdgeBench Report

## System Information

- Execution location: local machine running this benchmark command or Streamlit app.
- Operating system: Windows 10 (AMD64)
- Python version: 3.9.13 (main, Aug 25 2022, 23:51:50) [MSC v.1916 64 bit (AMD64)]
- CPU information: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
- CUDA available: True
- GPU information: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
- PyTorch version: 2.5.1+cu121
- PyTorch CUDA available: True
- PyTorch CUDA version: 12.1
- ONNX Runtime version: 1.19.2
- ONNX Runtime providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider

## Executive Summary

- Benchmarks executed: 10 result rows.
- Fastest row: `rtx4060_matmul_256` on `onnxruntime_cpu` / `fp32` at 0.0741 ms.
- Slowest row: `rtx4060_conv3x3_64ch` on `cpu` / `fp32` at 3.0196 ms.
- Highest mean absolute error: 0.000940.
- Highest observed process RSS: 842.758 MB.

## Backend Ranking

| Backend | Median Latency (ms) | Rows |
| --- | ---: | ---: |
| onnxruntime_cpu | 0.1853 | 2 |
| torch_cpu | 0.4667 | 2 |
| torch_cuda | 0.6118 | 2 |
| onnxruntime_cuda | 0.8502 | 2 |
| cpu | 1.6724 | 2 |

## Bottleneck Rows

| Benchmark | Operator | Precision | Backend | Latency (ms) |
| --- | --- | --- | --- | ---: |
| rtx4060_conv3x3_64ch | conv2d | fp32 | cpu | 3.0196 |
| rtx4060_conv3x3_64ch | conv2d | fp32 | onnxruntime_cuda | 1.1836 |
| rtx4060_conv3x3_64ch | conv2d | fp32 | torch_cuda | 0.7771 |
| rtx4060_conv3x3_64ch | conv2d | fp32 | torch_cpu | 0.6872 |
| rtx4060_matmul_256 | matmul | fp32 | onnxruntime_cuda | 0.5168 |

## Reproduce

Run the same YAML config on the target machine with:

```bash
python -m tinyedgebench.benchmark --config path/to/config.yaml --history
```

Compare two saved runs with:

```bash
tinyedgebench compare results/runs/<baseline> results/runs/<candidate>
```

## Results

| Benchmark | Operator | Precision | Backend | Median (ms) | P90 (ms) | Std (ms) | Peak RSS (MB) | Power (W) | Energy (mJ) | Mean Abs Error |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rtx4060_matmul_256 | matmul | fp32 | cpu | 0.3252 | 0.3670 | 0.0321 | 59.395 |  |  | 0.000000 |
| rtx4060_matmul_256 | matmul | fp32 | torch_cpu | 0.2461 | 0.2941 | 0.0339 | 396.516 |  |  | 0.000000 |
| rtx4060_matmul_256 | matmul | fp32 | torch_cuda | 0.4464 | 0.4993 | 0.0485 | 513.129 | 15.535 | 6.935 | 0.000001 |
| rtx4060_matmul_256 | matmul | fp32 | onnxruntime_cpu | 0.0741 | 0.1079 | 0.0199 | 539.156 |  |  | 0.000000 |
| rtx4060_matmul_256 | matmul | fp32 | onnxruntime_cuda | 0.5168 | 0.6255 | 0.0930 | 583.441 | 15.796 | 8.164 | 0.000940 |
| rtx4060_conv3x3_64ch | conv2d | fp32 | cpu | 3.0196 | 3.2448 | 0.1828 | 588.492 |  |  | 0.000000 |
| rtx4060_conv3x3_64ch | conv2d | fp32 | torch_cpu | 0.6872 | 0.7592 | 0.0701 | 593.977 |  |  | 0.000001 |
| rtx4060_conv3x3_64ch | conv2d | fp32 | torch_cuda | 0.7771 | 1.1133 | 0.1652 | 697.086 | 15.870 | 12.333 | 0.000001 |
| rtx4060_conv3x3_64ch | conv2d | fp32 | onnxruntime_cpu | 0.2965 | 0.3358 | 0.0559 | 705.352 |  |  | 0.000001 |
| rtx4060_conv3x3_64ch | conv2d | fp32 | onnxruntime_cuda | 1.1836 | 1.3105 | 0.1381 | 842.758 | 9.645 | 11.416 | 0.000001 |

## Plots

![Latency plot](latency_plot.png)

![Error plot](error_plot.png)

## Notes

- `int8_sim` uses symmetric int8 quantization with int32 accumulation and float dequantization.
- `shift_only` rounds operands to signed powers of two to approximate shift-only arithmetic.
- `cpu` is the default NumPy backend and remains available without optional dependencies.
- `torch_cpu`, `torch_cuda`, `onnxruntime_cpu`, and `onnxruntime_cuda` measure real local backend kernels when the matching optional dependencies and hardware are available.
- Real backend comparison currently reports FP32 timings; `int8_sim` and `shift_only` are simulation modes unless a backend-specific quantized kernel is added.
- CPU memory uses process RSS when `psutil` is installed; CUDA memory uses PyTorch peak allocation/reservation for `torch_cuda` runs.
- Power and energy are opportunistic estimates from `nvidia-smi power.draw` for CUDA-style backends; use an external meter or stable sampler for publishable energy numbers.
- GitHub Pages can showcase the project, but benchmark data is generated only on the machine where TinyEdgeBench is run.

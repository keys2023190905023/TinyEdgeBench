# TinyEdgeBench Benchmark Protocol

TinyEdgeBench uses a lightweight, reproducible protocol for operator-to-deployment benchmarking on local hardware.

## 1. Hardware Information

Every published run should include:

- CPU model and core count when available
- GPU/NPU/FPGA device name when used
- Memory capacity when available
- Board name, clock, bitstream, and measurement timer for FPGA trace runs
- Power meter or telemetry source for energy numbers

## 2. Software Environment

Record:

- Operating system
- Python version
- TinyEdgeBench version or commit
- NumPy, PyTorch, ONNX Runtime, OpenVINO, TVM, or TensorRT versions
- CUDA driver/runtime version when GPU backends are used
- Exact YAML config and command used to reproduce the run

## 3. Warmup Policy

Use at least two warmup runs for interactive exploration and at least five warmup runs for publishable comparisons. Warmup rows are not included in reported latency statistics.

## 4. Number Of Runs

Use at least 20 timed runs for stable operator comparisons. Tiny experiments can use fewer runs during development, but published result cards should disclose the run count.

## 5. Statistics

TinyEdgeBench reports:

- `latency_median_ms`
- `latency_p90_ms`
- `latency_std_ms`
- `valid_runs`
- `failed_runs`
- `oom_runs`

The primary ranking metric is median latency. P90 and standard deviation are used to identify unstable backends or noisy hardware conditions.

## 6. Synchronization Policy

CUDA-style backends must synchronize before and after each timed region. CPU and NumPy runs use wall-clock timing around the executor call.

## 7. Precision Mode Definitions

- `fp32`: float32 execution on the selected backend.
- `int8_sim`: symmetric int8 quantization with int32-style accumulation and float dequantization in the local simulation path.
- `shift_only`: operands are approximated by signed powers of two to study shift-friendly arithmetic.
- Real INT8 backend names such as `onnxruntime_int8_cpu`, `tensorrt_int8_calibrated`, or `fpga_int8_mac_trace` should only be used when the backend executes quantized kernels or board-side traces.

## 8. Error Metrics

Numerical error is measured against the local FP32 NumPy reference output:

- `mean_abs_error`
- `max_abs_error`

Model-level quality metrics are roadmap items and should be reported separately from operator error.

## 9. Memory Measurement

- CPU memory: process RSS via `psutil` when available.
- PyTorch CUDA memory: `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`.
- Other backends: use runtime-specific memory telemetry when available.

## 10. Energy Measurement

TinyEdgeBench can record opportunistic GPU power samples from `nvidia-smi power.draw` for CUDA-style runs. Treat these as engineering estimates. Publishable energy data should use a stable sampling interval or external power meter.

Report:

- `power_w`
- `energy_mj`
- `edp_mj_ms`

For FPGA/NPU boards, prefer board-side timers plus an external power meter or vendor telemetry.

## 11. Stage Timing

For end-to-end model benchmarks, separate:

- `preprocess_ms`
- `inference_ms`
- `postprocess_ms`

Operator microbenchmarks usually report inference time only.

## 12. Reproducibility Checklist

- Commit hash recorded
- YAML config included
- `summary.csv` included
- `report.md` included
- System information included
- Backend availability documented
- Power measurement source documented
- Any missing fields explicitly left blank rather than fabricated

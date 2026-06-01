# Verified Local CPU And GPU Results

TinyEdgeBench distinguishes verified local CPU/GPU measurements from example configs. A result is considered verified only when the run directory contains system information, the exact YAML/config command, and generated artifacts.

## Current Result Directories

| Platform | Directory | Status |
| --- | --- | --- |
| Local CPU baseline | `docs/results/cpu_baseline/` | Verified on the current development machine |
| RTX 4060 Laptop GPU | `docs/results/rtx4060_laptop/` | Verified when CUDA artifacts are present |

## Result Card Schema

| Platform | Backend | Workload | Precision | Key Result |
| --- | --- | --- | --- | --- |
| Laptop CPU | NumPy / Torch CPU / ONNX CPU | Conv / MatMul | FP32 / INT8-sim / Shift-only | Median latency, P90 latency, error, peak RSS |
| RTX 4060 Laptop | Torch CUDA / ONNX CUDA | MatMul / Conv | FP32 | Median latency, throughput, GPU memory, estimated energy |

## Local Execution Contract

TinyEdgeBench reports always belong to the machine that runs the command:

1. Install TinyEdgeBench on the target laptop, workstation, or server.
2. Run a CPU or GPU YAML config locally.
3. Keep `summary.csv`, `report.md`, plots, and `system.md` together.
4. Treat the result as evidence for that exact CPU/GPU/runtime stack.

## Required Files Per Hardware Run

- `system.md`
- `summary.csv`
- `report.md`
- `environment.yml` or equivalent package lock
- Plots when generated

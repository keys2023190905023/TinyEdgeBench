# Verified Hardware Results

TinyEdgeBench distinguishes verified local measurements from roadmap targets. A result is considered verified only when the run directory contains system information, the exact YAML/config command, and generated artifacts or board-side logs.

## Current Result Directories

| Platform | Directory | Status |
| --- | --- | --- |
| Local CPU baseline | `docs/results/cpu_baseline/` | Verified on the current development machine |
| RTX 4060 Laptop GPU | `docs/results/rtx4060_laptop/` | Verified when CUDA artifacts are present |
| PYNQ-Z2 FPGA | `docs/results/pynq_z2_fpga/` | Trace adapter scaffold, board run pending |

## Result Card Schema

| Platform | Backend | Workload | Precision | Key Result |
| --- | --- | --- | --- | --- |
| Laptop CPU | NumPy / Torch CPU / ONNX CPU | Conv / MatMul | FP32 / INT8-sim / Shift-only | Median latency, P90 latency, error, peak RSS |
| RTX 4060 Laptop | Torch CUDA / ONNX CUDA | MatMul / Conv | FP32 | Median latency, throughput, GPU memory, estimated energy |
| PYNQ-Z2 | FPGA trace adapter | Conv / MAC / shift-only op | INT8 / shift-only | Board-side latency and output error after trace replay |
| Optional NPU target | Vendor runtime adapter | YOLO block / MobileNet block | INT8 | End-to-end latency and memory when hardware is available |

## FPGA Trace Adapter Contract

The FPGA path is intentionally trace-first:

1. TinyEdgeBench exports operator metadata, input tensors, weights, and expected FP32 output.
2. The PYNQ-Z2 board runs the RTL/IP or Python-controlled overlay.
3. The board writes `board_latency_log.csv` and output tensors.
4. TinyEdgeBench imports board output, computes error, and records throughput.

This lets the project evaluate FPGA-friendly low-bit operators before implementing a full production runtime.

## Required Files Per Hardware Run

- `system.md`
- `summary.csv` or board-side latency CSV
- `report.md`
- `environment.yml` or equivalent package lock
- Plots when generated
- Raw board logs for FPGA/NPU trace adapters

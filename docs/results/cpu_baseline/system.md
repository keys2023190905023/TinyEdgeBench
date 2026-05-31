# CPU Baseline Verified Run

Status: verified local run.

## Command

```bash
python -m tinyedgebench.benchmark --config configs/verified_cpu_baseline.yaml
```

## Hardware

- Platform: Windows laptop CPU baseline
- CPU: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
- GPU: not used for this run

## Software

- Python: 3.9.13
- PyTorch: 2.5.1+cu121
- ONNX Runtime: 1.19.2
- Providers observed: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider

## Workloads

- `cpu_conv3x3_32ch`
- `cpu_matmul_128`

## Notes

This run compares the default NumPy CPU path with optional `torch_cpu` and `onnxruntime_cpu` FP32 backends. Low-bit simulation is still available through the default `cpu` backend, but this verified run focuses on real FP32 backend comparison.

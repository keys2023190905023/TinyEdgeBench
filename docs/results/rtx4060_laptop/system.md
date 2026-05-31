# RTX 4060 Laptop Verified Run

Status: verified local run.

## Command

```bash
python -m tinyedgebench.benchmark --config configs/verified_rtx4060.yaml
```

## Hardware

- Platform: Windows laptop
- CPU: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
- NVIDIA driver: 571.96

## Software

- Python: 3.9.13
- PyTorch: 2.5.1+cu121
- PyTorch CUDA: 12.1
- ONNX Runtime: 1.19.2
- ONNX Runtime providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider

## Workloads

- `rtx4060_matmul_256`
- `rtx4060_conv3x3_64ch`

## Notes

This run includes `cpu`, `torch_cpu`, `torch_cuda`, `onnxruntime_cpu`, and `onnxruntime_cuda` FP32 measurements. Power and energy fields are opportunistic estimates from `nvidia-smi power.draw`, not calibrated lab-grade measurements.

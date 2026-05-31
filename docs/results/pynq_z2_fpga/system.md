# PYNQ-Z2 FPGA Result Scaffold

Status: board-side run pending.

This directory defines the artifact contract for future PYNQ-Z2 validation. It should not be treated as a verified hardware result until `board_latency_log.csv`, `operator_trace.csv`, board output tensors, and a completed `report.md` are added from an actual board run.

## Target

- Board: PYNQ-Z2
- FPGA adapter: `fpga_shift_only_trace` / `fpga_int8_mac_trace` planned
- Measurement: board-side timer plus optional external USB power meter

## Required Board Artifacts

- `board_latency_log.csv`
- `operator_trace.csv`
- output tensor dump
- board clock / bitstream metadata
- power measurement source if energy is reported

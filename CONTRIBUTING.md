# Contributing

TinyEdgeBench welcomes focused contributions that make low-bit edge-AI benchmarking more reproducible.

## Good First Contributions

- Add a benchmark suite YAML under `benchmark_suites/`.
- Add a verified hardware result directory under `docs/results/`.
- Improve backend availability checks.
- Add tests for a new operator or report field.
- Document a hardware measurement protocol.

## Result Contribution Rules

Only mark a result as verified when it includes:

- `system.md`
- `summary.csv`
- `report.md`
- environment metadata
- the exact command used

Leave unavailable energy or memory fields blank rather than guessing.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m tinyedgebench.benchmark --config configs/default.yaml
```

## Style

- Keep benchmark logic testable and backend-agnostic.
- Prefer small, reproducible YAML suites over large opaque scripts.
- Do not fabricate hardware measurements.
- Update `PROGRESS.md` when adding major capabilities.

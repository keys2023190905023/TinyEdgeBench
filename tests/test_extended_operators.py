from pathlib import Path

from tinyedgebench.benchmark import run_config
from tinyedgebench.config import BenchmarkCase, BenchmarkConfig, SUPPORTED_OPERATORS, load_config
from tinyedgebench.network_presets import NETWORK_PRESETS, build_network_preset
from tinyedgebench.runner import run_benchmarks


def test_all_supported_operators_run_fp32() -> None:
    cases = [
        BenchmarkCase("conv2d", "conv2d", ["fp32"], input_shape=(1, 2, 8, 8), output_channels=3, kernel_size=(3, 3), padding=1),
        BenchmarkCase("depthwise_conv2d", "depthwise_conv2d", ["fp32"], input_shape=(1, 2, 8, 8), kernel_size=(3, 3), padding=1),
        BenchmarkCase("pointwise_conv2d", "pointwise_conv2d", ["fp32"], input_shape=(1, 2, 8, 8), output_channels=3, kernel_size=(1, 1)),
        BenchmarkCase("matmul", "matmul", ["fp32"], matrix_m=4, matrix_k=5, matrix_n=3),
        BenchmarkCase("batch_matmul", "batch_matmul", ["fp32"], batch_size=2, matrix_m=4, matrix_k=5, matrix_n=3),
        BenchmarkCase("linear", "linear", ["fp32"], matrix_m=4, matrix_k=5, matrix_n=3),
        BenchmarkCase("embedding", "embedding", ["fp32"], batch_size=1, sequence_length=4, vocab_size=16, embedding_dim=8),
        BenchmarkCase("attention", "scaled_dot_product_attention", ["fp32"], batch_size=1, sequence_length=4, embedding_dim=8, num_heads=2),
        BenchmarkCase("causal_attention", "causal_self_attention", ["fp32"], batch_size=1, sequence_length=4, embedding_dim=8, num_heads=2),
    ]
    generic_ops = sorted(SUPPORTED_OPERATORS - {case.operator for case in cases})
    for op in generic_ops:
        image_ops = {
            "maxpool2d",
            "avgpool2d",
            "global_avgpool2d",
            "batchnorm2d",
            "groupnorm",
            "instance_norm",
            "upsample_nearest2d",
            "pad",
            "channel_shuffle",
            "space_to_depth",
            "depth_to_space",
            "adaptive_avgpool2d",
            "adaptive_maxpool2d",
            "pixel_norm",
        }
        shape = (1, 4, 8, 8) if op in image_ops else (2, 4, 8)
        cases.append(
            BenchmarkCase(
                op,
                op,
                ["fp32"],
                input_shape_generic=shape,
                kernel_size=(2, 2) if op in {"maxpool2d", "avgpool2d"} else None,
                stride=2,
                groups=2,
                target_shape=shape if op == "reshape" else None,
            )
        )

    results = run_benchmarks(BenchmarkConfig(warmup=0, runs=1, benchmarks=cases))

    assert {result.operator for result in results} == SUPPORTED_OPERATORS


def test_network_presets_and_extended_config_run(tmp_path: Path) -> None:
    for preset_name in NETWORK_PRESETS:
        config_for_preset = BenchmarkConfig(warmup=0, runs=1, benchmarks=build_network_preset(preset_name, ["fp32"]))
        assert run_benchmarks(config_for_preset)

    config = load_config("configs/extended_operators.yaml")
    config = BenchmarkConfig(
        output_dir=tmp_path,
        warmup=0,
        runs=1,
        backend=config.backend,
        seed=config.seed,
        benchmarks=config.benchmarks,
    )
    artifacts = run_config(config)

    assert artifacts["summary"].exists()
    assert artifacts["report"].exists()

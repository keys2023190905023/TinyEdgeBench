from __future__ import annotations

from tinyedgebench.config import BenchmarkCase


NETWORK_PRESETS = {
    "tiny_cnn": "Conv/BN/ReLU/Pool/Linear image pipeline",
    "mobilenet_block": "Depthwise separable convolution block",
    "resnet_basic_block": "Residual Conv/BN/ReLU/Add block",
    "transformer_encoder_tiny": "Attention, normalization, MLP, and softmax block",
    "mlp_edge": "Small MLP-style matrix and activation block",
}


def build_network_preset(name: str, precision_modes: list[str] | None = None) -> list[BenchmarkCase]:
    modes = precision_modes or ["fp32", "int8_sim", "shift_only"]
    if name == "tiny_cnn":
        return [
            BenchmarkCase("tiny_cnn_conv3x3", "conv2d", modes, input_shape=(1, 3, 32, 32), output_channels=16, kernel_size=(3, 3), padding=1),
            BenchmarkCase("tiny_cnn_batchnorm", "batchnorm2d", modes, input_shape_generic=(1, 16, 32, 32)),
            BenchmarkCase("tiny_cnn_relu", "relu", modes, input_shape_generic=(1, 16, 32, 32)),
            BenchmarkCase("tiny_cnn_maxpool", "maxpool2d", modes, input_shape_generic=(1, 16, 32, 32), kernel_size=(2, 2), stride=2),
            BenchmarkCase("tiny_cnn_global_avgpool", "global_avgpool2d", modes, input_shape_generic=(1, 16, 16, 16)),
            BenchmarkCase("tiny_cnn_classifier", "linear", modes, matrix_m=1, matrix_k=16, matrix_n=10),
        ]
    if name == "mobilenet_block":
        return [
            BenchmarkCase("mobilenet_depthwise", "depthwise_conv2d", modes, input_shape=(1, 16, 32, 32), kernel_size=(3, 3), padding=1),
            BenchmarkCase("mobilenet_pointwise", "pointwise_conv2d", modes, input_shape=(1, 16, 32, 32), output_channels=32, kernel_size=(1, 1)),
            BenchmarkCase("mobilenet_relu6", "relu6", modes, input_shape_generic=(1, 32, 32, 32)),
            BenchmarkCase("mobilenet_avgpool", "avgpool2d", modes, input_shape_generic=(1, 32, 32, 32), kernel_size=(2, 2), stride=2),
        ]
    if name == "resnet_basic_block":
        return [
            BenchmarkCase("resnet_conv1", "conv2d", modes, input_shape=(1, 16, 32, 32), output_channels=16, kernel_size=(3, 3), padding=1),
            BenchmarkCase("resnet_bn1", "batchnorm2d", modes, input_shape_generic=(1, 16, 32, 32)),
            BenchmarkCase("resnet_relu1", "relu", modes, input_shape_generic=(1, 16, 32, 32)),
            BenchmarkCase("resnet_conv2", "conv2d", modes, input_shape=(1, 16, 32, 32), output_channels=16, kernel_size=(3, 3), padding=1),
            BenchmarkCase("resnet_bn2", "batchnorm2d", modes, input_shape_generic=(1, 16, 32, 32)),
            BenchmarkCase("resnet_skip_add", "add", modes, input_shape_generic=(1, 16, 32, 32)),
        ]
    if name == "transformer_encoder_tiny":
        return [
            BenchmarkCase("transformer_layernorm", "layernorm", modes, input_shape_generic=(1, 16, 64)),
            BenchmarkCase("transformer_attention", "scaled_dot_product_attention", modes, batch_size=1, sequence_length=16, embedding_dim=64, num_heads=4),
            BenchmarkCase("transformer_projection", "linear", modes, matrix_m=16, matrix_k=64, matrix_n=64),
            BenchmarkCase("transformer_ffn1", "linear", modes, matrix_m=16, matrix_k=64, matrix_n=128),
            BenchmarkCase("transformer_gelu", "gelu", modes, input_shape_generic=(1, 16, 128)),
            BenchmarkCase("transformer_ffn2", "linear", modes, matrix_m=16, matrix_k=128, matrix_n=64),
            BenchmarkCase("transformer_softmax", "softmax", modes, input_shape_generic=(1, 4, 16, 16), axis=-1),
        ]
    if name == "mlp_edge":
        return [
            BenchmarkCase("mlp_linear1", "linear", modes, matrix_m=8, matrix_k=64, matrix_n=128),
            BenchmarkCase("mlp_silu", "silu", modes, input_shape_generic=(8, 128)),
            BenchmarkCase("mlp_linear2", "linear", modes, matrix_m=8, matrix_k=128, matrix_n=32),
            BenchmarkCase("mlp_layernorm", "layernorm", modes, input_shape_generic=(8, 32)),
            BenchmarkCase("mlp_softmax", "softmax", modes, input_shape_generic=(8, 32), axis=-1),
        ]
    raise ValueError(f"Unknown network preset: {name}")

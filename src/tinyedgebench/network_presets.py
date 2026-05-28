from __future__ import annotations

from tinyedgebench.config import BenchmarkCase


NETWORK_PRESETS = {
    "tiny_cnn": "Conv/BN/ReLU/Pool/Linear image pipeline",
    "mobilenet_block": "Depthwise separable convolution block",
    "resnet_basic_block": "Residual Conv/BN/ReLU/Add block",
    "transformer_encoder_tiny": "Attention, normalization, MLP, and softmax block",
    "mlp_edge": "Small MLP-style matrix and activation block",
    "efficientnet_mbconv": "Mobile inverted bottleneck convolution block",
    "convnext_block": "ConvNeXt-style depthwise convolution and pointwise MLP block",
    "unet_encoder_block": "UNet downsampling encoder block",
    "unet_decoder_block": "UNet upsampling decoder block",
    "deeplab_aspp_tiny": "Tiny atrous spatial pyramid style segmentation block",
    "fpn_lateral_block": "Feature pyramid lateral fusion block",
    "yolo_head_tiny": "Tiny detection head style convolution block",
    "detection_neck_pan": "PAN-style detection neck fusion block",
    "segmentation_head": "Lightweight semantic segmentation head",
    "vit_patch_embed": "Vision Transformer patch embedding and normalization block",
    "swin_window_attention_tiny": "Tiny Swin-style attention and MLP block",
    "bert_ffn_block": "BERT-style feed-forward and normalization block",
    "gpt_decoder_tiny": "Tiny decoder attention and MLP block",
    "recommender_embedding_mlp": "Embedding plus MLP recommendation block",
    "speech_command_cnn": "Small speech-command convolution block",
    "wav2vec_conv_frontend": "Speech representation convolution frontend approximation",
    "autoencoder_bottleneck": "Encoder bottleneck and decoder projection block",
    "gan_generator_block": "Generator-style upsampling convolution block",
    "super_resolution_block": "Pixel-shuffle-like super-resolution block",
    "lstm_gate_block": "LSTM gate approximation with linear, sigmoid, tanh, and elementwise ops",
    "gru_gate_block": "GRU gate approximation with linear and elementwise ops",
    "pointnet_mlp_block": "PointNet-style per-point MLP and global reduction block",
    "graphsage_mlp_block": "GraphSAGE-style aggregate and projection block",
    "anomaly_mlp": "Small anomaly-detection MLP block",
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
    if name == "efficientnet_mbconv":
        return [
            BenchmarkCase("mbconv_expand", "pointwise_conv2d", modes, input_shape=(1, 16, 32, 32), output_channels=64, kernel_size=(1, 1)),
            BenchmarkCase("mbconv_depthwise", "depthwise_conv2d", modes, input_shape=(1, 64, 32, 32), kernel_size=(3, 3), padding=1),
            BenchmarkCase("mbconv_squeeze", "global_avgpool2d", modes, input_shape_generic=(1, 64, 32, 32)),
            BenchmarkCase("mbconv_hardswish", "hard_swish", modes, input_shape_generic=(1, 64, 32, 32)),
            BenchmarkCase("mbconv_project", "pointwise_conv2d", modes, input_shape=(1, 64, 32, 32), output_channels=16, kernel_size=(1, 1)),
        ]
    if name == "convnext_block":
        return [
            BenchmarkCase("convnext_depthwise7x7", "depthwise_conv2d", modes, input_shape=(1, 32, 32, 32), kernel_size=(7, 7), padding=3),
            BenchmarkCase("convnext_layernorm", "layernorm", modes, input_shape_generic=(1, 32, 32, 32)),
            BenchmarkCase("convnext_expand", "pointwise_conv2d", modes, input_shape=(1, 32, 32, 32), output_channels=128, kernel_size=(1, 1)),
            BenchmarkCase("convnext_gelu", "gelu", modes, input_shape_generic=(1, 128, 32, 32)),
            BenchmarkCase("convnext_project", "pointwise_conv2d", modes, input_shape=(1, 128, 32, 32), output_channels=32, kernel_size=(1, 1)),
        ]
    if name == "unet_encoder_block":
        return [
            BenchmarkCase("unet_enc_conv1", "conv2d", modes, input_shape=(1, 8, 64, 64), output_channels=16, kernel_size=(3, 3), padding=1),
            BenchmarkCase("unet_enc_relu1", "relu", modes, input_shape_generic=(1, 16, 64, 64)),
            BenchmarkCase("unet_enc_conv2", "conv2d", modes, input_shape=(1, 16, 64, 64), output_channels=16, kernel_size=(3, 3), padding=1),
            BenchmarkCase("unet_enc_relu2", "relu", modes, input_shape_generic=(1, 16, 64, 64)),
            BenchmarkCase("unet_enc_pool", "maxpool2d", modes, input_shape_generic=(1, 16, 64, 64), kernel_size=(2, 2), stride=2),
        ]
    if name == "unet_decoder_block":
        return [
            BenchmarkCase("unet_dec_upsample", "upsample_nearest2d", modes, input_shape_generic=(1, 32, 32, 32), scale_factor=2),
            BenchmarkCase("unet_dec_concat", "concat", modes, input_shape_generic=(1, 32, 64, 64), axis=1),
            BenchmarkCase("unet_dec_conv1", "conv2d", modes, input_shape=(1, 64, 64, 64), output_channels=32, kernel_size=(3, 3), padding=1),
            BenchmarkCase("unet_dec_relu", "relu", modes, input_shape_generic=(1, 32, 64, 64)),
            BenchmarkCase("unet_dec_conv2", "conv2d", modes, input_shape=(1, 32, 64, 64), output_channels=16, kernel_size=(3, 3), padding=1),
        ]
    if name == "deeplab_aspp_tiny":
        return [
            BenchmarkCase("aspp_conv1x1", "pointwise_conv2d", modes, input_shape=(1, 32, 32, 32), output_channels=16, kernel_size=(1, 1)),
            BenchmarkCase("aspp_conv3x3_a", "conv2d", modes, input_shape=(1, 32, 32, 32), output_channels=16, kernel_size=(3, 3), padding=1),
            BenchmarkCase("aspp_conv3x3_b", "conv2d", modes, input_shape=(1, 32, 32, 32), output_channels=16, kernel_size=(3, 3), padding=1),
            BenchmarkCase("aspp_concat", "concat", modes, input_shape_generic=(1, 16, 32, 32), axis=1),
            BenchmarkCase("aspp_project", "pointwise_conv2d", modes, input_shape=(1, 32, 32, 32), output_channels=16, kernel_size=(1, 1)),
        ]
    if name == "fpn_lateral_block":
        return [
            BenchmarkCase("fpn_lateral_conv", "pointwise_conv2d", modes, input_shape=(1, 64, 32, 32), output_channels=32, kernel_size=(1, 1)),
            BenchmarkCase("fpn_topdown_upsample", "upsample_nearest2d", modes, input_shape_generic=(1, 32, 16, 16), scale_factor=2),
            BenchmarkCase("fpn_add", "add", modes, input_shape_generic=(1, 32, 32, 32)),
            BenchmarkCase("fpn_smooth", "conv2d", modes, input_shape=(1, 32, 32, 32), output_channels=32, kernel_size=(3, 3), padding=1),
        ]
    if name == "yolo_head_tiny":
        return [
            BenchmarkCase("yolo_neck_conv", "conv2d", modes, input_shape=(1, 64, 20, 20), output_channels=64, kernel_size=(3, 3), padding=1),
            BenchmarkCase("yolo_silu", "silu", modes, input_shape_generic=(1, 64, 20, 20)),
            BenchmarkCase("yolo_pred_conv", "pointwise_conv2d", modes, input_shape=(1, 64, 20, 20), output_channels=45, kernel_size=(1, 1)),
            BenchmarkCase("yolo_sigmoid", "sigmoid", modes, input_shape_generic=(1, 45, 20, 20)),
        ]
    if name == "detection_neck_pan":
        return [
            BenchmarkCase("pan_upsample", "upsample_nearest2d", modes, input_shape_generic=(1, 64, 20, 20), scale_factor=2),
            BenchmarkCase("pan_concat", "concat", modes, input_shape_generic=(1, 64, 40, 40), axis=1),
            BenchmarkCase("pan_conv1", "conv2d", modes, input_shape=(1, 128, 40, 40), output_channels=64, kernel_size=(3, 3), padding=1),
            BenchmarkCase("pan_hardswish", "hard_swish", modes, input_shape_generic=(1, 64, 40, 40)),
        ]
    if name == "segmentation_head":
        return [
            BenchmarkCase("seg_conv3x3", "conv2d", modes, input_shape=(1, 32, 64, 64), output_channels=32, kernel_size=(3, 3), padding=1),
            BenchmarkCase("seg_batchnorm", "batchnorm2d", modes, input_shape_generic=(1, 32, 64, 64)),
            BenchmarkCase("seg_relu", "relu", modes, input_shape_generic=(1, 32, 64, 64)),
            BenchmarkCase("seg_logits", "pointwise_conv2d", modes, input_shape=(1, 32, 64, 64), output_channels=8, kernel_size=(1, 1)),
            BenchmarkCase("seg_upsample", "upsample_nearest2d", modes, input_shape_generic=(1, 8, 64, 64), scale_factor=2),
        ]
    if name == "vit_patch_embed":
        return [
            BenchmarkCase("vit_patch_conv", "conv2d", modes, input_shape=(1, 3, 32, 32), output_channels=64, kernel_size=(4, 4), stride=4),
            BenchmarkCase("vit_flatten", "flatten", modes, input_shape_generic=(1, 64, 8, 8)),
            BenchmarkCase("vit_layernorm", "layernorm", modes, input_shape_generic=(1, 64, 64)),
            BenchmarkCase("vit_projection", "linear", modes, matrix_m=64, matrix_k=64, matrix_n=64),
        ]
    if name == "swin_window_attention_tiny":
        return [
            BenchmarkCase("swin_layernorm", "layernorm", modes, input_shape_generic=(1, 49, 64)),
            BenchmarkCase("swin_attention", "scaled_dot_product_attention", modes, batch_size=1, sequence_length=49, embedding_dim=64, num_heads=4),
            BenchmarkCase("swin_mlp1", "linear", modes, matrix_m=49, matrix_k=64, matrix_n=128),
            BenchmarkCase("swin_gelu", "gelu", modes, input_shape_generic=(1, 49, 128)),
            BenchmarkCase("swin_mlp2", "linear", modes, matrix_m=49, matrix_k=128, matrix_n=64),
        ]
    if name == "bert_ffn_block":
        return [
            BenchmarkCase("bert_layernorm", "layernorm", modes, input_shape_generic=(1, 16, 64)),
            BenchmarkCase("bert_linear1", "linear", modes, matrix_m=16, matrix_k=64, matrix_n=256),
            BenchmarkCase("bert_gelu", "gelu", modes, input_shape_generic=(1, 16, 256)),
            BenchmarkCase("bert_linear2", "linear", modes, matrix_m=16, matrix_k=256, matrix_n=64),
            BenchmarkCase("bert_residual_add", "add", modes, input_shape_generic=(1, 16, 64)),
        ]
    if name == "gpt_decoder_tiny":
        return [
            BenchmarkCase("gpt_rotary", "rotary_embedding", modes, input_shape_generic=(1, 4, 16, 16)),
            BenchmarkCase("gpt_causal_attention", "causal_self_attention", modes, batch_size=1, sequence_length=16, embedding_dim=64, num_heads=4),
            BenchmarkCase("gpt_linear", "linear", modes, matrix_m=16, matrix_k=64, matrix_n=64),
            BenchmarkCase("gpt_ffn", "linear", modes, matrix_m=16, matrix_k=64, matrix_n=256),
            BenchmarkCase("gpt_silu", "silu", modes, input_shape_generic=(1, 16, 256)),
        ]
    if name == "recommender_embedding_mlp":
        return [
            BenchmarkCase("rec_embedding", "embedding", modes, batch_size=4, sequence_length=8, vocab_size=512, embedding_dim=32),
            BenchmarkCase("rec_flatten", "flatten", modes, input_shape_generic=(4, 8, 32)),
            BenchmarkCase("rec_dense1", "linear", modes, matrix_m=4, matrix_k=256, matrix_n=64),
            BenchmarkCase("rec_relu", "relu", modes, input_shape_generic=(4, 64)),
            BenchmarkCase("rec_dense2", "linear", modes, matrix_m=4, matrix_k=64, matrix_n=1),
        ]
    if name == "speech_command_cnn":
        return [
            BenchmarkCase("speech_conv1", "conv2d", modes, input_shape=(1, 1, 40, 64), output_channels=8, kernel_size=(3, 3), padding=1),
            BenchmarkCase("speech_relu1", "relu", modes, input_shape_generic=(1, 8, 40, 64)),
            BenchmarkCase("speech_pool", "avgpool2d", modes, input_shape_generic=(1, 8, 40, 64), kernel_size=(2, 2), stride=2),
            BenchmarkCase("speech_conv2", "conv2d", modes, input_shape=(1, 8, 20, 32), output_channels=16, kernel_size=(3, 3), padding=1),
            BenchmarkCase("speech_classifier", "linear", modes, matrix_m=1, matrix_k=16, matrix_n=12),
        ]
    if name == "wav2vec_conv_frontend":
        return [
            BenchmarkCase("wav_frontend_conv_a", "conv2d", modes, input_shape=(1, 1, 32, 128), output_channels=16, kernel_size=(3, 3), stride=2, padding=1),
            BenchmarkCase("wav_frontend_gelu", "gelu", modes, input_shape_generic=(1, 16, 16, 64)),
            BenchmarkCase("wav_frontend_conv_b", "conv2d", modes, input_shape=(1, 16, 16, 64), output_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            BenchmarkCase("wav_frontend_layernorm", "layernorm", modes, input_shape_generic=(1, 32, 8, 32)),
        ]
    if name == "autoencoder_bottleneck":
        return [
            BenchmarkCase("ae_encoder", "linear", modes, matrix_m=4, matrix_k=128, matrix_n=32),
            BenchmarkCase("ae_tanh", "tanh", modes, input_shape_generic=(4, 32)),
            BenchmarkCase("ae_bottleneck", "linear", modes, matrix_m=4, matrix_k=32, matrix_n=16),
            BenchmarkCase("ae_decoder", "linear", modes, matrix_m=4, matrix_k=16, matrix_n=128),
            BenchmarkCase("ae_sigmoid", "sigmoid", modes, input_shape_generic=(4, 128)),
        ]
    if name == "gan_generator_block":
        return [
            BenchmarkCase("gan_dense", "linear", modes, matrix_m=1, matrix_k=64, matrix_n=1024),
            BenchmarkCase("gan_reshape", "reshape", modes, input_shape_generic=(1, 1024), target_shape=(1, 64, 4, 4)),
            BenchmarkCase("gan_upsample", "upsample_nearest2d", modes, input_shape_generic=(1, 64, 4, 4), scale_factor=2),
            BenchmarkCase("gan_conv", "conv2d", modes, input_shape=(1, 64, 8, 8), output_channels=32, kernel_size=(3, 3), padding=1),
            BenchmarkCase("gan_tanh", "tanh", modes, input_shape_generic=(1, 32, 8, 8)),
        ]
    if name == "super_resolution_block":
        return [
            BenchmarkCase("sr_conv_expand", "conv2d", modes, input_shape=(1, 3, 32, 32), output_channels=12, kernel_size=(3, 3), padding=1),
            BenchmarkCase("sr_depth_to_space", "depth_to_space", modes, input_shape_generic=(1, 12, 32, 32), scale_factor=2),
            BenchmarkCase("sr_relu", "relu", modes, input_shape_generic=(1, 3, 64, 64)),
            BenchmarkCase("sr_refine", "conv2d", modes, input_shape=(1, 3, 64, 64), output_channels=3, kernel_size=(3, 3), padding=1),
        ]
    if name == "lstm_gate_block":
        return [
            BenchmarkCase("lstm_input_gate", "linear", modes, matrix_m=8, matrix_k=64, matrix_n=64),
            BenchmarkCase("lstm_sigmoid_i", "sigmoid", modes, input_shape_generic=(8, 64)),
            BenchmarkCase("lstm_candidate", "tanh", modes, input_shape_generic=(8, 64)),
            BenchmarkCase("lstm_mul", "mul", modes, input_shape_generic=(8, 64)),
            BenchmarkCase("lstm_state_add", "add", modes, input_shape_generic=(8, 64)),
        ]
    if name == "gru_gate_block":
        return [
            BenchmarkCase("gru_reset_gate", "linear", modes, matrix_m=8, matrix_k=64, matrix_n=64),
            BenchmarkCase("gru_sigmoid", "sigmoid", modes, input_shape_generic=(8, 64)),
            BenchmarkCase("gru_candidate", "tanh", modes, input_shape_generic=(8, 64)),
            BenchmarkCase("gru_mix_mul", "mul", modes, input_shape_generic=(8, 64)),
            BenchmarkCase("gru_mix_add", "add", modes, input_shape_generic=(8, 64)),
        ]
    if name == "pointnet_mlp_block":
        return [
            BenchmarkCase("pointnet_linear1", "linear", modes, matrix_m=128, matrix_k=3, matrix_n=64),
            BenchmarkCase("pointnet_batchnorm", "batchnorm2d", modes, input_shape_generic=(1, 64, 1, 128)),
            BenchmarkCase("pointnet_relu", "relu", modes, input_shape_generic=(128, 64)),
            BenchmarkCase("pointnet_linear2", "linear", modes, matrix_m=128, matrix_k=64, matrix_n=128),
            BenchmarkCase("pointnet_global_max", "reduce_max", modes, input_shape_generic=(1, 128, 128), axis=1),
        ]
    if name == "graphsage_mlp_block":
        return [
            BenchmarkCase("sage_neighbor_mean", "reduce_mean", modes, input_shape_generic=(32, 8, 64), axis=1),
            BenchmarkCase("sage_concat", "concat", modes, input_shape_generic=(32, 64), axis=1),
            BenchmarkCase("sage_linear", "linear", modes, matrix_m=32, matrix_k=128, matrix_n=64),
            BenchmarkCase("sage_l2norm", "l2_normalize", modes, input_shape_generic=(32, 64), axis=-1),
        ]
    if name == "anomaly_mlp":
        return [
            BenchmarkCase("anom_dense1", "linear", modes, matrix_m=16, matrix_k=32, matrix_n=64),
            BenchmarkCase("anom_softplus", "softplus", modes, input_shape_generic=(16, 64)),
            BenchmarkCase("anom_dense2", "linear", modes, matrix_m=16, matrix_k=64, matrix_n=16),
            BenchmarkCase("anom_l2norm", "l2_normalize", modes, input_shape_generic=(16, 16), axis=-1),
            BenchmarkCase("anom_score", "reduce_sum", modes, input_shape_generic=(16, 16), axis=-1),
        ]
    raise ValueError(f"Unknown network preset: {name}")

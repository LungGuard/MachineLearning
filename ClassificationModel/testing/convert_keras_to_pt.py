import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn


class CancerClassificationModelPT(nn.Module):
    """
    PyTorch equivalent of the Keras CancerClassificationModel.
    Architecture matched from the saved checkpoint summary:
      Conv(relu)->BN -> Conv(relu)->BN -> MaxPool  (x3 blocks)
      Flatten -> Dropout -> Dense(relu)->BN -> Dense(relu)->BN -> Dense(softmax)
    """

    def __init__(self, num_classes=4):
        super().__init__()

        # Keras BatchNormalization uses eps=1e-3, PyTorch defaults to 1e-5.
        # Must match for correct output.
        eps = 1e-3

        # Block 1: 1 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32, eps=eps),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32, eps=eps),
            nn.MaxPool2d(2),
        )
        # Block 2: 32 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64, eps=eps),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64, eps=eps),
            nn.MaxPool2d(2),
        )
        # Block 3: 64 -> 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128, eps=eps),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128, eps=eps),
            nn.MaxPool2d(2),
        )
        # Classifier: 128*28*28 -> 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 28 * 28, 256), nn.ReLU(), nn.BatchNorm1d(256, eps=eps),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128, eps=eps),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (N, 1, 224, 224) in NCHW format
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return torch.softmax(x, dim=1)


def transfer_weights(keras_model, pytorch_model):
    """Transfer all weights from Keras model to PyTorch model by layer type and order."""
    keras_layers = keras_model.layers

    keras_convs = [l for l in keras_layers if isinstance(l, tf.keras.layers.Conv2D)]
    keras_bns = [l for l in keras_layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    keras_denses = [l for l in keras_layers if isinstance(l, tf.keras.layers.Dense)]

    pt_convs = [m for m in pytorch_model.modules() if isinstance(m, nn.Conv2d)]
    pt_bns = [m for m in pytorch_model.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
    pt_linears = [m for m in pytorch_model.modules() if isinstance(m, nn.Linear)]

    assert len(keras_convs) == len(pt_convs), \
        f"Conv count mismatch: Keras has {len(keras_convs)}, PyTorch has {len(pt_convs)}"
    assert len(keras_bns) == len(pt_bns), \
        f"BN count mismatch: Keras has {len(keras_bns)}, PyTorch has {len(pt_bns)}"
    assert len(keras_denses) == len(pt_linears), \
        f"Dense count mismatch: Keras has {len(keras_denses)}, PyTorch has {len(pt_linears)}"

    # Conv2D: Keras (H, W, C_in, C_out) -> PyTorch (C_out, C_in, H, W)
    for k_layer, p_layer in zip(keras_convs, pt_convs):
        kernel, bias = k_layer.get_weights()
        p_layer.weight.data = torch.from_numpy(kernel.transpose(3, 2, 0, 1).copy())
        p_layer.bias.data = torch.from_numpy(bias.copy())
        print(f"  {k_layer.name:30s} -> {p_layer}")

    # BatchNormalization: gamma, beta, moving_mean, moving_var
    for k_layer, p_layer in zip(keras_bns, pt_bns):
        gamma, beta, moving_mean, moving_var = k_layer.get_weights()
        p_layer.weight.data = torch.from_numpy(gamma.copy())
        p_layer.bias.data = torch.from_numpy(beta.copy())
        p_layer.running_mean.copy_(torch.from_numpy(moving_mean.copy()))
        p_layer.running_var.copy_(torch.from_numpy(moving_var.copy()))
        print(f"  {k_layer.name:30s} -> {p_layer}")

    # Dense: Keras (in, out) -> PyTorch (out, in)
    for k_layer, p_layer in zip(keras_denses, pt_linears):
        kernel, bias = k_layer.get_weights()
        p_layer.weight.data = torch.from_numpy(kernel.T.copy())
        p_layer.bias.data = torch.from_numpy(bias.copy())
        print(f"  {k_layer.name:30s} -> {p_layer}")


def convert_keras_to_pytorch(keras_file_path, output_torch_path):
    print(f"[*] Loading Keras model from: {keras_file_path}")
    keras_model = tf.keras.models.load_model(keras_file_path)
    keras_model.summary()

    # Build PyTorch model and transfer weights
    num_classes = keras_model.layers[-1].get_weights()[1].shape[0]
    pytorch_model = CancerClassificationModelPT(num_classes=num_classes)

    print("\n[*] Transferring weights...")
    transfer_weights(keras_model, pytorch_model)
    pytorch_model.eval()

    # Save
    torch.save(pytorch_model.state_dict(), output_torch_path)
    print(f"\n[V] PyTorch model saved to: {output_torch_path}")

    # --- Verification ---
    print("\n[*] Running verification...")
    dummy_nhwc = np.random.rand(1, 224, 224, 1).astype(np.float32)

    keras_out = keras_model.predict(dummy_nhwc, verbose=0)

    # PyTorch expects NCHW
    dummy_nchw = torch.from_numpy(dummy_nhwc.transpose(0, 3, 1, 2))
    with torch.no_grad():
        pytorch_out = pytorch_model(dummy_nchw).numpy()

    print(f"  Keras output:   {keras_out[0]}")
    print(f"  PyTorch output: {pytorch_out[0]}")
    diff = np.abs(keras_out - pytorch_out).max()
    print(f"  Max difference: {diff:.8f}")

    if diff < 1e-4:
        print("\n[V] Conversion successful! Outputs match.")
    else:
        print(f"\n[!] Warning: outputs differ by {diff}. Check layer mapping.")

    return pytorch_model


# --- Run ---
keras_path = 'ClassificationModel/testing/Checkpoints/best_model.keras'
torch_path = 'best_model.pt'

convert_keras_to_pytorch(keras_path, torch_path)

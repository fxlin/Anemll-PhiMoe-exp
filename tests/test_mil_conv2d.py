#!/usr/bin/env python3

# source ~/workspace-apple-silicon/myenv-python39/bin/activate


import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
import numpy as np
import torch

# Matrix size
N = 4096
bs = 8

# 1. Dummy weights (N_out, N_in)
W_torch = torch.randn(N, N)  # Conv weight shape: (N_out, N_in, 1, 1)
W_conv = W_torch.unsqueeze(-1).unsqueeze(-1).numpy().astype(np.float16)

# 2. Define MIL program using convolution
@mb.program(input_specs=[mb.TensorSpec(shape=(1, N), dtype=types.fp16)], opset_version=ct.target.iOS18)
def conv_matmul_model(X):
    # Reshape (1, N) -> (1, N, 1, 1)
    X_reshaped = mb.reshape(x=X, shape=[1, N, 1, 1])

    # Create conv kernel of shape (N_out, N_in, 1, 1)
    W = mb.const(val=W_conv, name="conv_weight")

    # Perform convolution: output shape will be (1, N_out, 1, 1)
    Y = mb.conv(x=X_reshaped, weight=W, name="output_conv0", strides=[1, 1], pad_type="valid")

    # Reshape back to (1, N)
    Y_out = mb.reshape(x=Y, shape=[1, N], name="output_conv")
    return Y_out

# 3. Convert to CoreML
mlmodel_conv = ct.convert(
    conv_matmul_model,
    compute_precision=ct.precision.FLOAT16,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)

# 4. Save
mlmodel_conv.save("/tmp/SimpleMatmulConv.mlpackage")

# 5. Test
X_sample = np.random.rand(1, N).astype(np.float16)
result = mlmodel_conv.predict({"X": X_sample})
print("Output shape (conv):", result["output_conv"].shape)

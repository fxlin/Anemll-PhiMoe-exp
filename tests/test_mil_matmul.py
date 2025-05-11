#!/usr/bin/env python3
'''
source ~/workspace-apple-silicon/myenv-python39/bin/activate
python tests/test_coreml_matmul.py

also cf
test code
https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/converters/mil/mil/ops/tests/iOS18/test_tensor_transformation.py

results: seems always map to CPU (not ANE)?? contrast to conv2d which maps to ANE well 
'''

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
import numpy as np
import torch

# 1. Create dummy PyTorch weights
N=4096
bs=8
W_torch = torch.randn(N,N)  # Shape (1024, 1024)

# 2. Convert to numpy array
# W_np = W_torch.numpy().astype(np.float32)
W_np = W_torch.numpy().astype(np.float16)

# 3. Define MIL program: input X (bs,1024) -> output = X @ W
@mb.program(input_specs=[mb.TensorSpec(shape=(bs, N), dtype=types.fp16)], opset_version=ct.target.iOS18)
def simple_matmul_model(X):
    W = mb.const(val=W_np, name="weight_matrix")  # Frozen weight
    Y = mb.matmul(x=X, y=W, name="output_matmul")
    # Y = mb.add(x=X, y=X, name="output_matmul")
    return Y

print(simple_matmul_model)
'''
main[CoreML8](%X: (1, 512, fp16)(Tensor)) {
  block0() {
    %output_matmul: (1, 1, 512, fp16)(Tensor) = matmul(x=%X, y=%weight_matrix, transpose_x=False, transpose_y=False, name="output_matmul")
  } -> (%output_matmul)
}
'''

# 4. Convert to CoreML
mlmodel = ct.convert(
    simple_matmul_model,
    compute_precision=ct.precision.FLOAT16,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)

# 6. Save - unquantized
mlmodel.save("/tmp/SimpleMatmul.mlpackage")

#  5. --- quant ---- 
# https://apple.github.io/coremltools/source/coremltools.optimize.coreml.utilities.html#coremltools.optimize.coreml.OptimizationConfig
# https://apple.github.io/coremltools/source/coremltools.optimize.coreml.palettization.html#coremltools.optimize.coreml.OpPalettizerConfig
import coremltools.optimize as cto
LUT_BITS = 8
GROUP_SIZE = 32  # default 

try:
    # Set up quantization config     XXX can try other quant config... (not just LUT
    config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=LUT_BITS,
            granularity="per_grouped_channel",
            group_size=GROUP_SIZE,
            num_kmeans_workers=1
        ),
    )
    
    # Apply quantization in a try-except block      fxl: direclty call coremltools' LUT quant schemes
    #     fxl: https://apple.github.io/coremltools/docs-guides/source/opt-overview.html 
    try:
        mlmodel = cto.coreml.palettize_weights(mlmodel, config)
        print("LUT quantization completed")
    except ValueError as e:
        if "Pool not running" in str(e):
            print("Warning: Multiprocessing pool error, retrying with single process...")
            # Retry with single process
            config.global_config.num_kmeans_workers = 1
            mlmodel = cto.coreml.palettize_weights(mlmodel, config)
            print("LUT quantization completed (single process)")
        else:
            raise
except Exception as e:
    print(f"Warning: LUT quantization failed: {str(e)}")
    print("Continuing with unquantized model...")
    
# 6. Save - quant 
mlmodel.save("/tmp/SimpleMatmul-LUT%d.mlpackage" %LUT_BITS)

# 7. Test
# X_sample = np.random.rand(bs, 1024).astype(np.float16)
X_sample = np.random.rand(bs, N)
result = mlmodel.predict({"X": X_sample})
print("Output shape:", result["output_matmul"].shape)

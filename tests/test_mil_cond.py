#!/usr/bin/env python3

'''
test coreml MIL programs
cf: https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html
https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/ops/defs/iOS15/linear.py

'''
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch

N=4096
bs=8
# 1. Create dummy PyTorch weights (replace with your actual weights)
A_torch = torch.randn(N, 2)  # Shape (1024, 2)
B_torch = torch.randn(bs,N,N)  # Shape (1024, 1024)
C_torch = torch.randn(bs,N,N)  # Shape (1024, 1024)

# 2. Convert to numpy arrays
A_np = A_torch.numpy().astype(np.float16)
B_np = B_torch.numpy().astype(np.float16)
C_np = C_torch.numpy().astype(np.float16)

# 3. Define MIL program with frozen weights
@mb.program(input_specs=[mb.TensorSpec(shape=(1, N), dtype=types.fp16),], opset_version=ct.target.iOS18)
# @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1024), dtype=types.fp32),])  # Only X is input
def conditional_matmul_model(X):
    # Load weights as constants
    A = mb.const(val=A_np, name="A_weight")
    B = mb.const(val=B_np, name="B_weight")
    C = mb.const(val=C_np, name="C_weight")
    
    # Step 1: X @ A -> (1,2)
    xa = mb.matmul(x=X, y=A, name="xa_matmul")
    
    # Step 2: Compare elements in xa[0]
    is_first_larger = mb.greater(
        x=mb.slice_by_index(x=xa, begin=[0,0], end=[1,1], squeeze_mask=[True,True]),  # xa[0,0]
        y=mb.slice_by_index(x=xa, begin=[0,1], end=[1,2], squeeze_mask=[True,True]),  # xa[0,1]
        name="compare_elements"
    )
    
    # Step 3: Conditional multiplication
    def true_fn():
        return mb.matmul(x=X, y=B, name="xb_matmul")
    
    def false_fn():
        return mb.matmul(x=X, y=C, name="xc_matmul")
    
    output = mb.cond(
        pred=is_first_larger,
        _true_fn=true_fn,
        _false_fn=false_fn,
        name="conditional_output"
    )
    
    return output

print(conditional_matmul_model)
'''
main[CoreML8](%X: (1, 1024, fp16)(Tensor)) {
  block0() {
    %xa_matmul: (1, 2, fp16)(Tensor) = matmul(x=%X, y=%A_weight, transpose_x=False, transpose_y=False, name="xa_matmul")
    %slice_by_index_0: (fp16)(Scalar) = slice_by_index(x=%xa_matmul, begin=[0, 0], end=[1, 1], squeeze_mask=[True, True], name="slice_by_index_0")
    %slice_by_index_1: (fp16)(Scalar) = slice_by_index(x=%xa_matmul, begin=[0, 1], end=[1, 2], squeeze_mask=[True, True], name="slice_by_index_1")
    %compare_elements: (bool)(Scalar) = greater(x=%slice_by_index_0, y=%slice_by_index_1, name="compare_elements")
    %conditional_output: (1, 1024, fp16)(Tensor) = cond(pred=%compare_elements, name="conditional_output")
      conditional_output_true() {
        %xb_matmul: (1, 1024, fp16)(Tensor) = matmul(x=%X, y=%B_weight, transpose_x=False, transpose_y=False, name="xb_matmul")
      } -> (%xb_matmul)
      conditional_output_false() {
        %xc_matmul: (1, 1024, fp16)(Tensor) = matmul(x=%X, y=%C_weight, transpose_x=False, transpose_y=False, name="xc_matmul")
      } -> (%xc_matmul)
  } -> (%conditional_output)
}
'''

# 4. Convert to CoreML model
mlmodel = ct.convert(
    conditional_matmul_model,
    compute_precision=ct.precision.FLOAT16,
    convert_to="mlprogram",  # or "mlprogram" for newer iOS versions
    minimum_deployment_target=ct.target.iOS18,    # not helping fp16 problem?
    compute_units=ct.ComputeUnit.CPU_AND_NE  # Use CPU/GPU/ANE
)

# 5. Save and test
mlmodel.save("/tmp/ConditionalMatmul.mlpackage")

# Test prediction
X_sample = np.random.rand(1, N).astype(np.float16)
result = mlmodel.predict({"X": X_sample})
print("Output shape:", result["conditional_output"].shape)  # Should be (1, 1024)
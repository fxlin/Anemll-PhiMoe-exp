#!/usr/bin/env python3

'''
test coreml MIL programs
cf: https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html
https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/ops/defs/iOS15/linear.py


update: test results (vs python execution, at the end of this file) sugges that the cond branch is executed, it's just coreml's ComputePlan does not include
 that estimate. (that is, ComputePlan does not look into a "cond" op and evaluate the branch function

 so ComputePlan is mostly static analysis --- won’t profile both branches exhaustively

but even we look at Instruments, it's 100% on cpu... may be "cond" (cpu only) is too close to the "conv" after it, so 
coreml stays on cpu?

'''
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch

N=4096
bs=32
# 1. Create dummy PyTorch weights (replace with your actual weights)
A_torch = torch.randn(N, 2)  # Shape (1024, 2)
B_torch = torch.randn(N,N)  # Shape (1024, 1024)
C_torch = torch.randn(N,N)  # Shape (1024, 1024)

# for conv2d
B_torch = B_torch.unsqueeze(-1).unsqueeze(-1)  # Shape (1024, 1024, 1, 1)
C_torch = C_torch.unsqueeze(-1).unsqueeze(-1)  # Shape (1024, 1024, 1, 1)

# 2. Convert to numpy arrays
A_np = A_torch.numpy().astype(np.float16)
B_np = B_torch.numpy().astype(np.float16)
C_np = C_torch.numpy().astype(np.float16)

# 3. Define MIL program with frozen weights
@mb.program(input_specs=[mb.TensorSpec(shape=(bs, N), dtype=types.fp16),], opset_version=ct.target.iOS18)
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
    
    # Step 3: Conditional multiplication   (XXX why mb.conv not executed at runtime? XXX maybe compare to a traced program
    def true_fn():
        # return mb.matmul(x=X, y=B, name="xb_matmul")
        X_reshaped = mb.reshape(x=X, shape=[bs, N, 1, 1])
        Y =  mb.conv(x=X_reshaped, weight=B, name="xb_matmul", strides=[1, 1], pad_type="valid")
        return Y
    
    def false_fn():
        # return mb.matmul(x=X, y=C, name="xc_matmul")
        X_reshaped = mb.reshape(x=X, shape=[bs, N, 1, 1])
        Y = mb.conv(x=X_reshaped, weight=C, name="xc_matmul", strides=[1, 1], pad_type="valid")
        return Y
    
    output = mb.cond(
        pred=is_first_larger,
        _true_fn=true_fn,
        _false_fn=false_fn,
        name="conditional_output_pre"
    )
    # output = mb.reshape(x=output, shape=[bs, N], name="conditional_output")
    
    return output

# conditional_matmul_model -- type: <class 'coremltools.converters.mil.mil.program.Program'>
print(conditional_matmul_model)

'''
main[CoreML8](%X: (8, 4096, fp16)(Tensor)) {
  block0() {
    %xa_matmul: (8, 2, fp16)(Tensor) = matmul(x=%X, y=%A_weight, transpose_x=False, transpose_y=False, name="xa_matmul")
    %slice_by_index_0: (fp16)(Scalar) = slice_by_index(x=%xa_matmul, begin=[0, 0], end=[1, 1], squeeze_mask=[True, True], name="slice_by_index_0")
    %slice_by_index_1: (fp16)(Scalar) = slice_by_index(x=%xa_matmul, begin=[0, 1], end=[1, 2], squeeze_mask=[True, True], name="slice_by_index_1")
    %compare_elements: (bool)(Scalar) = greater(x=%slice_by_index_0, y=%slice_by_index_1, name="compare_elements")
    %conditional_output_pre: (8, 4096, 1, 1, fp16)(Tensor) = cond(pred=%compare_elements, name="conditional_output_pre")
      conditional_output_pre_true() {
        %reshape_0: (8, 4096, 1, 1, fp16)(Tensor) = reshape(x=%X, shape=[8, 4096, 1, 1], name="reshape_0")
        %xb_matmul: (8, 4096, 1, 1, fp16)(Tensor) = conv(x=%reshape_0, weight=%B_weight, strides=[1, 1], pad_type="valid", pad=[0, 0, 0, 0], dilations=[1, 1], groups=1, name="xb_matmul")
      } -> (%xb_matmul)
      conditional_output_pre_false() {
        %reshape_1: (8, 4096, 1, 1, fp16)(Tensor) = reshape(x=%X, shape=[8, 4096, 1, 1], name="reshape_1")
        %xc_matmul: (8, 4096, 1, 1, fp16)(Tensor) = conv(x=%reshape_1, weight=%C_weight, strides=[1, 1], pad_type="valid", pad=[0, 0, 0, 0], dilations=[1, 1], groups=1, name="xc_matmul")
      } -> (%xc_matmul)
    %conditional_output: (8, 4096, fp16)(Tensor) = reshape(x=%conditional_output_pre, shape=[8, 4096], name="conditional_output")
  } -> (%conditional_output)
}
'''

###################################################################
# another attmpt to get true/fals function to execute....
# still no luck, same as above ....  xb_matmul and xc_matmul not executed at run time....

AA_torch = torch.randn(N, 2)  # Shape (1024, 2)
BB_torch = torch.randn(N,N)  # Shape (1024, 1024)
CC_torch = torch.randn(N,N)  # Shape (1024, 1024)

# 2. Convert to numpy arrays
AA_np = AA_torch.numpy().astype(np.float16)
BB_np = BB_torch.numpy().astype(np.float16)
CC_np = CC_torch.numpy().astype(np.float16)

@mb.program(input_specs=[mb.TensorSpec(shape=(bs, N), dtype=types.fp16)], opset_version=ct.target.iOS18)
def conditional_matmul_model1(X):
    A = mb.const(val=AA_np, name="A_weight")
    B = mb.const(val=BB_np, name="B_weight")
    C = mb.const(val=CC_np, name="C_weight")
    
    xa = mb.matmul(x=X, y=A, name="xa_matmul")
    
    # Compare against a constant to ensure dynamic behavior
    is_first_larger = mb.greater(
        x=mb.slice_by_index(x=xa, begin=[0,0], end=[1,1], squeeze_mask=[True,True]),
        # x=mb.const(val=np.array(2.0, dtype=np.float16)),   # debugging
        y=mb.const(val=np.array(0.0, dtype=np.float16)),
        name="compare_elements"
    )
    
    def true_fn():
        return mb.matmul(x=X, y=B, name="xb_matmul")
    
    def false_fn():
        return mb.matmul(x=X, y=C, name="xc_matmul")
    
    return mb.cond(
        pred=is_first_larger,
        _true_fn=true_fn,
        _false_fn=false_fn,
        name="conditional_output"
    )

# print(conditional_matmul_model1)   # IR before opt
'''
main[CoreML8](%X: (32, 4096, fp16)(Tensor)) {
  block1() {
    %xa_matmul: (32, 2, fp16)(Tensor) = matmul(x=%X, y=%A_weight, transpose_x=False, transpose_y=False, name="xa_matmul")
    %slice_by_index_2: (fp16)(Scalar) = slice_by_index(x=%xa_matmul, begin=[0, 0], end=[1, 1], squeeze_mask=[True, True], name="slice_by_index_2")
    %compare_elements: (bool)(Tensor) = greater(x=%slice_by_index_2, y=0.0, name="compare_elements")
    %conditional_output: (32, 4096, fp16)(Tensor) = cond(pred=%compare_elements, name="conditional_output")
      conditional_output_true() {
        %xb_matmul: (32, 4096, fp16)(Tensor) = matmul(x=%X, y=%B_weight, transpose_x=False, transpose_y=False, name="xb_matmul")
      } -> (%xb_matmul)
      conditional_output_false() {
        %xc_matmul: (32, 4096, fp16)(Tensor) = matmul(x=%X, y=%C_weight, transpose_x=False, transpose_y=False, name="xc_matmul")
      } -> (%xc_matmul)
  } -> (%conditional_output)
}
'''

# 4. Convert to CoreML model
mlmodel = ct.convert(
    conditional_matmul_model,
    # conditional_matmul_model1,
    compute_precision=ct.precision.FLOAT16,
    convert_to="mlprogram",  # or "mlprogram" for newer iOS versions
    minimum_deployment_target=ct.target.iOS18,    # not helping fp16 problem?
    compute_units=ct.ComputeUnit.CPU_AND_NE  # Use CPU/GPU/ANE
)
print(mlmodel._mil_program)   # IR after optimization

# Test prediction -- works 
'''
X_input = np.random.rand(bs, N).astype(np.float16)
outputs_ml = mlmodel.predict({"X": X_input})
print("Output shape:", outputs_ml["conditional_output"].shape)  # Should be (1, 1024)

# Compute manually for comparison
# Step 1: xa = X @ A
A_tensor = torch.from_numpy(AA_np).half()
X_tensor = torch.from_numpy(X_input).half()
xa_tensor = X_tensor @ A_tensor

# Step 2: check condition
is_first_larger_py = xa_tensor[0, 0].item() > 0.0

# Step 3: compute expected result
if is_first_larger_py:
    expected_output = X_tensor @ torch.from_numpy(BB_np).half()
else:
    expected_output = X_tensor @ torch.from_numpy(CC_np).half()

# Compare outputs
output_ml = torch.from_numpy(outputs_ml["conditional_output"]).half()
diff = (output_ml - expected_output).abs().max().item()

print(f"Condition evaluated to {is_first_larger_py}")
print(f"Max difference between CoreML and expected output: {diff:.6f}")

breakpoint()
'''

# 5. Save and test
mlmodel.save("/tmp/ConditionalMatmul.mlpackage")
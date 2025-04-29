#!/usr/bin/env python3

'''
Test CoreML MIL programs with loop operation

not working TBD
'''

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch

N = 4096
bs = 8

# 1. Create dummy PyTorch weights
A_torch = torch.randn(N, 2)  # Shape (N, 2)
B_torch = torch.randn(bs, N, N)  # Shape (bs, N, N)
C_torch = torch.randn(bs, N, N)  # Shape (bs, N, N)

# 2. Convert to numpy arrays
A_np = A_torch.numpy().astype(np.float16)
B_np = B_torch.numpy().astype(np.float16)
C_np = C_torch.numpy().astype(np.float16)

# 3. Define MIL program with loop operation
@mb.program(input_specs=[mb.TensorSpec(shape=(1, N), dtype=types.fp16)], opset_version=ct.target.iOS18)
def loop_matmul_model(X):
    # Load weights as constants
    A = mb.const(val=A_np, name="A_weight")
    B = mb.const(val=B_np, name="B_weight")
    C = mb.const(val=C_np, name="C_weight")
    
    # Initial counter
    counter = mb.const(val=0, name="initial_counter")
    max_iterations = mb.const(val=3, name="max_iterations")  # Loop 3 times
    
    # Loop state: (counter, accumulated_output)
    initial_state = (counter, X)
    
    def cond(counter, accumulated_output):
        return mb.less(x=counter, y=max_iterations, name="loop_cond")
    
    def body(counter, accumulated_output):
        # Step 1: X @ A -> (1,2)
        xa = mb.matmul(x=accumulated_output, y=A, name="xa_matmul")
        
        # Step 2: Compare elements in xa[0]
        is_first_larger = mb.greater(
            x=mb.slice_by_index(x=xa, begin=[0,0], end=[1,1], squeeze_mask=[True,True]),
            y=mb.slice_by_index(x=xa, begin=[0,1], end=[1,2], squeeze_mask=[True,True]),
            name="compare_elements"
        )
        
        # Step 3: Conditional multiplication
        def true_fn():
            return mb.matmul(x=accumulated_output, y=B[0], name="xb_matmul")  # Use first batch
        
        def false_fn():
            return mb.matmul(x=accumulated_output, y=C[0], name="xc_matmul")  # Use first batch
        
        new_output = mb.cond(
            pred=is_first_larger,
            _true_fn=true_fn,
            _false_fn=false_fn,
            name="conditional_output"
        )
        
        # Increment counter
        new_counter = mb.add(x=counter, y=mb.const(val=1), name="increment_counter")
        
        return (new_counter, new_output)
    
    # Execute the while loop
    final_counter, final_output = mb.while_loop(
        _cond=cond,
        _body=body,
        loop_vars=initial_state,
        name="while_loop"
    )
    
    return final_output

print("MIL Program with Loop:")
print(loop_matmul_model)

# 4. Convert to CoreML model
mlmodel = ct.convert(
    loop_matmul_model,
    compute_precision=ct.precision.FLOAT16,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE
)

# 5. Save and test
mlmodel.save("/tmp/LoopMatmul.mlpackage")

# Test prediction
X_sample = np.random.rand(1, N).astype(np.float16)
result = mlmodel.predict({"X": X_sample})
print("\nTest Prediction Results:")
print("Output shape:", result["while_loop_1_while_loop_exit"] if "while_loop_1_while_loop_exit" in result else result["while_loop_0_while_loop_exit"].shape)
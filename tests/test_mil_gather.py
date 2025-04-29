import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
import numpy as np
import torch

# works ok 

# 1. Create dummy PyTorch weights
N = 4096
bs = 8
W_torch = torch.randn(bs, N, N)  # Shape (8, 4096, 4096)

# 2. Convert to numpy array, float16
W_np = W_torch.numpy().astype(np.float16)

# 3. Now define the MIL program
@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(bs, N), dtype=types.fp16),        # x: (bs, 4096)
        mb.TensorSpec(shape=(bs, 1), dtype=types.int32),        # indices: (bs, 1)
    ],
    opset_version=ct.target.iOS18
)
def prog(x, indices):
    # Load weights as constant
    weights = mb.const(val=W_np)  # Shape (8, 4096, 4096)

    # ====== Clip indices into range 0~7 using mod operation ======
    # indices_wrapped = mb.elementwise_binary(
    #     x=indices,
    #     y=mb.const(val=8, dtype=types.int32),  # modulus base
    #     op="mod"
    # )
    # indices_wrapped = indices
    indices_wrapped = mb.mod(x=indices, y=bs)

    # Now gather weights
    gathered_weights = mb.gather(
        x=weights,
        indices=indices_wrapped,
        axis=0,
    )  # (bs, 1, 4096, 4096)

    # Expand x to match
    x_expanded = mb.expand_dims(x=x, axes=[1])  # (bs, 1, 4096)

    # Batch matmul
    out = mb.matmul(x=x_expanded, y=gathered_weights)  # (bs, 1, 1, 4096)

    return out

print(prog)


# 4. Convert to CoreML
mlmodel = ct.convert(
    prog,
    compute_precision=ct.precision.FLOAT16,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)

mlmodel.save("/tmp/gathermm.mlpackage")

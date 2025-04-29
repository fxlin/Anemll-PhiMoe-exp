'''
trace torch program. 

many ops cannot map to ANE (only GPU)
likely b/c torch code coiuld have been written better... also weight file too large

main[CoreML8](%x: (8, 4096, fp16)(Tensor),
              %indices: (8, 1, int32)(Tensor)) {
  block0() {
    %indices_wrapped_div: (8, 1, int32)(Tensor) = floor_div(x=%indices, y=8, name="indices_wrapped_div")
    %indices_wrapped_div_scaled: (8, 1, int32)(Tensor) = mul(x=%indices_wrapped_div, y=8, name="indices_wrapped_div_scaled")
    %indices_wrapped: (8, 1, int32)(Tensor) = sub(x=%indices, y=%indices_wrapped_div_scaled, name="indices_wrapped")
    %var_15: (8, 1, 1, int32)(Tensor) = reshape(x=%indices_wrapped, shape=[np.int32(-1), np.int32(1), np.int32(1)], name="op_15")
    %indices_expanded: (8, 1, 4096, int32)(Tensor) = tile(x=%var_15, reps=[1, np.int32(1), np.int32(4096)], name="indices_expanded")
    %var_35: (8, 1, 4096, 1, int32)(Tensor) = expand_dims(x=%indices_expanded, axes=[np.int32(-1)], name="op_35")
    %var_42: (8, 1, 4096, 4096, int32)(Tensor) = tile(x=%var_35, reps=[1, 1, 1, np.int32(4096)], name="op_42")
    %var_42_to_int16: (8, 1, 4096, 4096, int16)(Tensor) = cast(x=%var_42, dtype="int16", name="cast_4")
    %gathered_weights_cast_uint16: (8, 1, 4096, 4096, fp16)(Tensor) = gather_along_axis(x=%var_33, indices=%var_42_to_int16, axis=1, validate_indices=False, name="gathered_weights_cast_uint16")
    %x_expanded_cast_fp16: (8, 1, 4096, fp16)(Tensor) = expand_dims(x=%x, axes=[np.int32(1)], name="x_expanded_cast_fp16")
    %var_50: (8, 8, 1, 4096, fp16)(Tensor) = matmul(x=%x_expanded_cast_fp16, y=%gathered_weights_cast_uint16, transpose_x=False, transpose_y=False, name="op_50_cast_fp16")
  } -> (%var_50)
}

'''
import torch
import torch.nn as nn
import numpy as np
import coremltools as ct

class GatherMatMul(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', torch.from_numpy(weights))  # Shape (8, 4096, 4096)
        
    def forward(self, x, indices):
        # Convert indices to int64 and clip using mod
        # indices_wrapped = torch.fmod(indices.long(), 8)
        indices_wrapped = indices.long() % 8    # result in very large weights (redundant tracing?
        
        # Gather weights - need to prepare indices correctly
        # For torch.gather, we need indices with same number of dims as input
        indices_expanded = indices_wrapped.view(-1, 1, 1).expand(-1, 1, 4096)
        
        # Gather along dim 0 (the batch dimension we'll add)
        gathered_weights = torch.gather(
            self.weights.unsqueeze(0).expand(x.shape[0], -1, -1, -1),  # (bs, 8, 4096, 4096)
            1,
            indices_expanded.unsqueeze(-1).expand(-1, -1, -1, 4096)    # (bs, 1, 4096, 4096)
        )  # result shape (bs, 1, 4096, 4096)
        
        # Expand x to match - shape (bs, 1, 4096)
        x_expanded = x.unsqueeze(1)
        
        # Batch matmul - result shape (bs, 1, 4096)
        out = torch.matmul(x_expanded, gathered_weights.squeeze(2))
        
        return out

# 1. Create dummy PyTorch weights
N = 4096
bs = 8
W_torch = torch.randn(bs, N, N)  # Shape (8, 4096, 4096)
W_np = W_torch.numpy().astype(np.float16)

# 2. Create model and sample inputs
model = GatherMatMul(W_np)
model.eval()

# Sample inputs - note indices must be int64 for CoreML compatibility
x_sample = torch.randn(bs, N).to(torch.float16)
indices_sample = torch.randint(0, 16, (bs, 1)).to(torch.int64)  # Changed to int64

# 3. Trace the model
traced_model = torch.jit.trace(model, (x_sample, indices_sample))

# 4. Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="x", shape=(bs, N), dtype=np.float16),
        ct.TensorType(name="indices", shape=(bs, 1), dtype=np.int64),  # Match torch.long
    ],
    compute_precision=ct.precision.FLOAT16,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)

# Print MIL program
print("Generated MIL program:")
print(mlmodel._mil_program)

print("MIL program specs:")
print(mlmodel.get_spec())

# mlmodel.save("/tmp/gathermm_torch.mlpackage")
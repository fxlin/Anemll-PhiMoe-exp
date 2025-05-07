import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
import numpy as np

# Define external weights with grouped expert shape
NUM_EXPERTS = 16
HIDDEN=512
INTERMEDIATE=1024
# HIDDEN=4096
# INTERMEDIATE=6400


# Conv weight shape: (N_out, N_in, 1, 1)
EXPERT_UP = np.zeros((NUM_EXPERTS, INTERMEDIATE, HIDDEN, 1, 1), dtype=np.float16)  
EXPERT_DOWN = np.zeros((NUM_EXPERTS, HIDDEN, INTERMEDIATE, 1, 1), dtype=np.float16) 
GATE_WEIGHT = np.zeros((NUM_EXPERTS, HIDDEN, 1, 1), dtype=np.float16)

'''
# not working, see comments inline
@mb.program(input_specs=[mb.TensorSpec((1, 1, HIDDEN), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog(input):
    x = mb.transpose(x=input, perm=[0, 2, 1], name="transpose_0")
    x = mb.expand_dims(x=x, axes=[-1], name="input_1_cast_fp16")  # (1, HIDDEN, 1, 1)

    gate_weight = mb.const(val=GATE_WEIGHT, name="gate_weight_promoted_to_fp16")
    gate_out = mb.conv(
        x=x,
        weight=gate_weight,
        strides=[1, 1],
        pad_type="valid",
        name="op_36_cast_fp16"
    )
    gate_out = mb.squeeze(x=gate_out, axes=[-1], name="op_38_cast_fp16")
    router_logits = mb.squeeze(x=gate_out, axes=[-1], name="router_logits_cast_fp16")
    routing_weights = mb.softmax(x=router_logits, axis=-1, name="routing_weights_cast_fp16")

    # XXX topk, ANE runtime error: "RCAS ins't supported in this architecutre"  on M2  -- fallback to CPU
    #   RCAS -- ReduceCompareAndSelect? maybe try on newer hardware like M4?
    topk_vals, topk_indices = mb.topk(
        x=routing_weights,
        k=2,
        axis=-1,
        return_indices=True,
        # output_indices_dtype="uint16",  # int32 unsupported on ANE 
        name="topk_weights_1_cast_fp16"
    )

    # === Manually implement topk ===
    # gave up, no clean solution; it seems topk cannot be cleanly implemented on ANE (tyipe mismatch, no support for reduce_argmax int16, etc, etc
    max_val = mb.reduce_max(x=routing_weights, axes=[-1], keep_dims=True, name="top1_val")
    max_idx = mb.reduce_argmax(x=routing_weights, axis=-1, output_dtype="uint16", name="top1_idx")

    range_tensor = mb.range_1d(start=0, end=NUM_EXPERTS, step=1, name="expert_range")
    range_tensor = mb.reshape(x=range_tensor, shape=(1, NUM_EXPERTS))
    max_idx_u16 = mb.cast(x=max_idx, dtype=types.uint16)
    mask = mb.not_equal(x=range_tensor, y=max_idx_u16)
    mask = mb.cast(x=mask, dtype=types.fp16)
    masked_values = mb.mul(x=routing_weights, y=mask, name="masked_weights")

    second_val = mb.reduce_max(x=masked_values, axes=[-1], keep_dims=True, name="top2_val")
    second_idx = mb.reduce_argmax(x=masked_values, axis=-1, output_dtype="uint16", name="top2_idx")

    topk_vals = mb.concat(values=[max_val, second_val], axis=-1, name="topk_vals")
    topk_indices = mb.stack(values=[max_idx, second_idx], axis=-1, name="topk_indices")
    topk_indices = mb.expand_dims(x=topk_indices, axes=[0])
    # ================================

    norm = mb.reduce_sum(x=topk_vals, axes=[-1], keep_dims=True, name="op_54_cast_fp16")
    topk_weights = mb.real_div(x=topk_vals, y=norm, name="topk_weights_cast_fp16")

    # w1 and w2 (scalar multipliers)
    w1 = mb.slice_by_index(
        x=topk_weights,
        begin=[0, 0],
        end=[1, 1],
        end_mask=[False, False],
        squeeze_mask=[True, True],
        name="op_92_cast_fp16"
    )
    w2 = mb.slice_by_index(
        x=topk_weights,
        begin=[0, 1],
        end=[1, 2],
        end_mask=[False, False],
        squeeze_mask=[True, True],
        name="op_123_cast_fp16"
    )

    # Fetch expert indices
    i1 = mb.slice_by_index(x=topk_indices, begin=[0, 0], end=[1, 1], squeeze_mask=[True, True])
    i2 = mb.slice_by_index(x=topk_indices, begin=[0, 1], end=[1, 2], squeeze_mask=[True, True])

    # Expert weights as const tensors
    expert_up = mb.const(val=EXPERT_UP, name="expert_up_weight")  # shape [nexp, in, out, 1, 1]
    expert_down = mb.const(val=EXPERT_DOWN, name="expert_down_weight")

    # XXX dynamically slice expert weights. 
    #  ---> not working, b/c it seems that either both tensors and indces are static, or both are dynamic (???)_
    # Slice expert1 weights
    offset = mb.const(val=np.array(1, dtype=np.int32))
    end_var = mb.add(x=i1, y=offset)
    # end_index = i1 + 1
    up1 = mb.slice_by_index(
        x=expert_up,
        begin=[i1],
        # end=[mb.add(x=i1, y=mb.const(val=np.array(1, dtype=np.uint16)))],
        # end=[mb.add(x=i1, y=mb.const(val=1))],
        end=[end_var],      # MIL is trying to treat end_var as a constant, if "expert_up" is static shape....
        # end=[end_index],
        squeeze_mask=[True]
    )
    down1 = mb.slice_by_index(
        x=expert_down,
        begin=[i1],
        # end=[mb.add(x=i1, y=mb.const(val=np.array(1, dtype=np.uint16)))],
        end=[mb.add(x=i1, y=mb.const(val=np.array(1, dtype=np.int32)))],
        squeeze_mask=[True]
    )
    x1 = mb.conv(x=x, weight=up1, strides=[1, 1], pad_type="valid", name="expert1_up")
    x1 = mb.silu(x=x1, name="expert1_act")
    x1 = mb.conv(x=x1, weight=down1, strides=[1, 1], pad_type="valid", name="expert1_down")
    x1 = mb.squeeze(x=mb.squeeze(x=x1, axes=[-1]), axes=[-1], name="expert1_out")
    out1 = mb.mul(x=w1, y=x1, name="out1_weighted")

    # Slice expert2 weights
    up2 = mb.slice_by_index(
        x=expert_up,
        begin=[i2],
        # end=[mb.add(x=i2, y=mb.const(val=np.array(1, dtype=np.uint16)))],
        end=[mb.add(x=i2, y=mb.const(val=np.array(1, dtype=np.int32)))],
        squeeze_mask=[True]
    )
    down2 = mb.slice_by_index(
        x=expert_down,
        begin=[i2],
        # end=[mb.add(x=i2, y=mb.const(val=np.array(1, dtype=np.uint16))
        end=[mb.add(x=i2, y=mb.const(val=np.array(1, dtype=np.int32))
)],
        squeeze_mask=[True]
    )
    x2 = mb.conv(x=x, weight=up2, strides=[1, 1], pad_type="valid", name="expert2_up")
    x2 = mb.silu(x=x2, name="expert2_act")
    x2 = mb.conv(x=x2, weight=down2, strides=[1, 1], pad_type="valid", name="expert2_down")
    x2 = mb.squeeze(x=mb.squeeze(x=x2, axes=[-1]), axes=[-1], name="expert2_out")
    out2 = mb.mul(x=w2, y=x2, name="out2_weighted")

    combined = mb.add(x=out1, y=out2, name="combined_output")
    output = mb.reshape(x=combined, shape=[1, 1, HIDDEN], name="final_output")

    return output
'''
# print(prog)

# 1 token sent to all experts. use routing weights to sum up 
@mb.program(input_specs=[mb.TensorSpec((1, 1, HIDDEN), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog1(input):
    # Reshape [1, 1, HIDDEN] -> [1, HIDDEN, 1, 1]
    x = mb.transpose(x=input, perm=[0, 2, 1])              # [1, H, 1]
    x = mb.expand_dims(x=x, axes=[-1])                     # [1, H, 1, 1]

    # Router logits: [1, NUM_EXPERTS, 1, 1]
    gate_weight_var = mb.const(val=GATE_WEIGHT)
    logits = mb.conv(x=x, weight=gate_weight_var, strides=[1, 1], pad_type="valid")

    # Squeeze to [1, NUM_EXPERTS]
    logits = mb.squeeze(x=logits, axes=[-1, -2])
    routing_weights = mb.softmax(x=logits, axis=-1)        # [1, NUM_EXPERTS]

    # Normalize top-K (emulate top-k mask — here simplified: identity mask for testing)
    # Step 1: zero out all but top-k — for simplicity, just keep all and assume softmax ~ topk already
    # You could add masking logic here if needed
    weights_full = routing_weights                         # [1, NUM_EXPERTS]

    # Expand weights across channels [1, HIDDEN * NUM_EXPERTS]
    weights_mask = mb.tile(
        x=weights_full,
        reps=[1, HIDDEN]
    )  # shape: [1, HIDDEN * NUM_EXPERTS]

    # (1) Expand input: [1, HIDDEN, 1, 1] → repeat over experts
    x_repeated = mb.tile(x=x, reps=[1, NUM_EXPERTS, 1, 1])  # [1, HIDDEN * N, 1, 1]

    # (2) Up projection
    up_proj_weight = mb.const(val=EXPERT_UP.reshape(NUM_EXPERTS * INTERMEDIATE, HIDDEN, 1, 1))
    # Apply up projection with grouped conv
    up_proj = mb.conv(
        x=x_repeated,
        weight=up_proj_weight,
        pad_type="valid",
        groups=NUM_EXPERTS
    )

    act = mb.silu(x=up_proj)

    # (3) Down projection
    down_proj_weight = mb.const(val=EXPERT_DOWN.reshape(NUM_EXPERTS * HIDDEN, INTERMEDIATE, 1, 1))
    down_proj = mb.conv(
        x=act,
        weight=down_proj_weight,
        pad_type="valid",
        groups=NUM_EXPERTS
    )  # [1, HIDDEN * N, 1, 1]

    hidden_flat = mb.squeeze(x=down_proj, axes=[-1, -2])  # → [1, HIDDEN * N]

    # (4) Multiply by routing weights
    hidden_weighted = mb.mul(
        x=hidden_flat,
        y=weights_mask
    )

    # (5) Reshape to [1, NUM_EXPERTS, HIDDEN]
    hidden = mb.reshape(x=hidden_weighted, shape=[1, NUM_EXPERTS, HIDDEN])

    # (6) Reduce across expert axis
    combined = mb.reduce_sum(x=hidden, axes=[1])

    # (7) Reshape back to [1, 1, HIDDEN]
    output = mb.reshape(x=combined, shape=[1, 1, HIDDEN])

    return output


mlmodel = ct.convert(
    # prog,
    prog1,
    compute_precision=ct.precision.FLOAT16,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)

print("Generated MIL program:")
# print(mlmodel._mil_program)
mil_program_path = "/tmp/mil_program.txt"
with open(mil_program_path, "w") as f:
    f.write(str(mlmodel._mil_program))
print(f"MIL program written to: {mil_program_path}")

# Save - unquantized
mlmodel.save("/tmp/mil-moesingle.mlpackage")

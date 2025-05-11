import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
import numpy as np

# Define external weights with grouped expert shape
# NUM_EXPERTS = 2    # can gen MIL program, but takes long. why?
# NUM_EXPERTS = 4
NUM_EXPERTS = 8

# test 
# HIDDEN=512
# INTERMEDIATE=1024

# as in phi 3.5 moe
HIDDEN=4096
INTERMEDIATE=6400

'''
benchmark res
https://docs.google.com/spreadsheets/d/1VLT8dAXyJv-VEhJd30I9MWF0WiVvF1CXKn7pGLh1fAI/edit?usp=sharing

'''

# Conv weight shape: (N_out, N_in, 1, 1)
EXPERT_UP = np.zeros((NUM_EXPERTS, INTERMEDIATE, HIDDEN, 1, 1), dtype=np.float16)  
EXPERT_DOWN = np.zeros((NUM_EXPERTS, HIDDEN, INTERMEDIATE, 1, 1), dtype=np.float16) 
GATE_WEIGHT = np.zeros((NUM_EXPERTS, HIDDEN, 1, 1), dtype=np.float16)

# use slice_by_size to put together the expert weights
# limitations: weights by copy and cpu only. see comments inline
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
        # output_indices_dtype="uint16",  # int32 unsupported on ANE, but uint16 not compatibler with indexing etc
        name="topk_weights_1_cast_fp16"
    )

    norm = mb.reduce_sum(x=topk_vals, axes=[-1], keep_dims=True, name="op_54_cast_fp16")
    topk_weights = mb.real_div(x=topk_vals, y=norm, name="topk_weights_cast_fp16")

    # w1 and w2 (scalar multipliers)
    w1 = mb.slice_by_index(
        x=topk_weights,
        begin=[0, 0],   # either python type, list/tuple of python type, or coreml "Var"
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
    i1 = mb.slice_by_index(x=topk_indices, begin=[0, 0], end=[1, 1], squeeze_mask=[False, True])
    i2 = mb.slice_by_index(x=topk_indices, begin=[0, 1], end=[1, 2], squeeze_mask=[False, True])
    
    # Expert weights as const tensors
    expert_up = mb.const(val=EXPERT_UP, name="expert_up_weight")  # shape [nexp, in, out, 1, 1]
    expert_down = mb.const(val=EXPERT_DOWN, name="expert_down_weight")

    begin = mb.concat(values=[
    i1,  #Var
    mb.const(val=[0, 0, 0, 0])
    ], axis=0)
    size = mb.const(val=[1, -1, -1, -1, -1])

    up1 = mb.slice_by_size(
        x=expert_up,
        begin=begin,
        size=size,
    )

    begin = mb.concat(values=[
    i1,  #Var
    mb.const(val=[0, 0, 0, 0])
    ], axis=0)
    size = mb.const(val=[1, -1, -1, -1, -1])    

    down1 = mb.slice_by_size(
        x=expert_down,
        begin=begin,
        size=size,
    )

    up1 = mb.squeeze(x=up1, axes=[0], name="squeezed_up1")   
    down1 = mb.squeeze(x=down1, axes=[0], name="squeezed_down1")   

    x1 = mb.conv(x=x, weight=up1, strides=[1, 1], pad_type="valid", name="expert1_up")
    x1 = mb.silu(x=x1, name="expert1_act")
    x1 = mb.conv(x=x1, weight=down1, strides=[1, 1], pad_type="valid", name="expert1_down")
    x1 = mb.squeeze(x=mb.squeeze(x=x1, axes=[-1]), axes=[-1], name="expert1_out")
    out1 = mb.mul(x=w1, y=x1, name="out1_weighted")

    # Slice expert2 weights
    begin = mb.concat(values=[
        i2,  #Var
        mb.const(val=[0, 0, 0, 0])
    ], axis=0)
    size = mb.const(val=[1, -1, -1, -1, -1])

    up2 = mb.slice_by_size(
        x=expert_up,
        begin=begin,
        size=size,
    )

    begin = mb.concat(values=[
        i2,  #Var
        mb.const(val=[0, 0, 0, 0])
    ], axis=0)
    size = mb.const(val=[1, -1, -1, -1, -1])    
    down2 = mb.slice_by_size(
        x=expert_down,
        begin=begin,
        size=size,
    )

    up2 = mb.squeeze(x=up2, axes=[0], name="squeezed_up2")   
    down2 = mb.squeeze(x=down2, axes=[0], name="squeezed_down2")
    x2 = mb.conv(x=x, weight=up2, strides=[1, 1], pad_type="valid", name="expert2_up")
    x2 = mb.silu(x=x2, name="expert2_act")
    x2 = mb.conv(x=x2, weight=down2, strides=[1, 1], pad_type="valid", name="expert2_down")
    x2 = mb.squeeze(x=mb.squeeze(x=x2, axes=[-1]), axes=[-1], name="expert2_out")
    out2 = mb.mul(x=w2, y=x2, name="out2_weighted")

    combined = mb.add(x=out1, y=out2, name="combined_output")
    output = mb.reshape(x=combined, shape=[1, 1, HIDDEN], name="final_outputs")
    return output

# print(prog)

# below: experts no longer unsqueezed.  use matmul instead of conv. test if slice_by_size can avoid copy (-nope
EXPERT_UP = np.zeros((NUM_EXPERTS, INTERMEDIATE, HIDDEN), dtype=np.float16)  # up: hidden -> intermediate
EXPERT_DOWN = np.zeros((NUM_EXPERTS, HIDDEN, INTERMEDIATE), dtype=np.float16)  # down: intermediate -> hidden
GATE_WEIGHT = np.zeros((NUM_EXPERTS, HIDDEN, 1, 1), dtype=np.float16)  # unchanged, used for conv2d gate

@mb.program(input_specs=[mb.TensorSpec((1, 1, HIDDEN), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog2(input):
    x = mb.transpose(x=input, perm=[0, 2, 1])  # shape (1, HIDDEN, 1)
    x = mb.squeeze(x=x, axes=[-1])  # shape (1, HIDDEN)

    gate_weight = mb.const(val=GATE_WEIGHT, name="gate_weight")
    x4gate = mb.expand_dims(x=x, axes=[-1])
    x4gate = mb.expand_dims(x=x4gate, axes=[-1])  # shape (1, HIDDEN, 1, 1)
    gate_out = mb.conv(x=x4gate, weight=gate_weight, strides=[1, 1], pad_type="valid")
    gate_out = mb.squeeze(x=gate_out, axes=[-1])
    router_logits = mb.squeeze(x=gate_out, axes=[-1])
    routing_weights = mb.softmax(x=router_logits, axis=-1)

    topk_vals, topk_indices = mb.topk(
        x=routing_weights, k=2, axis=-1, return_indices=True, name="topk"
    )

    norm = mb.reduce_sum(x=topk_vals, axes=[-1], keep_dims=True)
    topk_weights = mb.real_div(x=topk_vals, y=norm)

    w1 = mb.slice_by_index(x=topk_weights, begin=[0, 0], end=[1, 1], squeeze_mask=[True, True])
    w2 = mb.slice_by_index(x=topk_weights, begin=[0, 1], end=[1, 2], squeeze_mask=[True, True])

    i1 = mb.slice_by_index(x=topk_indices, begin=[0, 0], end=[1, 1], squeeze_mask=[False, True])
    i2 = mb.slice_by_index(x=topk_indices, begin=[0, 1], end=[1, 2], squeeze_mask=[False, True])

    expert_up = mb.const(val=EXPERT_UP, name="expert_up")
    expert_down = mb.const(val=EXPERT_DOWN, name="expert_down")

    def apply_expert(i, w, name_suffix):
        begin = mb.concat(values=[i, mb.const(val=[0, 0])], axis=0)
        size = mb.const(val=[1, -1, -1])

        up = mb.slice_by_size(x=expert_up, begin=begin, size=size)
        down = mb.slice_by_size(x=expert_down, begin=begin, size=size)

        up = mb.squeeze(x=up, axes=[0])
        down = mb.squeeze(x=down, axes=[0])

        h = mb.matmul(x=x, y=mb.transpose(x=up, perm=[1, 0]), name=f"up_{name_suffix}")
        h = mb.silu(x=h)
        h = mb.matmul(x=h, y=mb.transpose(x=down, perm=[1, 0]), name=f"down_{name_suffix}")
        return mb.mul(x=h, y=w, name=f"weighted_{name_suffix}")

    out1 = apply_expert(i1, w1, "1")
    out2 = apply_expert(i2, w2, "2")

    combined = mb.add(x=out1, y=out2)
    output = mb.reshape(x=combined, shape=[1, 1, HIDDEN])
    return output


#  below: use the "list" ops... still slow???
IN=HIDDEN
OUT=INTERMEDIATE
EXPERT_UP_LIST = [np.random.rand(INTERMEDIATE, HIDDEN, 1, 1).astype(np.float16) for _ in range(NUM_EXPERTS)]
@mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, IN), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog_list(input):
    # Step 1: reshape input for conv
    x = mb.transpose(x=input, perm=[0, 2, 1])          # (1, HIDDEN, 1)
    x = mb.expand_dims(x=x, axes=[-1])                 # (1, HIDDEN, 1, 1)

    # Step 2: Create expert list
    expert_list = mb.make_list(
        init_length=NUM_EXPERTS,
        dynamic_length=False,
        elem_shape=[OUT, IN, 1, 1],
        # dtype=types.fp16,
        dtype='fp16',
        name="expert_list"
    )

    # Step 3: Write weights into expert_list
    #   list_write is fast, likely zero copy
    for idx, expert_array in enumerate(EXPERT_UP_LIST):
        const_weight = mb.const(val=expert_array, name=f"expert_{idx}_const")
        expert_list = mb.list_write(
            ls=expert_list,
            index=mb.const(val=idx),
            value=const_weight,
            name=f"write_expert_{idx}"
        )

    # Step 4: Dynamically select expert index (mock index 1 for this demo)
    #    list_read is cpu only, and it seems to copy.... expensive bad 
    #    update: even if we run on 'cpu only', list_read still expensive (likely copy)
    selected_index = mb.const(val=np.int32(1))  # or any runtime scalar
    selected_weight = mb.list_read(
        ls=expert_list,
        index=selected_index,
        name="selected_expert_weight"
    )

    # Step 5: Apply selected expert conv
    x = mb.conv(
        x=x,
        weight=selected_weight,
        strides=[1, 1],
        pad_type="valid",
        name="expert_conv"
    )

    # Step 6: Squeeze trailing dims and return output
    x = mb.squeeze(x=x, axes=[-2, -1], name="output_squeezed")  # (1, OUT)
    x = mb.reshape(x=x, shape=[1, 1, OUT], name="output_reshaped")
    return x

# use cascade select to get the expert weight .. silly 
EXPERT_UP_LIST = [np.random.rand(INTERMEDIATE, HIDDEN, 1, 1).astype(np.float16) for _ in range(NUM_EXPERTS)]
@mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, IN), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog_select(input):
    # Step 1: reshape input for conv
    x = mb.transpose(x=input, perm=[0, 2, 1])          # (1, HIDDEN, 1)
    x = mb.expand_dims(x=x, axes=[-1])                 # (1, HIDDEN, 1, 1)

    # Step 4: Dynamically select expert index (mock index 1 for this demo)

    x0 = mb.slice_by_index(
        x=x,
        begin=[0, 0, 0, 0],   # either python type, list/tuple of python type, or coreml "Var"
        end=[0,0,0,1],
        end_mask=[False, False,False,False],
        squeeze_mask=[True, True,True,True],
    )
    # selected_index = mb.const(val=np.int32(1))  # or any runtime scalar
    selected_index = mb.cast(x=x0,dtype="int32")  # or any runtime scalar

    y = mb.const(val=np.zeros((1, OUT, 1, 1), dtype=np.float16))

    # 0th expert...
    xx = mb.select(
        cond=mb.equal(x=selected_index, y=mb.const(val=np.int32(0))),
        a=mb.conv(x=x,weight=EXPERT_UP_LIST[0], strides=[1, 1], pad_type="valid"),
        b=mb.const(val=np.zeros((1, OUT, 1, 1), dtype=np.float16))
    )
    y = mb.add(x=y, y=xx)
    
    # 1st expert...
    xx = mb.select(
        cond=mb.equal(x=selected_index, y=mb.const(val=np.int32(1))),
        a=mb.conv(x=x, weight=EXPERT_UP_LIST[1], strides=[1, 1], pad_type="valid"),
        b=mb.const(val=np.zeros((1, OUT, 1, 1), dtype=np.float16))
    )
    y = mb.add(x=y, y=xx)

    # more...

    # Step 6: Squeeze trailing dims and return output
    y = mb.squeeze(x=y, axes=[-2, -1], name="output_squeezed")  # (1, OUT)
    y = mb.reshape(x=y, shape=[1, 1, OUT], name="output_reshaped")
    return y

TOP_K = 2

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
    
    ####  below, zero out all but top-k. works, but ... XXX  ######
    # no clean solution. topK can be done (cpu, not ANE), 
    #    but scatter seems quite tedious (and even if done, it's most likely cpu only????
    ''' 
    # 1.1: Get top-k indices and values
    _, indices = mb.topk(x=routing_weights, k=np.int32(TOP_K), axis=-1, return_indices=True)  # [1, K]
    updates = mb.gather(x=routing_weights, indices=indices, axis=-1)  # [1, K]

    # 1.2: Reshape for scatter compatibility
    updates = mb.expand_dims(x=updates, axes=[-1])  # [1, K] → [1, K, 1]

    # 1.3: Zero tensor: shape must match routing_weights + trailing dims (1 extra dim)
    zero_tensor = mb.fill(shape=[1, NUM_EXPERTS, 1], value=np.float16(0.0))  # [1, NUM_EXPERTS, 1]
    # zero_tensor = mb.mul(x=routing_weights, y=mb.const(val=np.array(0.0, dtype=np.float16))) # silly

    # 1.4: Scatter into position
    weights_full = mb.scatter(
        data=zero_tensor,
        indices=indices,
        updates=updates,
        axis=1
    )  # [1, NUM_EXPERTS, 1]

    # 1.5: Squeeze last dim back to [1, NUM_EXPERTS]
    weights_full = mb.squeeze(x=weights_full, axes=[-1])
    '''
    #######################################

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

def quant(mlmodel_conv, q=[8]):
    # 6. --- quant, and save again 
    import coremltools.optimize as cto
    GROUP_SIZE = 32  # default 
    mlmodel = mlmodel_conv

    for LUT_BITS in q:
        try:
            # Set up quantization config     XXX can try other quant config... (not just LUT
            # https://apple.github.io/coremltools/source/coremltools.optimize.coreml.palettization.html
            config = cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(
                    mode="kmeans",
                    nbits=LUT_BITS,
                    granularity="per_grouped_channel",
                    group_size=GROUP_SIZE,
                    num_kmeans_workers=1   # fxl: can only do 1
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
        
        # 6. Save
        PATH = "/tmp/mil-moesingle-NEXP%d-LUT%d.mlpackage" %(NUM_EXPERTS, LUT_BITS)
        mlmodel.save(PATH)
        print("Saved quantized model to:", PATH)


# return converted mlmodel
def convert():
    mlmodel = ct.convert(
        # prog,
        # prog1,
        # prog2,
        # prog_list,
        prog_select,
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
    mlmodel.save("/tmp/mil-moesingle-NEXP%d.mlpackage" %NUM_EXPERTS)
    print("Saved model to: /tmp/mil-moesingle-NEXP%d.mlpackage" %NUM_EXPERTS)    

    # 5. Test
    import time

    dummy_input = {"input": np.random.rand(1, 1, HIDDEN).astype(np.float16)}

    # Warm-up run
    mlmodel.predict(dummy_input)

    # Measure 10 runs
    timings = []
    for _ in range(10):
        start = time.time()
        _ = mlmodel.predict(dummy_input)
        end = time.time()
        timings.append(end - start)

    # Report median time
    median_time = np.median(timings)
    print(f"Median prediction time over 10 runs: {median_time:.6f} seconds")

    return mlmodel

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert and optionally quantize a CoreML model.")
    parser.add_argument("--quant", nargs="+", type=int, help="List of integers for quantization (e.g., --quant 2 4 6 8).")
    args = parser.parse_args()

    model = convert()

    if args.quant:
        quant(model, args.quant)
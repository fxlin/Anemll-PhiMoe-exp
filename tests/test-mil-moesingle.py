import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
import numpy as np


# fewer experts --> longer compilation time. why??? bad 

# NUM_EXPERTS = 2    # can gen MIL program, but takes long. why?
# NUM_EXPERTS = 4
# NUM_EXPERTS = 8
NUM_EXPERTS = 16

TOP_K = 2

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


#  below: use the "list" ops... NOT working. see comments
IN=HIDDEN
OUT=INTERMEDIATE
EXPERT_UP_LIST = [np.random.rand(INTERMEDIATE, HIDDEN, 1, 1).astype(np.float16) for _ in range(NUM_EXPERTS)]
EXPERT_DOWN_LIST = [np.random.rand(HIDDEN, INTERMEDIATE, 1, 1).astype(np.float16) for _ in range(NUM_EXPERTS)]
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

# use cascade select to get the expert weight .. BAD, see comments
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
    selected_index = mb.mod(x=selected_index, y=mb.const(val=np.int32(NUM_EXPERTS)))  # ensure within bounds

    y = mb.const(val=np.zeros((1, OUT, 1, 1), dtype=np.float16))   # accumulator 
    zzz = mb.const(val=np.zeros((1, OUT, 1, 1), dtype=np.float16))

    # # 0th expert...
    # xx = mb.select(
    #     cond=mb.equal(x=selected_index, y=mb.const(val=np.int32(0))),
    #     a=mb.conv(x=x,weight=EXPERT_UP_LIST[0], strides=[1, 1], pad_type="valid"),
    #     b=mb.const(val=np.zeros((1, OUT, 1, 1), dtype=np.float16))
    # )
    # y = mb.add(x=y, y=xx)

    # build a static graph like this. will be unrolled at compoile time 
    for i in range(NUM_EXPERTS):
        xx = mb.select(
            cond=mb.equal(x=selected_index, y=mb.const(val=np.int32(i))),
            a=mb.conv(x=x, weight=EXPERT_UP_LIST[i], strides=[1, 1], pad_type="valid"),
            b=zzz
        )
        y = mb.add(x=y, y=xx)

    '''
    bad - both values in select() will be eagerly evaluated, prior to select(), so:

    %conv_3: (1, 6400, 1, 1, fp16)(Tensor) = conv(x=%expand_dims_3, weight=%conv_3_weight_0, strides=[1, 1], pad_type="valid", pad=[0, 0, 0, 0], dilations=[1, 1], groups=1, name="conv_3")
    %select_2: (1, 6400, 1, 1, fp16)(Tensor) = select(cond=%equal_2, a=%conv_3, b=%const_22, name="select_2")
    %add_3: (1, 6400, 1, 1, fp16)(Tensor) = add(x=%add_2, y=%select_2, name="add_3")
    %equal_3: (bool)(Scalar) = equal(x=%mod_0, y=3, name="equal_3")
    ^ ^ conv will be executed regardless of the condition
    i.e. select() does not build code blocks
    '''

    # Step 6: Squeeze trailing dims and return output
    y = mb.squeeze(x=y, axes=[-2, -1], name="output_squeezed")  # (1, OUT)
    y = mb.reshape(x=y, shape=[1, 1, OUT], name="output_reshaped")
    return y


# use cond
'''
works on cpu, NOT WORKING on ANE>.....
- this seems to compile as epexfted (conditional branch). 
- one branhc seems executed, although static analysis cannot confirm. 
X however the "cond" blocks seem always on cpu (including its true/false fns), 
X  regardless of how expensive how true/false fns are....

M2 max: 
bs=1     2 ms
bs=8     8 ms
'''

# bs=8
bs=1
@mb.program(input_specs=[mb.TensorSpec(shape=(1, bs, IN), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog_cond(input):
    # Step 1: reshape input for conv
    # x = mb.transpose(x=input, perm=[0, 2, 1])          # (1, HIDDEN, 1)
    x = mb.transpose(x=input, perm=[1, 2, 0])          # (bs, HIDDEN, 1)
    # x = mb.expand_dims(x=input, axes=[-1])                 # (bs, HIDDEN, 1)
    x = mb.expand_dims(x=x, axes=[-1])                 # (bs, HIDDEN, 1, 1)

    # below: fake a dynamic index ... select one expert 
    x0 = mb.slice_by_index(
        x=x,
        begin=[0, 0, 0, 0],   # either python type, list/tuple of python type, or coreml "Var"
        end=[0,0,0,1],
        end_mask=[False, False,False,False],
        squeeze_mask=[True, True,True,True],
    )
    selected_index = mb.cast(x=x0,dtype="int32")  # or any runtime scalar
    selected_index = mb.mod(x=selected_index, y=mb.const(val=np.int32(NUM_EXPERTS)))  # ensure within bounds

    y = mb.const(val=np.zeros((bs, HIDDEN, 1, 1), dtype=np.float16))   # accumulator 
    zzz = mb.const(val=np.zeros((bs, HIDDEN, 1, 1), dtype=np.float16))
    
    # build a static graph like this. will be unrolled at compoile time 
    for i in range(NUM_EXPERTS):
        def true_fn(): 
            # return mb.conv(x=x, weight=EXPERT_UP_LIST[i], strides=[1, 1], pad_type="valid")
            xx = mb.conv(x=x, weight=EXPERT_UP_LIST[i], strides=[1, 1], pad_type="valid")   # up proj
            xx = mb.silu(x=xx)  # activation
            xx = mb.conv(x=xx, weight=EXPERT_DOWN_LIST[i], strides=[1, 1], pad_type="valid")  # down proj
            return xx
        def false_fn():
            return zzz
        
        xxx = mb.cond(
            pred=mb.equal(x=selected_index, y=mb.const(val=np.int32(i))),
            _true_fn=true_fn,
            _false_fn=false_fn
        )
        y = mb.add(x=y, y=xxx)

    y = mb.squeeze(x=y, axes=[-2, -1], name="output_squeezed")  # (1, HIDDEN)
    y = mb.reshape(x=y, shape=[1, bs, HIDDEN], name="output_reshaped")
    return y

# based on above, but now with gate + topk for exp selection 
@mb.program(input_specs=[mb.TensorSpec(shape=(1, bs, IN), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog_cond_gate(input):
    # Step 1: reshape input for conv
    # x = mb.transpose(x=input, perm=[0, 2, 1])          # (1, HIDDEN, 1)
    x = mb.transpose(x=input, perm=[1, 2, 0])          # (bs, HIDDEN, 1)
    # x = mb.expand_dims(x=input, axes=[-1])                 # (bs, HIDDEN, 1)
    x = mb.expand_dims(x=x, axes=[-1])                 # (bs, HIDDEN, 1, 1)

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

    # cpu only 
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

    # selected expert indices
    i1 = mb.slice_by_index(x=topk_indices, begin=[0, 0], end=[1, 1], squeeze_mask=[False, True])
    i2 = mb.slice_by_index(x=topk_indices, begin=[0, 1], end=[1, 2], squeeze_mask=[False, True])

    # iterate through all experts ... 
    y = mb.const(val=np.zeros((bs, HIDDEN, 1, 1), dtype=np.float16))   # accumulator 
    zzz = mb.const(val=np.zeros((bs, HIDDEN, 1, 1), dtype=np.float16))
    
    # build a static graph like this. will be unrolled at compoile time 
    for i in range(NUM_EXPERTS):
        def true_fn(): 
            # return mb.conv(x=x, weight=EXPERT_UP_LIST[i], strides=[1, 1], pad_type="valid")
            xx = mb.conv(x=x, weight=EXPERT_UP_LIST[i], strides=[1, 1], pad_type="valid")   # up proj
            xx = mb.silu(x=xx)  # activation
            xx = mb.conv(x=xx, weight=EXPERT_DOWN_LIST[i], strides=[1, 1], pad_type="valid")  # down proj
            return xx
        def false_fn():
            return zzz
        
        xxx = mb.cond(
            pred=mb.equal(x=i1, y=mb.const(val=np.int32(i))),
            _true_fn=true_fn,
            _false_fn=false_fn
        )
        xxx = mb.mul(x=xxx, y=w1)   # weighted
        y = mb.add(x=y, y=xxx)

        xxx = mb.cond(
            pred=mb.equal(x=i2, y=mb.const(val=np.int32(i))),
            _true_fn=true_fn,
            _false_fn=false_fn
        )
        xxx = mb.mul(x=xxx, y=w2)   # weighted
        y = mb.add(x=y, y=xxx)

    y = mb.squeeze(x=y, axes=[-2, -1], name="output_squeezed")  # (1, HIDDEN)
    y = mb.reshape(x=y, shape=[1, bs, HIDDEN], name="output_reshaped")
    return y

##################################
# a full transformer block with moe, prefill mode
# based on prog_cond_gate
BS=64
CONTEXT_LEN=128
HEAD_DIM=128
NUM_Q_HEADS = HIDDEN // HEAD_DIM
NUM_KV_GROUPS = 4   # for group query
NUM_KV_HEADS = NUM_Q_HEADS // NUM_KV_GROUPS
NUM_LAYERS = 1   # for kvcache sz
#   weights XXX add bias
Q_WEIGHT = np.random.rand(HIDDEN, HIDDEN, 1, 1).astype(np.float16)
K_WEIGHT = np.random.rand(HIDDEN//NUM_KV_GROUPS, HIDDEN, 1, 1).astype(np.float16)    # b/c group query, we have 4x fewer k/v
V_WEIGHT = np.random.rand(HIDDEN//NUM_KV_GROUPS, HIDDEN, 1, 1).astype(np.float16)    # b/c group query, we have 4x fewer k/v
# O_WEIGHT = np.random.rand(HIDDEN, HIDDEN, 1, 1).astype(np.float16)   # output proj
O_WEIGHT = np.random.rand(HIDDEN, HIDDEN).astype(np.float16)   # output proj, using matmul not conv (to avoid reshaping for subsequent residual connection?

# states shape (1, NUM_HEADS, BS, HEAD_DIM). then apply rotation to each head
#   NB: b/c of elementwise op and bdcast, this applies to both K and V, depsite their different head number (group query)
#  cos shape [1,1,bs=64,head_dim=128].. there's redundancy. the 1st & 2nd halves are same values...
def helper_rotate(states, cos, sin):
    x1=mb.slice_by_size(x=states, begin=[0, 0, 0, 0], size=[-1, -1, -1, HEAD_DIM//2])     # 1st half of hidden
    x2=mb.slice_by_size(x=states, begin=[0, 0, 0, HEAD_DIM//2], size=[-1, -1, -1, HEAD_DIM//2])   # 2nd half of hidden

    # cut cos/sin in half, as there's redundancy
    cos = mb.slice_by_size(x=cos, begin=[0, 0, 0, 0], size=[-1, -1, -1, HEAD_DIM//2])
    sin = mb.slice_by_size(x=sin, begin=[0, 0, 0, 0], size=[-1, -1, -1, HEAD_DIM//2])

    # BELOW -- elementwise op, bdcast to higher dims
    # rotated = torch.cat([
    #             x1 * cos - x2 * sin,
    #             x2 * cos + x1 * sin
    #         ], dim=-1)
    r1 = mb.sub(x=mb.mul(x=x1, y=cos), y=mb.mul(x=x2, y=sin))
    r2 = mb.add(x=mb.mul(x=x1, y=sin), y=mb.mul(x=x2, y=cos))
    rotated = mb.concat(values=[r1, r2], axis=-1)
    return rotated

layer_id = 0    # in the future, a sequence of layer ids will be statically defined 
current_pos = 0 # shall be passed in as a static value (XXX as a tensor?? too much dynamism)

# cf: phimoe_model.py process_layer_prefill()
@mb.program(input_specs=[
    mb.TensorSpec(shape=(1, BS, HIDDEN), dtype=types.fp16),     # hidden_states
    mb.TensorSpec(shape=(1, BS), dtype=types.int32),            # position_ids
    # mb.TensorSpec(shape=(1,), dtype=types.fp32),       # current_pos, for indexing kvcache
    # mb.TensorSpec(shape=(1, 1, BS, CONTEXT_LEN), dtype=types.bool),            # causal_mask, bool, reused across layers so should be passed in
    mb.TensorSpec(shape=(1, 1, BS, CONTEXT_LEN), dtype=types.fp16),            # causal_mask. although scaled_dot_product_attention can deals with bool, "Core ML I/O" does not support input as bool??
    mb.TensorSpec(shape=(1, BS, 1, HEAD_DIM), dtype=types.fp16),            # rotary_emb_cos
    mb.TensorSpec(shape=(1, BS, 1, HEAD_DIM), dtype=types.fp16),            # rotary_emb_sin
    mb.StateTensorSpec(shape=(2*NUM_LAYERS, NUM_KV_HEADS, CONTEXT_LEN, HEAD_DIM), dtype=types.fp16),            # kvcache (as state)
    ], opset_version=ct.target.iOS18)
def prog_prefill(hidden_states_input, position_ids, 
                #  current_pos, 
                 causal_mask, rotary_emb_cos, rotary_emb_sin, kvcache_state):
    # hidden_states_input: (1, BS, HIDDEN), pre-norm (pre-ln) input as the residual connection

    # input layernorm (should we do rmsnorm instead? but coreml has no such op yet, can customize if needed
    #   XXX axes=-1 correct? inner most axis.
    hidden_states = mb.layer_norm(x=hidden_states_input, axes=[-1], name="layernorm_1", epsilon=1e-5)

    hidden_states = mb.transpose(x=hidden_states, perm=[0, 2, 1])  # (1, HIDDEN, BS)
    hidden_states = mb.expand_dims(x=hidden_states, axes=[2])  # (1, HIDDEN, 1, BS)
    #  channel first, last axis non-zero... good for ANE

    query_states = mb.conv(
        x=hidden_states,
        weight=mb.const(val=Q_WEIGHT, name="query_weight"),
        strides=[1, 1],
        pad_type="valid",
        name="query_conv"
    )
    key_states = mb.conv(
        x=hidden_states,
        weight=mb.const(val=K_WEIGHT, name="key_weight"),
        strides=[1, 1],
        pad_type="valid",
        name="key_conv"
    )
    value_states = mb.conv(
        x=hidden_states,
        weight=mb.const(val=V_WEIGHT, name="value_weight"),
        strides=[1, 1],
        pad_type="valid",
        name="value_conv"
    )
    # arrange by head, head first 
    query_states = mb.reshape(x=query_states, shape=[1, NUM_Q_HEADS, HEAD_DIM, BS])
    query_states = mb.transpose(x=query_states, perm=[0, 1, 3, 2])  # (1, NUM_HEADS, BS, HEAD_DIM)
    key_states = mb.reshape(x=key_states, shape=[1, NUM_KV_HEADS, HEAD_DIM, BS])
    key_states = mb.transpose(x=key_states, perm=[0, 1, 3, 2])  # (1, NUM_KV_HEADS, BS, HEAD_DIM)
    value_states = mb.reshape(x=value_states, shape=[1, NUM_KV_HEADS, HEAD_DIM, BS])
    value_states = mb.transpose(x=value_states, perm=[0, 1, 3, 2])  # (1, NUM_KV_HEADS, BS, HEAD_DIM)

    # rotary embedding. the shape faciliates elementwise op when applying to each head of q/k 
    cos = mb.transpose(x=rotary_emb_cos, perm=[0, 2, 1, 3])  # (1,1,BS,HEAD_DIM)
    sin = mb.transpose(x=rotary_emb_sin, perm=[0, 2, 1, 3])  # (1,1,BS,HEAD_DIM)
    #  apply rotary embedding to q/k, same rotation applied to each head, different across positions
    query_states = helper_rotate(states=query_states, cos=cos, sin=sin)
    key_states = helper_rotate(states=key_states, cos=cos, sin=sin)

    ###########################
    # kvcache manitianence 
    # first load from "state" (copy??? expensive???)
    kvcache = mb.read_state(input=kvcache_state)
    # then update kvcache. WILL THIS CREATE A NEW KVCACHE TENSOR?? 
    kvcache = mb.slice_update(
        x=kvcache,
        update=key_states,
        begin=[layer_id, 0, current_pos, 0],
        end=[layer_id+1, NUM_KV_HEADS, current_pos + BS, HEAD_DIM],
    )
    kvcache = mb.slice_update(
        x=kvcache,
        update=value_states,
        begin=[layer_id+NUM_LAYERS, 0, current_pos, 0],
        end=[layer_id+NUM_LAYERS+1, NUM_KV_HEADS, current_pos + BS, HEAD_DIM],
    )
    mb.coreml_update_state(state=kvcache_state, value=kvcache)  # write back kvcache 

    #################################
    # attention wrt the kvcache

    # extract from kvcache, and duplicate heads for gropu query (to match query head)
    K_layer_cache = mb.slice_by_size(
        x=kvcache,
        begin=[layer_id, 0, 0, 0],
        size=[1, -1, CONTEXT_LEN, -1],        
    ) # shape (1, NUM_KV_HEADS, CONTEXT_LEN, HEAD_DIM)
    K_layer_cache = mb.tile(x=K_layer_cache, reps=[1, NUM_Q_HEADS // NUM_KV_HEADS, 1, 1])  
    # shape (1, NUM_Q_HEADS, CONTEXT_LEN, HEAD_DIM)
    V_layer_cache = mb.slice_by_size(
        x=kvcache,
        begin=[layer_id+NUM_LAYERS, 0, 0, 0],
        size=[1, -1, CONTEXT_LEN, -1]
    )
    V_layer_cache = mb.tile(x=V_layer_cache, reps=[1, NUM_Q_HEADS // NUM_KV_HEADS, 1, 1]) # ditto 

    # https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#module-coremltools.converters.mil.mil.ops.defs.iOS18.transformers 
    attn_output = mb.scaled_dot_product_attention(
        query = query_states,
        key = K_layer_cache,
        value = V_layer_cache,
        attn_mask = causal_mask
    )     # shape [1, NUM_Q_HEADS, BS, HEAD_DIM]

    # concate heads
    attn_output = mb.transpose(x=attn_output, perm=[0, 1, 3, 2])
    attn_output = mb.reshape(x=attn_output, shape=[1, BS, HIDDEN])   # shape [1, BS, HIDDEN]

    # proj output --- not using conv? (as in Anemll code) why?
    attn_output = mb.matmul(x=attn_output, y=mb.const(val=O_WEIGHT, name="output_weight"), name="output_proj")

    # residual connect, from pre-norm input
    attn_output = mb.add(x=attn_output, y=hidden_states_input)   # shape [1, BS, HIDDEN]

    ######################################
    ###########  MLP ####################
    mlp_input = mb.layer_norm(x=attn_output, axes=[-1], name="layernorm_2", epsilon=1e-5) # norm for mlp   XXX is axes=[-1] correct?

    # expert router 
    x = mb.transpose(x=mlp_input, perm=[1, 2, 0])          # (BS, HIDDEN, 1)
    x = mb.expand_dims(x=x, axes=[-1])                 # (BS, HIDDEN, 1, 1)

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

    # cpu only 
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
    
    # selected expert indices
    i1 = mb.slice_by_index(x=topk_indices, begin=[0, 0], end=[1, 1], squeeze_mask=[False, True])
    i2 = mb.slice_by_index(x=topk_indices, begin=[0, 1], end=[1, 2], squeeze_mask=[False, True])

    # iterate through all experts ... 
    y = mb.const(val=np.zeros((BS, HIDDEN, 1, 1), dtype=np.float16))   # accumulator 
    zzz = mb.const(val=np.zeros((BS, HIDDEN, 1, 1), dtype=np.float16))

    # build a static graph like this. will be unrolled at compoile time 
    for i in range(NUM_EXPERTS):
        def true_fn(): 
            # return mb.conv(x=x, weight=EXPERT_UP_LIST[i], strides=[1, 1], pad_type="valid")
            xx = mb.conv(x=x, weight=EXPERT_UP_LIST[i], strides=[1, 1], pad_type="valid")   # up proj
            xx = mb.silu(x=xx)  # activation
            xx = mb.conv(x=xx, weight=EXPERT_DOWN_LIST[i], strides=[1, 1], pad_type="valid")  # down proj
            return xx
        def false_fn():
            return zzz
        
        xxx = mb.cond(
            pred=mb.equal(x=i1, y=mb.const(val=np.int32(i))),
            _true_fn=true_fn,
            _false_fn=false_fn
        )
        xxx = mb.mul(x=xxx, y=w1)   # weighted
        y = mb.add(x=y, y=xxx)

        xxx = mb.cond(
            pred=mb.equal(x=i2, y=mb.const(val=np.int32(i))),
            _true_fn=true_fn,
            _false_fn=false_fn
        )
        xxx = mb.mul(x=xxx, y=w2)   # weighted
        y = mb.add(x=y, y=xxx)

    y = mb.squeeze(x=y, axes=[-2, -1], name="output_squeezed")  # (1, HIDDEN)
    y = mb.reshape(x=y, shape=[1, BS, HIDDEN], name="output_reshaped")

    result = mb.add(x=y, y=attn_output)   # residual connect     (pre-ln)
    return result

# 1 token sent to all experts. use routing weights to sum up. can mapp well to ANE
# M2 max (mostly on ANE): 
#   4 experts    7ms
#   8 experts    14ms (almost 2x slower
#   16 experts   40ms (~3x slower???
# SLOWER THAN prog_cond which selectivesly run experts on CPU
@mb.program(input_specs=[mb.TensorSpec((1, 1, HIDDEN), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog_all_experts(input):
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

    # Expand routing weights ("scores") across channels [1, HIDDEN * NUM_EXPERTS]
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
        # prog_all_experts,
        # prog2,
        # prog_list,
        # prog_select,
        # prog_cond,
        # prog_cond_gate,
        prog_prefill,
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

    breakpoint()

    # Save - unquantized
    mlmodel.save("/tmp/mil-moesingle-NEXP%d.mlpackage" %NUM_EXPERTS)
    print("Saved model to: /tmp/mil-moesingle-NEXP%d.mlpackage" %NUM_EXPERTS)    

    # 5. Test
    import time

    dummy_input = {"input": np.random.rand(1, bs, HIDDEN).astype(np.float16)}

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
#!/usr/bin/env python3

# source ~/workspace-apple-silicon/myenv-python39/bin/activate
# ok, can offload to ANE well 

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
import numpy as np
import torch
import argparse

# Matrix size
N = 4096
bs = 8

# 1. Dummy weights (N_out, N_in)
W_torch = torch.randn(N, N)  # Conv weight shape: (N_out, N_in, 1, 1)
W_conv = W_torch.unsqueeze(-1).unsqueeze(-1).numpy().astype(np.float16)

# 2. Define MIL program using convolution
@mb.program(input_specs=[mb.TensorSpec(shape=(bs, N), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog(X):
    # Reshape input (bs, N) -> (bs, N, 1, 1)
    X_reshaped = mb.reshape(x=X, shape=[bs, N, 1, 1])

    # Create weighits; of shape (N_out, N_in, 1, 1)
    W = mb.const(val=W_conv, name="conv_weight")

    # Perform convolution: output shape will be (bs, N_out, 1, 1)
    Y = mb.conv(x=X_reshaped, weight=W, name="output_conv0", strides=[1, 1], pad_type="valid")

    # Reshape back to (bs, N)
    Y_out = mb.reshape(x=Y, shape=[bs, N], name="output_conv")
    return Y_out

# a different way to reshape, permute dims
@mb.program(input_specs=[mb.TensorSpec(shape=(bs, N), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog1(X):
    # Step 1: reshape (bs, N) → (bs, N_in, 1, 1)
    X_reshaped = mb.reshape(x=X, shape=[bs, N, 1, 1])

    # Step 2: permute → (1, N_in, bs, 1)
    X_permuted = mb.transpose(x=X_reshaped, perm=[2, 1, 0, 3])  # (bs, N, 1, 1) → (1, N, bs, 1)

    # Step 3: weight of shape (N_out, N_in=1, kernel_H=1, kernel_W=1)
    # Create weighits; of shape (N_out, N_in, 1, 1)
    W = mb.const(val=W_conv, name="conv_weight")

    # Step 4: perform convolution
    Y = mb.conv(
        x=X_permuted,
        weight=W,
        name="output_conv0",
        strides=[1, 1],
        pad_type="valid"
    )  # output: (1, N_out=N, bs, 1)

    # Step 5: permute back → (bs, N)
    Y_permuted = mb.transpose(x=Y, perm=[2, 1, 0, 3])  # (1, N, bs, 1) → (bs, N, 1, 1)
    Y_out = mb.reshape(x=Y_permuted, shape=[bs, N], name="output_conv")

    return Y_out

# return converted mlmodel
def convert():
    # 3. Convert to CoreML
    mlmodel_conv = ct.convert(
        # prog,
        prog1,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    # 4. Save
    PATH = "/tmp/SimpleMatmulConv.mlpackage"
    mlmodel_conv.save(PATH)
    print("Saved model to:", PATH)

    # 5. Test
    X_sample = np.random.rand(bs, N).astype(np.float16)
    result = mlmodel_conv.predict({"X": X_sample})
    print("Output shape (conv):", result["output_conv"].shape)
    return mlmodel_conv

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
        PATH = "/tmp/SimpleMatmulConv-LUT%d.mlpackage" %LUT_BITS
        mlmodel.save(PATH)
        print("Saved quantized model to:", PATH)

        # 5. Test again 
        X_sample = np.random.rand(bs, N).astype(np.float16)
        result = mlmodel.predict({"X": X_sample})
        print("Output shape (conv):", result["output_conv"].shape)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert and optionally quantize a CoreML model.")
    parser.add_argument("--quant", nargs="+", type=int, help="List of integers for quantization (e.g., --quant 2 4 6 8).")
    args = parser.parse_args()

    mlmodel_conv = convert()

    if args.quant:
        quant(mlmodel_conv, args.quant)
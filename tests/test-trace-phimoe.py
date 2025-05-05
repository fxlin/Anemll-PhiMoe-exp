import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

'''
 from top level, do 
python3 -m tests.test-trace-phimoe
'''
# from ..anemll.models.phimoe_model import (
from anemll.models.phimoe_model import (
    PhimoeSparseMoeBlock,
    # PhimoeConfig
)

# Conversion to CoreML
def convert_to_coreml(model, sample_input, output_path):
    traced_model = torch.jit.trace(model, sample_input)
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=sample_input.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        debug=True,
    )
    # print(mlmodel._mil_program)    # works 
    with open("/tmp/phimoe.mil", "w") as f:
        f.write(str(mlmodel._mil_program))
    mlmodel.save(output_path)
    return mlmodel

seqlen = 64    # test prefill 

if __name__ == "__main__":
    # fxl: fake a "config" insetad of going full PhimoeConfig
    class Config:
        hidden_size = 512
        intermediate_size = 1024
        # hidden_size = 4096      # phi
        # intermediate_size = 6400        # phi           
        num_local_experts = 16
        num_experts_per_tok = 2
        hidden_act = 'silu'
        router_jitter_noise = 0.1
        input_jitter_noise = 0.1
    
    config = Config()
    model = PhimoeSparseMoeBlock(config)
    model.eval()
    model.half()

    sample_input = torch.randn(1, seqlen, config.hidden_size).half()     # bs=1, seqlen=32 
    coreml_model = convert_to_coreml(model, sample_input, "/tmp/phimoe-Conv2d.mlpackage")
    
    # TODo: quant  the model

    with torch.no_grad():
        torch_output = model(sample_input)

    coreml_output = coreml_model.predict({"input": sample_input.cpu().numpy()})["output"]

    print("PyTorch output shape:", torch_output.shape)
    print("CoreML output shape:", coreml_output.shape)
    print("Max difference:", np.max(np.abs(torch_output.cpu().numpy() - coreml_output)))

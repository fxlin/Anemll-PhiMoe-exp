import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

'''
by deepseek, simplified MoE
'''
class TraceablePhimoeSparseMoeBlock(nn.Module):
    """
    A traceable version of PhimoeSparseMoeBlock that can be converted to CoreML
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        
        # Use Conv1d instead of Conv2d for better CoreML compatibility
        self.gate = nn.Conv1d(self.hidden_dim, self.num_experts, kernel_size=1, bias=False)
        
        # Experts as sequential layers
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.ffn_dim, kernel_size=1, bias=False),
                nn.SiLU(),  # Using SiLU as it's well supported
                nn.Conv1d(self.ffn_dim, self.hidden_dim, kernel_size=1, bias=False)
            ) for _ in range(self.num_experts)
        ])

        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Reshape for conv1d
        hidden_states_reshaped = hidden_states.transpose(1, 2)  # [batch, hidden, seq]
        
        # Get router logits
        router_logits = self.gate(hidden_states_reshaped).transpose(1, 2)  # [batch, seq, experts]
        router_logits = router_logits.reshape(-1, self.num_experts)  # [batch*seq, experts]
        
        # Simplified top-k routing (without training noise)
        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_indices = topk_indices.to(torch.int32)  # not helpful
        
        # Normalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Process through each expert
        final_hidden_states = torch.zeros(
            batch_size * sequence_length, 
            hidden_dim, 
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # Create expert mask
        expert_mask = torch.zeros(
            self.num_experts,
            batch_size * sequence_length,
            dtype=torch.int32,
            device=hidden_states.device
        )
        
        # Set mask for tokens assigned to each expert
        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx).any(dim=-1)
            expert_mask[expert_idx] = mask
        
        # Process through experts
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim).unsqueeze(-1)  # [batch*seq, hidden, 1]
        
        # WILL BE UNROLL.... 
        for expert_idx in range(self.num_experts):
            mask = expert_mask[expert_idx]
            # if not mask.any():
            #     continue
                
            # Get weights for this expert   (fxl: "weights" as in "topk weights")
            expert_weights = torch.where(
                topk_indices == expert_idx,
                topk_weights,
                torch.zeros_like(topk_weights)
            ).sum(dim=-1)[mask]
            
            # Process through expert
            expert_out = self.experts[expert_idx](hidden_states_flat[mask]).to(dtype=hidden_states.dtype)
            expert_out = expert_out.squeeze(-1) * expert_weights.unsqueeze(-1)

            expert_out = expert_out.to(dtype=final_hidden_states.dtype)   # works...

            # print(f"[DEBUG] expert_out dtype: {expert_out.dtype}, final_hidden_states dtype: {final_hidden_states.dtype}")

            # Accumulate results
            final_hidden_states[mask] += expert_out
        
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

# Conversion to CoreML
def convert_to_coreml(model, sample_input, output_path):
    # Trace the model
    traced_model = torch.jit.trace(model, sample_input)

    # Convert to CoreML with fp16 precision
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=sample_input.shape, dtype=np.float16)],
        outputs=[
                ct.TensorType(name="output", dtype=np.float16)
            ],
        compute_precision=ct.precision.FLOAT16,  # <- Force CoreML output to be float16
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        debug=True,
    )
    
    print("Generated MIL program:")
    print(mlmodel._mil_program)

    # Save the model
    mlmodel.save(output_path)
    return mlmodel

# Example usage
if __name__ == "__main__":
    class Config:
        hidden_size = 512
        intermediate_size = 1024
        num_local_experts = 8
        num_experts_per_tok = 2
        router_jitter_noise = 0.1
        input_jitter_noise = 0.1
    
    config = Config()
    model = TraceablePhimoeSparseMoeBlock(config)
    model.eval()
    model.half()  # <-- important!

    # Create sample input with float16
    sample_input = torch.randn(1, 32, config.hidden_size).half()  # [batch, seq, hidden]
    
    # Convert to CoreML
    coreml_model = convert_to_coreml(model, sample_input, "/tmp/SparseMoeBlock.mlpackage")
    
    # Test prediction
    with torch.no_grad():
        torch_output = model(sample_input)

    # CoreML input must be float16 numpy array
    coreml_output = coreml_model.predict({"input": sample_input.cpu().numpy()})["output"]

    print("PyTorch output shape:", torch_output.shape)
    print("CoreML output shape:", coreml_output.shape)
    print("Max difference:", np.max(np.abs(torch_output.cpu().numpy() - coreml_output)))

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

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
        
        # Use Conv2d instead of Conv1d
        self.gate = nn.Conv2d(self.hidden_dim, self.num_experts, kernel_size=(1,1), bias=False)
        
        # Experts as sequential layers (use Conv2d)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.ffn_dim, kernel_size=(1,1), bias=False),
                nn.SiLU(),
                nn.Conv2d(self.ffn_dim, self.hidden_dim, kernel_size=(1,1), bias=False)
            ) for _ in range(self.num_experts)
        ])

        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Reshape for Conv2d: [batch, hidden, seq, 1]
        hidden_states_reshaped = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # [batch, hidden, seq, 1]
        
        # Get router logits
        router_logits = self.gate(hidden_states_reshaped).squeeze(-1).permute(0, 2, 1)  # [batch, seq, experts]
        router_logits = router_logits.reshape(-1, self.num_experts)  # [batch*seq, experts]
        
        # Simplified top-k routing
        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        final_hidden_states = torch.zeros(
            batch_size * sequence_length, 
            hidden_dim, 
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # Create expert mask  # shape: (num_experts=16, bs=64*seq_len)
        expert_mask = torch.zeros(
            self.num_experts,
            batch_size * sequence_length,
            dtype=torch.bool,
            device=hidden_states.device
        )
        
        for expert_idx in range(self.num_experts):
            # mask set to True for any topk indices that match expert_idx (scalar)
            mask = (topk_indices == expert_idx).any(dim=-1)   
            expert_mask[expert_idx] = mask
            # mask shape: [bs], type bool
        
        # Flatten hidden states for expert processing: [batch*seq, hidden, 1, 1]
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim).unsqueeze(-1).unsqueeze(-1)
        
        # Process through experts
        for expert_idx in range(self.num_experts):
            mask = expert_mask[expert_idx]
                
            expert_weights = torch.where(
                topk_indices == expert_idx,
                topk_weights,
                torch.zeros_like(topk_weights)
            ).sum(dim=-1)[mask]         # fxl: "weights" as in "attention weights".... here, expert_weights may be empty. (expert is not used, which seems fine) 
            
            expert_out = self.experts[expert_idx](hidden_states_flat[mask]).to(dtype=hidden_states.dtype)  # [tokens, hidden, 1, 1]
            expert_out = expert_out.squeeze(-1).squeeze(-1) * expert_weights.unsqueeze(-1)  # [tokens, hidden]
            expert_out = expert_out.to(dtype=final_hidden_states.dtype)
            
            final_hidden_states[mask] += expert_out
        
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

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
    print("Generated MIL program:")
    print(mlmodel._mil_program)
    mlmodel.save(output_path)
    return mlmodel

seqlen = 64    # test prefill

if __name__ == "__main__":
    class Config:
        hidden_size = 512
        intermediate_size = 1024
        # hidden_size = 4096      # phi
        # intermediate_size = 6400        # phi   
        num_local_experts = 16        
        num_experts_per_tok = 2
        router_jitter_noise = 0.1
        input_jitter_noise = 0.1
    
    config = Config()
    model = TraceablePhimoeSparseMoeBlock(config)
    model.eval()
    model.half()

    sample_input = torch.randn(1, seqlen, config.hidden_size).half()     # bs=1
    coreml_model = convert_to_coreml(model, sample_input, "/tmp/SparseMoeBlock-Conv2d.mlpackage")
    
    with torch.no_grad():
        torch_output = model(sample_input)

    coreml_output = coreml_model.predict({"input": sample_input.cpu().numpy()})["output"]

    print("PyTorch output shape:", torch_output.shape)
    print("CoreML output shape:", coreml_output.shape)
    print("Max difference:", np.max(np.abs(torch_output.cpu().numpy() - coreml_output)))

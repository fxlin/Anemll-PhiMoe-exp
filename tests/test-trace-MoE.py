import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import torch.nn.functional as F
from datetime import datetime

MODEL_DTYPE = torch.float16 
MODEL_ITYPE = torch.int16
# MODEL_DTYPE = torch.float32

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
        # only needed in some forward() versions. wont be traced if we dont use it. so it's ok 
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.ffn_dim, kernel_size=(1,1), bias=False),
                nn.SiLU(),
                nn.Conv2d(self.ffn_dim, self.hidden_dim, kernel_size=(1,1), bias=False)
            ) for _ in range(self.num_experts)
        ])

        # batched experts (as grouped conv2d), faster than individual experts as conv2d
        self.up_proj = nn.Conv2d(
            in_channels=self.hidden_dim * self.num_experts,
            out_channels=self.ffn_dim * self.num_experts,
            kernel_size=(1, 1),
            groups=self.num_experts,
            bias=False
        )
        self.act = nn.SiLU()
        self.down_proj = nn.Conv2d(
            in_channels=self.ffn_dim * self.num_experts,
            out_channels=self.hidden_dim * self.num_experts,
            kernel_size=(1, 1),
            groups=self.num_experts,
            bias=False
        )

        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise

    # no mask, for testing 
    def forward_nomask(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
                
        # Flatten hidden states for expert processing: [batch*seq, hidden, 1, 1]
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim).unsqueeze(-1).unsqueeze(-1)
        
        # Process through experts
        for expert_idx in range(self.num_experts):
                
            expert_weights = torch.where(
                topk_indices == expert_idx,
                topk_weights,
                torch.zeros_like(topk_weights)
            ).sum(dim=-1)
            
            expert_out = self.experts[expert_idx](hidden_states_flat).to(dtype=hidden_states.dtype)  # [tokens, hidden, 1, 1]
            expert_out = expert_out.squeeze(-1).squeeze(-1) * expert_weights.unsqueeze(-1)  # [tokens, hidden]
            expert_out = expert_out.to(dtype=final_hidden_states.dtype)
            
            final_hidden_states += expert_out

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    
    # fixed fixed # of tokens per expert. padding to limit. 
    def forward_fixed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        B = batch_size * sequence_length

        # Reshape for Conv2d
        hidden_states_reshaped = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # [B, hidden, seq, 1]
        router_logits = self.gate(hidden_states_reshaped).squeeze(-1).permute(0, 2, 1)  # [B, seq, experts]
        router_logits = router_logits.reshape(B, self.num_experts)

        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_indices.to(torch.int16)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Flatten routing info
        flat_expert_ids = topk_indices.reshape(-1)            # [B * top_k]
        flat_token_ids = torch.arange(B, device=hidden_states.device).repeat_interleave(self.top_k)  # [B * top_k], like [0,0,1,1,...]
        flat_weights = topk_weights.reshape(-1)

        # Sort by expert id
        sorted_expert_ids, sort_idx = flat_expert_ids.sort()
        sorted_token_ids = flat_token_ids[sort_idx]    # token ids arranged by expert ids
        sorted_weights = flat_weights[sort_idx]   # ditto 

        # Count how many tokens routed to each expert
        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        MAX_TOKENS = (B // self.num_experts) + 16  # tolerance padding

        # Build offset within each expert row
        #expert_offsets = torch.zeros_like(sorted_expert_ids)
        #offset = torch.zeros(self.num_experts, dtype=torch.long, device=hidden_states.device)
        cumsum_per_expert = torch.cumsum(
            torch.nn.functional.one_hot(sorted_expert_ids, num_classes=self.num_experts),
            dim=0
        )
        # Gather the count at each row's own expert index and subtract 1
        expert_offsets = cumsum_per_expert.gather(1, sorted_expert_ids.unsqueeze(1)).squeeze(1) - 1
        
        valid_mask = expert_offsets < MAX_TOKENS

        expert_row = sorted_expert_ids[valid_mask]   # [N] expert ids
        expert_col = expert_offsets[valid_mask]      # [N] slot per expert
        token_ids = sorted_token_ids[valid_mask]     # [N] global token ids
        token_weights = sorted_weights[valid_mask]   # [N]

        # Initialize expert input/output buffers
        #   below --- "expert_inputs" is routing table. row: expert  col: token ids
        expert_inputs = torch.zeros(self.num_experts, MAX_TOKENS, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        expert_weights = torch.zeros_like(expert_inputs[..., 0], dtype=hidden_states.dtype)

        expert_outputs = torch.zeros_like(expert_inputs, dtype=hidden_states.dtype)
        
        # Gather hidden states per expert
        hidden_states_flat = hidden_states.view(B, hidden_dim).unsqueeze(-1).unsqueeze(-1)
        expert_inputs = expert_inputs.unsqueeze(-1).unsqueeze(-1)
        # for whatever reason, the cast to float16 is needed for coreml conversion to work. 
        expert_inputs[expert_row, expert_col] = hidden_states_flat[token_ids].to(MODEL_DTYPE)
        expert_weights[expert_row, expert_col] = token_weights.to(MODEL_DTYPE)

        # Process each expert and accumulate outputs        
        '''
        for expert_id in range(self.num_experts):
            x = self.experts[expert_id](expert_inputs[expert_id])  # [MAX_TOKENS, hidden]
            x = x.squeeze(-1).squeeze(-1)  # [MAX_TOKENS, hidden]
            expert_outputs[expert_id] = x * expert_weights[expert_id].unsqueeze(-1)  # [MAX_TOKENS, hidden]
            # trandlated to MIL slice_update which is unsupported by ANE? 
        '''

        expert_out_list = []     # this is ok, as trace will capture individual tensors
        for expert_id in range(self.num_experts):
            x = self.experts[expert_id](expert_inputs[expert_id])  # [MAX_TOKENS, hidden, 1, 1]
            x = x.squeeze(-1).squeeze(-1)                           # [MAX_TOKENS, hidden]
            expert_out_list.append(x)
        expert_out_tensor = torch.stack(expert_out_list, dim=0)
        expert_outputs = expert_out_tensor * expert_weights.unsqueeze(-1)


        # Scatter expert_outputs back to token space
        final_hidden_states = torch.zeros(B, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        scatter_values = expert_outputs[expert_row, expert_col]  # [N, hidden]
        # scatter_values = scatter_values.to(dtype=final_hidden_states.dtype)  # not helpful?

        # final_hidden_states.index_add_(dim=0, index=token_ids, source=scatter_values)
        # CoreML-compatible (using one_hot + matmul)
        one_hot_mask = torch.nn.functional.one_hot(token_ids, num_classes=final_hidden_states.size(0)).to(MODEL_DTYPE)  # [N, B]
        contributions = torch.matmul(one_hot_mask.T, scatter_values)   # this can be bad for ANE
        final_hidden_states += contributions
        
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)

    # get rid of mask, dynamic shapes 
    # TODO: use int16 for token id throughout, save lots of cast
    def forward_fixed1(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        B = batch_size * sequence_length

        # Reshape for Conv2d
        hidden_states_reshaped = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # [B, hidden, seq, 1]
        router_logits = self.gate(hidden_states_reshaped).squeeze(-1).permute(0, 2, 1)  # [B, seq, experts]
        router_logits = router_logits.reshape(B, self.num_experts)

        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # topk_indices.to(torch.int16)   # wont help; cause additional cast 

        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Flatten routing info
        flat_expert_ids = topk_indices.reshape(-1)            # [B * top_k]
        flat_token_ids = torch.arange(B, device=hidden_states.device).repeat_interleave(self.top_k)  # [B * top_k], like [0,0,1,1,...]
        flat_weights = topk_weights.reshape(-1)

        # Sort by expert id
        sorted_expert_ids, sort_idx = flat_expert_ids.sort()
        sorted_token_ids = flat_token_ids[sort_idx]    # token ids arranged by expert ids
        sorted_weights = flat_weights[sort_idx]   # ditto 

        # Count how many tokens routed to each expert
        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        MAX_TOKENS = (B // self.num_experts) + 16  # tolerance padding

        # Build offset within each expert row
        #expert_offsets = torch.zeros_like(sorted_expert_ids)
        #offset = torch.zeros(self.num_experts, dtype=torch.long, device=hidden_states.device)
        cumsum_per_expert = torch.cumsum(
            torch.nn.functional.one_hot(sorted_expert_ids, num_classes=self.num_experts),
            dim=0
        )
        # Gather the count at each row's own expert index and subtract 1
        expert_offsets = cumsum_per_expert.gather(1, sorted_expert_ids.unsqueeze(1)).squeeze(1) - 1
                
        expert_row = sorted_expert_ids
        expert_col = expert_offsets
        token_ids = sorted_token_ids
        token_weights = sorted_weights

        # to be used later (cast from int to fp, cpu only)
        one_hot_maskT = torch.nn.functional.one_hot(token_ids, num_classes=B).T.to(MODEL_DTYPE)  # [N, B]
        # one_hot_maskT[0][0] += 0.00000000001  # Minimal update to trigger any deferred operations without side effects
        one_hot_maskT += 0.0000001 
        # force cast to happen here? instead of deferred. maybe a dummy update to one_hot_maskT?

        # Initialize expert input/output buffers
        #   below --- "expert_inputs" is routing table. row: expert  col: token ids
        expert_inputs_full = torch.zeros(self.num_experts, B, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1).unsqueeze(-1)
        expert_weights_full = torch.zeros(self.num_experts, B, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Gather hidden states per expert
        hidden_states_flat = hidden_states.view(B, hidden_dim).unsqueeze(-1).unsqueeze(-1) # unsequze to appease conv2d
        # for whatever reason, the cast to float16 is needed for coreml conversion to work. 
        # below: invokes scatter_nd, unsupported by ANE. on cpu. can be expensive b/c of copy?
        expert_inputs_full[expert_row, expert_col] = hidden_states_flat[token_ids].to(MODEL_DTYPE) # does this actually copy? bad
        expert_weights_full[expert_row, expert_col] = token_weights.to(MODEL_DTYPE)

        # truncate to MAX_TOKENS
        expert_inputs = expert_inputs_full[:, :MAX_TOKENS, :]
        expert_weights = expert_weights_full[:, :MAX_TOKENS]

        # expert_outputs = torch.zeros_like(expert_inputs, dtype=hidden_states.dtype) # not useful?

        # Process each expert and accumulate outputs        
        expert_out_list = []     # this is ok, as trace will capture individual tensors
        for expert_id in range(self.num_experts):
            x = self.experts[expert_id](expert_inputs[expert_id])  # [MAX_TOKENS, hidden, 1, 1]
            x = x.squeeze(-1).squeeze(-1)                           # [MAX_TOKENS, hidden]

            # zero pad x from [MAX_TOKENS, hidden] to [B, hidden]
            pad_len = B - MAX_TOKENS
            padded = torch.cat([x, torch.zeros(pad_len, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)], dim=0)
            expert_out_list.append(padded)
        expert_out_tensor = torch.stack(expert_out_list, dim=0)
        expert_outputs = expert_out_tensor * expert_weights_full.unsqueeze(-1) # shape: [num_experts, B, hidden]

        # Scatter expert_outputs back to token space (safe b/c we padded expert_out_tensor)
        final_hidden_states = torch.zeros(B, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        # below translates to gather_nd, which must cast to uses int16 indices??
        scatter_values = expert_outputs[expert_row, expert_col]  # [B*topk, hidden]  flatten to a seq of hidden state

        # below, sum up to per token hidden state. index_add_  unimpelmented by coreml
        # final_hidden_states.index_add_(dim=0, index=token_ids, source=scatter_values)
        
        # sparse accumulation...
        # one_hot: cpu only; cast (int to fp): cpu only
        # one_hot_mask = torch.nn.functional.one_hot(token_ids, num_classes=B).to(MODEL_DTYPE)  # [N, B]
        contributions = torch.matmul(one_hot_maskT, scatter_values)   # supported on ANE??
        final_hidden_states += contributions

        ''' 
        # naive impl of index_add_. lots of small, cpu-only ops
        result = torch.zeros((B, hidden_dim), dtype=scatter_values.dtype, device=scatter_values.device)
        # Then scatter and sum:
        for idx, val in zip(token_ids, scatter_values):
            result[idx] += val      # internal tensor update, bad 
        final_hidden_states = result
        '''

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)
    
    # static shapes, grouped mm
    # TODO: use int16 for token id throughout, save lots of cast
    def forward_fixed2(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        B = batch_size * sequence_length

        # Reshape for Conv2d
        hidden_states_reshaped = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # [B, hidden, seq, 1]
        router_logits = self.gate(hidden_states_reshaped).squeeze(-1).permute(0, 2, 1)  # [B, seq, experts]
        router_logits = router_logits.reshape(B, self.num_experts)

        routing_weights = torch.softmax(router_logits, dim=-1)

        ############################
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        '''
        # attempt to replace topk (cpu only)-- not very useful. topk is fast anyway
        topk_weights = [] 
        topk_indices = []
        routing_weights_copy = routing_weights.clone()
        for _ in range(self.top_k):
            max_values, max_indices = routing_weights_copy.max(dim=-1)   # extract max values and indices..
            topk_weights.append(max_values)
            topk_indices.append(max_indices)

            # Mask out max by directly using gather-based broadcasting
            # This avoids one_hot by using range comparison
            mask = (torch.arange(routing_weights_copy.size(1), device=routing_weights.device)
                    .unsqueeze(0) == max_indices.unsqueeze(1))  # shape [B, num_experts]
            routing_weights_copy = routing_weights_copy.masked_fill(mask, -float('inf'))

        topk_weights = torch.stack(topk_weights, dim=-1)   # [B, top_k]
        topk_indices = torch.stack(topk_indices, dim=-1)   # [B, top_k]
        # topk_indices.to(torch.int16)   # wont help; cause additional cast 
        '''
        ############################

        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Flatten routing info
        flat_expert_ids = topk_indices.reshape(-1)            # [B * top_k]
        flat_token_ids = torch.arange(B, device=hidden_states.device).repeat_interleave(self.top_k)  # [B * top_k], like [0,0,1,1,...]
        flat_weights = topk_weights.reshape(-1)

        # Sort by expert id
        sorted_expert_ids, sort_idx = flat_expert_ids.sort()
        sorted_token_ids = flat_token_ids[sort_idx]    # token ids arranged by expert ids
        sorted_weights = flat_weights[sort_idx]   # ditto 

        # Count how many tokens routed to each expert
        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        MAX_TOKENS = (B // self.num_experts) + 16  # tolerance padding

        # Build offset within each expert row
        #expert_offsets = torch.zeros_like(sorted_expert_ids)
        #offset = torch.zeros(self.num_experts, dtype=torch.long, device=hidden_states.device)
        cumsum_per_expert = torch.cumsum(
            torch.nn.functional.one_hot(sorted_expert_ids, num_classes=self.num_experts),
            dim=0
        )
        # Gather the count at each row's own expert index and subtract 1
        expert_offsets = cumsum_per_expert.gather(1, sorted_expert_ids.unsqueeze(1)).squeeze(1) - 1
                
        expert_row = sorted_expert_ids
        expert_col = expert_offsets
        token_ids = sorted_token_ids
        token_weights = sorted_weights

        # to be used later (cast from int to fp, cpu only)
        one_hot_maskT = torch.nn.functional.one_hot(token_ids, num_classes=B).T.to(MODEL_DTYPE)  # [N, B]
        # one_hot_maskT[0][0] += 0.00000000001  # Minimal update to trigger any deferred operations without side effects
        one_hot_maskT += 0.0000001 
        # since we run on cpu at this time, force cast to happen here rather than later 

        # Initialize expert input/output buffers
        #   below --- "expert_inputs" is routing table. row: expert  col: token ids
        expert_inputs_full = torch.zeros(self.num_experts, B, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype) # .unsqueeze(-1).unsqueeze(-1)
        expert_weights_full = torch.zeros(self.num_experts, B, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Gather hidden states per expert
        hidden_states_flat = hidden_states.view(B, hidden_dim)
        # for whatever reason, the cast to float16 is needed for coreml conversion to work. 
        # below: invokes scatter_nd, unsupported by ANE. on cpu. can be expensive b/c of copy?
        expert_inputs_full[expert_row, expert_col] = hidden_states_flat[token_ids].to(MODEL_DTYPE) # does this actually copy? bad
        expert_weights_full[expert_row, expert_col] = token_weights.to(MODEL_DTYPE)

        # truncate to MAX_TOKENS
        expert_inputs = expert_inputs_full[:, :MAX_TOKENS, :]   # shape [#expert,MAX_TOKENS,D]
        expert_weights = expert_weights_full[:, :MAX_TOKENS]
        
        #### Grouped conv processing for all experts
        x = expert_inputs.reshape(-1, hidden_dim).unsqueeze(-1).unsqueeze(-1)  # [E*T(MAX_TOKENS), D, 1, 1]
        # XXX verify correctness below
        x = expert_inputs.reshape(1, self.num_experts * hidden_dim, MAX_TOKENS, 1)  # [1, input_chan=E*D, T, 1]
        x = self.up_proj(x)                     # [1, E*ffn_dim, T, 1]
        x = self.act(x)
        x = self.down_proj(x)                   # [1, E*D, T, 1]
        x = x.view(self.num_experts, hidden_dim, MAX_TOKENS).permute(0, 2, 1)  # [E, T, D]

        # Pad outputs to [E, B, D]
        pad_len = B - MAX_TOKENS
        pad = torch.zeros(self.num_experts, pad_len, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        expert_outputs = torch.cat([x, pad], dim=1)

        expert_outputs = expert_outputs * expert_weights_full.unsqueeze(-1) # shape: [num_experts, B, hidden]

        # Scatter expert_outputs back to token space (safe b/c we padded expert_out_tensor)
        final_hidden_states = torch.zeros(B, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        # below translates to gather_nd, which must cast to uses int16 indices??
        scatter_values = expert_outputs[expert_row, expert_col]  # [B*topk, hidden]  flatten to a seq of hidden state

        # below, sum up to per token hidden state. index_add_  unimpelmented by coreml
        # final_hidden_states.index_add_(dim=0, index=token_ids, source=scatter_values)
        
        # sparse accumulation...
        # one_hot: cpu only; cast (int to fp): cpu only
        # one_hot_mask = torch.nn.functional.one_hot(token_ids, num_classes=B).to(MODEL_DTYPE)  # [N, B]
        contributions = torch.matmul(one_hot_maskT, scatter_values)   # supported on ANE??
        final_hidden_states += contributions

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        
    # forward 1 token only, based on forward_fixed2. very "token-centric" (instead of expert-centric)
    # hidden_states shape [bs=1,len=1, hidden_dim]
    def forward_single(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        B = batch_size * sequence_length
        assert B == 1

        # Reshape for Conv2d
        hidden_states_reshaped = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # [1, hidden, 1, 1]
        router_logits = self.gate(hidden_states_reshaped).squeeze(-1).squeeze(-1)  # [1, num_experts]

        routing_weights = torch.softmax(router_logits, dim=-1)   # [1, num_experts]

        ##### topk #########
        # topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1) # [1, top_k]

        # below, max() twice to avoid topk(). but still cpu only (e.g. reduce_argmax with int32 unsupported on ANE
        topk_weights = []  # a list of tensors
        topk_indices = []
        routing_weights_copy = routing_weights[0].clone()  # tensor, shape: [num_experts]

        for _ in range(self.top_k):
            max_value, max_index = routing_weights_copy.max(dim=-1)   # extract max values and indices..
            max_index.to(torch.int16)
            topk_weights.append(max_value)
            topk_indices.append(max_index)
            # Mask out max by directly using gather-based broadcasting
            routing_weights_copy[max_index] = -float('inf')
        topk_weights = torch.stack(topk_weights)   # [1, top_k]
        topk_indices = torch.stack(topk_indices)   # [1, top_k]
        # breakpoint()
        ############# end of topk ##########

        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) # [1, top_k]

        # Extract top-2 expert indices and weights (flatten) --> below will lower to gather??
        # expert_ids = topk_indices[0]      # [top_k] = [e1, e2]
        # weights = topk_weights[0]         # [top_k] = [w1, w2]

        expert_ids = topk_indices
        weights = topk_weights

        # Apply experts in a loop
        x = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # [1, hidden, 1, 1]
        combined = 0.0
        for i in range(self.top_k):
            out = self.experts[expert_ids[i]](x).squeeze(-1).squeeze(-1)  # [1, hidden]
            combined += weights[i] * out
        return combined.view(batch_size, sequence_length, hidden_dim)

    # slice expert weights from a big tensor weight. expert index dynamic
    #   not successfully ... cf comments inline
    # forward 1 token only, based on forward_fixed2. very "token-centric" (instead of expert-centric)
    # hidden_states shape [bs=1,len=1, hidden_dim]
    def forward_single_slice(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        B = batch_size * sequence_length
        assert B == 1

        # Reshape for Conv2d
        hidden_states_reshaped = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # [1, hidden, 1, 1]
        router_logits = self.gate(hidden_states_reshaped).squeeze(-1).squeeze(-1)  # [1, num_experts]

        routing_weights = torch.softmax(router_logits, dim=-1)   # [1, num_experts]

        ##### topk #########
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1) # [1, top_k]
        # Extract top-2 expert indices and weights (flatten) --> below will lower to gather??
        expert_ids = topk_indices[0]      # [top_k] = [e1, e2]
        weights = topk_weights[0]         # [top_k] = [w1, w2]

        # below, max() twice to avoid topk(). but still cpu only (e.g. reduce_argmax with int32 unsupported on ANE
        '''
        topk_weights = []  # a list of tensors
        topk_indices = []
        routing_weights_copy = routing_weights[0].clone()  # tensor, shape: [num_experts]

        for _ in range(self.top_k):
            max_value, max_index = routing_weights_copy.max(dim=-1)   # extract max values and indices..
            max_index.to(torch.int16)
            topk_weights.append(max_value)
            topk_indices.append(max_index)
            # Mask out max by directly using gather-based broadcasting
            routing_weights_copy[max_index] = -float('inf')
        topk_weights = torch.stack(topk_weights)   # [1, top_k]
        topk_indices = torch.stack(topk_indices)   # [1, top_k]
        # breakpoint()
        expert_ids = topk_indices
        weights = topk_weights
        '''
        ############# end of topk ##########
        
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) # [1, top_k]

        # experts. 
        # Slice out the expert weights from batched weights, no copy weights
        x = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # [1, hidden, 1, 1]
        combined = 0.0
        for i in range(self.top_k):
            expert_id = expert_ids[i]
            # Directly slice weight views — no clone needed
            # NB: batched conv weight tensor shape: [num_exp*output_dim, input_dim, 1,1]
            up_w = self.up_proj.weight[
                expert_id * self.ffn_dim : (expert_id + 1) * self.ffn_dim
            ]
            down_w = self.down_proj.weight[
                expert_id * hidden_dim : (expert_id + 1) * hidden_dim
            ]
            # BELOW confuses coreml: "ValueError: C_in / groups = 512/1"
            #   although pytorch tracing is good. maybe coreml expects weights from nn.module?? (not F)
            #   a deeper reason: dynamic slicing from static weights poorly supported??? 
            hidden = F.conv2d(x, up_w)
            hidden = self.act(hidden)
            hidden = F.conv2d(hidden, down_w)
            out = hidden.squeeze(-1).squeeze(-1)
            combined += weights[i] * out
        return combined.view(batch_size, sequence_length, hidden_dim)
    
    # send the token through all experts ...
    def forward_single_all(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        B = batch_size * sequence_length
        assert B == 1

        # Reshape for Conv2d
        hidden_states_reshaped = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # [1, hidden, 1, 1]
        router_logits = self.gate(hidden_states_reshaped).squeeze(-1).squeeze(-1)  # [1, num_experts]

        routing_weights = torch.softmax(router_logits, dim=-1)   # [1, num_experts]

        ##### topk #########
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1) # [1, top_k]
        # Extract top-2 expert indices and weights (flatten) --> below will lower to gather??
        expert_ids = topk_indices[0]      # [top_k] = [e1, e2]
        weights = topk_weights[0]         # [top_k] = [w1, w2]        
        ############# end of topk ##########

        weights = weights / weights.sum(dim=-1, keepdim=True) # [1, top_k]
        weights.to(hidden_states.dtype)  # force cast to appease coreml

        # (2) Build full weights mask: [1, num_experts], unselected experts -> 0 weight 
        weights_full = torch.zeros(1, self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype)
        weights_full[0, expert_ids] = weights  # scatter XXX coreml captured as constant???

        # weights_full = routing_weights   #debugging, directly use softmax weights - not changing anything?

        # (3) Expand weights over channels: [1, hidden_dim * num_experts, 1, 1]
        # XXX weights_mask is captured as constant???
        weights_mask = weights_full.repeat_interleave(self.hidden_dim, dim=1)  # shape [1, hidden_dim * num_experts]
        
        x = hidden_states.permute(0, 2, 1).unsqueeze(-1)  # [1, hidden, 1, 1]

        # (4) Expand input over all experts: [1, hidden_dim * num_experts, 1, 1]
        x_repeated = x.repeat(1, self.num_experts, 1, 1)

        # (6) Run through grouped MLP
        hidden = self.up_proj(x_repeated)
        hidden = self.act(hidden)
        hidden = self.down_proj(hidden)  # [1, hidden_dim * num_experts, 1, 1]

        hidden = hidden.squeeze(-1).squeeze(-1)  # [1, hidden_dim * num_experts]

        # (5) Multiply: only top-k expert paths are active, weighted
        hidden = hidden * weights_mask  # [1, hidden_dim * num_experts]

        # Reshape to [1, num_experts, hidden_dim]
        hidden = hidden.view(1, self.num_experts, self.hidden_dim)

        # (7) Reduce: sum across expert outputs
        # sum across expert dimension
        combined = hidden.sum(dim=1)  # [1, hidden_dim]
        return combined.view(batch_size, sequence_length, self.hidden_dim)
        
    # orig 
    def forward0(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
            # expert_weights shape: [num_routed_tokens]

            expert_out = self.experts[expert_idx](hidden_states_flat[mask]).to(dtype=hidden_states.dtype)  # [tokens, hidden, 1, 1]
            expert_out = expert_out.squeeze(-1).squeeze(-1) * expert_weights.unsqueeze(-1)  # [tokens, hidden]
            expert_out = expert_out.to(dtype=final_hidden_states.dtype)
            
            final_hidden_states[mask] += expert_out
        
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    # still 100%cpu, see comments below 
    def forward_bucketed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        total_tokens = batch_size * sequence_length
        
        # --- Routing Logic (unchanged) ---
        # Reshape for Conv2d: [batch, hidden, seq, 1]
        hidden_states_reshaped = hidden_states.permute(0, 2, 1).unsqueeze(-1)
        
        # Get router logits
        router_logits = self.gate(hidden_states_reshaped).squeeze(-1).permute(0, 2, 1)  # [batch, seq, experts]
        router_logits = router_logits.reshape(-1, self.num_experts)  # [batch*seq, experts]
        
        # Top-k routing
        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # --- Bucketing Setup ---
        # Define power-of-two bucket sizes (adjust based on your typical distribution)
        bucket_sizes = [32, 64, 128, 256, 512]  # Example sizes - tune for your use case
        min_bucket = bucket_sizes[0]
        
        # Initialize output tensor
        final_hidden_states = torch.zeros(
            total_tokens,
            hidden_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # --- Expert Processing with Bucketing ---
        # STILL TOO MUCH DYNAMISIM... e.g. gatther weights, pads; also compiler may not figure out there's a finite set of input shapes (bucket sizes
        for expert_idx in range(self.num_experts):
            # Create mask for this expert
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            num_active = expert_mask.sum().item()
            
            # if num_active == 0:
            #     continue  # Skip unused experts
                
            # --- Bucket Selection ---
            # Find smallest bucket that fits (or largest if all are too small)
            bucket_size = next((s for s in bucket_sizes if s >= num_active), bucket_sizes[-1])
            
            # --- Prepare Bucketed Input ---
            # Get indices of active tokens
            active_indices = torch.where(expert_mask)[0]  # [num_active]
            
            # Pad indices to bucket size (repeating last element)
            padded_indices = torch.cat([
                active_indices,
                active_indices[-1].repeat(bucket_size - num_active)
            ])  # [bucket_size]
            
            # Get weights for this expert (sum across top-k positions)
            expert_weights = torch.where(
                topk_indices == expert_idx,
                topk_weights,
                torch.zeros_like(topk_weights)
            ).sum(dim=-1)  # [total_tokens]
            
            # Pad weights (with zeros for padding elements)
            padded_weights = torch.cat([
                expert_weights[active_indices],
                torch.zeros(bucket_size - num_active, device=expert_weights.device).to(dtype=MODEL_DTYPE)
            ])  # [bucket_size]
            
            # --- Process Bucket ---
            # Gather inputs (shape: [bucket_size, hidden_dim, 1, 1])
            bucketed_input = hidden_states.reshape(-1, hidden_dim)[padded_indices]
            bucketed_input = bucketed_input.unsqueeze(-1).unsqueeze(-1)
            
            # Process through expert (shape: [bucket_size, hidden_dim, 1, 1])
            expert_output = self.experts[expert_idx](bucketed_input)
            expert_output = expert_output.squeeze(-1).squeeze(-1)  # [bucket_size, hidden_dim]
            
            # Apply weights
            weighted_output = expert_output * padded_weights.unsqueeze(-1)  # [bucket_size, hidden_dim]
            weighted_output = weighted_output.to(dtype=MODEL_DTYPE)
            
            # --- Scatter Results ---
            # Only scatter the actual active tokens (not padding)
            final_hidden_states[active_indices] += weighted_output[:num_active]
        
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # return self.forward0(hidden_states)
        # return self.forward_nomask(hidden_states)
        # return self.forward_fixed1(hidden_states)
        # return self.forward_fixed2(hidden_states)
        # return self.forward_single(hidden_states)
        return self.forward_single_all(hidden_states)
        # return self.forward_single_slice(hidden_states)
        # return self.forward_bucketed(hidden_states)

# Conversion to CoreML
def convert_to_coreml(model, sample_input, output_path):
    if MODEL_DTYPE == torch.float16:
        COREML_PRECISION = ct.precision.FLOAT16
        NP_DTYPE = np.float16
    else:
        COREML_PRECISION = ct.precision.FLOAT32
        NP_DTYPE = np.float32

    traced_model = torch.jit.trace(model, sample_input)

    torch_output_path = "/tmp/traced_model.txt"
    with open(torch_output_path, "w") as f:
        # f.write(str(traced_model.graph))
        f.write(str(traced_model.inlined_graph))
    print(f"Saving torch IR program to {torch_output_path}")

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=sample_input.shape, dtype=NP_DTYPE)],
        outputs=[ct.TensorType(name="output", dtype=NP_DTYPE)],
        compute_precision=COREML_PRECISION,
        #compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
        debug=True,
    )
    print("Generated MIL program:")
    # print(mlmodel._mil_program)
    mil_program_path = "/tmp/mil_program.txt"
    with open(mil_program_path, "w") as f:
        f.write(str(mlmodel._mil_program))
    print(f"MIL program written to: {mil_program_path}")

    mlmodel.save(output_path)
    return mlmodel

# seqlen = 64    # test prefill
seqlen = 1    # test decode

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
    model.to(MODEL_DTYPE)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dtype_str = "fp16" if MODEL_DTYPE == torch.float16 else "fp32"
    MODEL_OUTPUT_PATH = f"/tmp/sparse-moeblock_{dtype_str}_{current_time}.mlpackage"

    sample_input = torch.randn(1, seqlen, config.hidden_size).to(MODEL_DTYPE)
    coreml_model = convert_to_coreml(model, sample_input, MODEL_OUTPUT_PATH)
    
    with torch.no_grad():
        torch_output = model(sample_input)

    coreml_output = coreml_model.predict({"input": sample_input.cpu().numpy()})["output"]

    print("PyTorch output shape:", torch_output.shape)
    print("CoreML output shape:", coreml_output.shape)
    print("Max difference:", np.max(np.abs(torch_output.cpu().numpy() - coreml_output)))

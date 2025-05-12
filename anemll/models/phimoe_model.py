#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

# 4/22/25, fxl, based on llama_model.py
#  also: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phimoe/modeling_phimoe.py

from ..ane_converter.base_converter import BaseConverter
import coremltools as ct
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import math
import os
import safetensors
import gc
from tqdm import tqdm
import json
from .base_model import BaseModel
import torch.nn.init as init

from .modeling_rope_utils import rope_config_validation

# Model configuration constants
MODEL_DTYPE = torch.float16  # Hardcoded to float16 for ANE support
MLP_UP_SPLIT = 1   # Number of splits for MLP up-projection
MLP_DOWN_SPLIT = 1 # Number of splits for MLP down-projection
ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
    "swish": F.silu,
}
#TEST_DEVICE = "mps" #if torch.backends.mps.is_available() else "cpu"  # Device for testing
TEST_DEVICE = "cpu" 
# Context and state length configuration
CONTEXT_LENGTH = 512  # Default context window size for testing
STATE_LENGTH = 512   # KV cache state length

# Cache configuration
FORCE_UNIFIED_CACHE = True  # Force using a single unified KV cache
ENABLE_UNIFIED_CACHE = True  # Enable unified KV cache by default       # fxl: means all transformer blocks share same kv cache (?) 

# LM head configuration
ENABLE_CONV2D = bool(1)      # Use Conv2d for LM head
ENABLE_VACAB_SPLIT = bool(1)  # Split vocab into 2 parts
ENABLE_VACAB_SPLIT8 = bool(1)  # Split vocab into 8 parts
ENABLE_LOGITS2 = bool(0)    # Return 2 logits arrays
ENABLE_COREML = bool(0)     # CoreML-specific returns

# Debug flags. NB some print messages will make pytorch emit ops like resolve_conj, failing coreml conversion
ENABLE_DEBUG =  bool(0)  # General debug info
ENABLE_DEBUG2 = bool(0)  # Detailed debug for single token generation
ENABLE_DEBUG3 = bool(0)  # Detailed debug for prefill mode
ENABLE_VALUES = bool(0)  # Print tensor values for debugging

# XXX TBD cf transformers: src/transformers/models/phimoe/configuration_phimoe.py
class PhimoeConfig:
    # fxl: here, construct "config" from kwargs. if from_json() is called, then only the following attributes are set
    #       (and the rest from json are ignored 
    def __init__(self, **kwargs):
        # self.architectures = kwargs.get("architectures", ["LlamaForCausalLM"])   # no in use?
        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.intermediate_size = kwargs.get("intermediate_size", 6400)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 4096*32)      # fxl: fo precomputed rope factors.
        self.model_type = kwargs.get("model_type", "phimoe")
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        # self.pretraining_tp = kwargs.get("pretraining_tp", 1)   # fxl what is this?
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-05)
        self.rope_theta = kwargs.get("rope_theta", 1e6)

        # fxl: MoE...
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 2)
        self.num_local_experts = kwargs.get("num_local_experts", 16)
        self.output_router_logits = kwargs.get("output_router_logits", False)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 0.001)
        self.router_jitter_noise = kwargs.get("router_jitter_noise", 0.01)
        self.input_jitter_noise = kwargs.get("input_jitter_noise", 0.0)

        # fxl: rope scaling ...
        self.rope_scaling = kwargs.get("rope_scaling", None)
        # if self.rope_scaling:
        #     self.rope_scaling["rope_type"] = self.rope_scaling.get("rope_type", "longrope")
        if isinstance(self.rope_scaling, dict):
            if "rope_type" not in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling.get("type", None)
            if "original_max_position_embeddings" in self.rope_scaling:
                self.original_max_position_embeddings = self.rope_scaling["original_max_position_embeddings"]
            rope_scaling_short_mscale = self.rope_scaling.get("short_mscale", None)
            rope_scaling_long_mscale = self.rope_scaling.get("long_mscale", None)
            if not isinstance(rope_scaling_short_mscale, (int, float)):
                raise ValueError(
                    f"`rope_scaling`'s short_mscale field must be a number, got {rope_scaling_short_mscale}"
                )
            if not isinstance(rope_scaling_long_mscale, (int, float)):
                raise ValueError(
                    f"`rope_scaling`'s long_mscale field must be a number, got {rope_scaling_long_mscale}"
                )        
        rope_config_validation(self)
        # breakpoint()

        # fxl: Whether the model's input and output word embeddings should be tied
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        # self.torch_required = kwargs.get("torch_dtype", "bfloat16")    # fxl: not in use?
        # self.transformers_version = kwargs.get("transformers_version", "4.40.0.dev0") # fxl: not in use?
        self.use_cache = kwargs.get("use_cache", True)
        self.vocab_size = kwargs.get("vocab_size", 32064)  # phi has 32k vocab, en only
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", STATE_LENGTH)

    @classmethod
    def from_json(cls, json_file):
        print(f"Loading config from {json_file}")
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in self.__dict__.items())

# fxl: this is a standard layernorm (double centering?, not actually RMSNorm (which does not substract mean
#         why? anyway, avoid using it 
'''
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm        
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        #self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):

        if ENABLE_DEBUG:
            print(f"LlamaRMSNorm forward - self.weight.dtype: {self.weight.dtype}")
        mean = hidden_states.mean(-1, keepdim=True)
        hidden_states = hidden_states - mean

        return F.layer_norm(hidden_states, self.weight.shape, self.weight, bias=None, eps=float(self.eps)).to(TEST_DEVICE).to(MODEL_DTYPE)
        #Convert to 4D tensor
        #hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], 1, hidden_states.shape[2])
        input_dtype = hidden_states.dtype
        #hidden_states = hidden_states.to(MODEL_DTYPE)
        mean = hidden_states.mean(-1, keepdim=True)
        hidden_states = hidden_states - mean
        variance = hidden_states.pow(2).mean(-1, keepdim=True)         
        #variance = torch.clamp(variance, max=5e2)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states=(self.weight * hidden_states).to(input_dtype)
        #hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[3])
        return hidden_states.to(MODEL_DTYPE)
'''

# fxl: not in use? XXX understand it better
'''
class NA_LayerNormANE(nn.Module):
    """ LayerNorm optimized for Apple Neural Engine (ANE) execution
    """
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        self.num_channels = hidden_size
        self.eps = eps
        self.max_clip = 3e2  # ANE tuned value!
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.weight.data = self.weight.data.to(MODEL_DTYPE)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(MODEL_DTYPE)
        hidden_states = torch.clamp(hidden_states, min=-1000, max=1000).to(MODEL_DTYPE)  # ANE tuned value!
        mean = hidden_states.mean(-1, keepdim=True)
        hidden_states = hidden_states - mean
        variance = (hidden_states * hidden_states).mean(-1, keepdim=True).to(MODEL_DTYPE)       
        variance = torch.clamp(variance, min=self.eps, max=self.max_clip).to(MODEL_DTYPE)  # ANE tuned value!
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(MODEL_DTYPE) 
        hidden_states = (self.weight * hidden_states).to(MODEL_DTYPE)
        return hidden_states
'''

# fxl: src/transformers/modeling_rope_utils.py

# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this dictionary to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
# ROPE_INIT_FUNCTIONS = {
    # "default": _compute_default_rope_parameters,
    # "linear": _compute_linear_scaling_rope_parameters,
    # "dynamic": _compute_dynamic_ntk_parameters,
    # "yarn": _compute_yarn_parameters,
    # "longrope": _compute_longrope_parameters,
    # "llama3": _compute_llama3_parameters,
# }

def _compute_longrope_parameters(
    config, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
    # No need to keep BC with longrope, unreleased when this new pattern was created.
    if len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` should be unset in `_compute_longrope_parameters`, got "
            f"{rope_kwargs}"
        )

    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    long_factor = config.rope_scaling["long_factor"]   # fxl: a 64dim vector, per-dim scaling
    short_factor = config.rope_scaling["short_factor"]  # fxl: a 64dim vector, per-dim scaling
    factor = config.rope_scaling.get("factor")   # fxl: the ratio between the original and new max position embeddings. config does not have it?
    attention_factor = config.rope_scaling.get("attention_factor")   # fxl: in fact we dont have this. seems for RoPE

    # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if hasattr(config, "original_max_position_embeddings"):
        original_max_position_embeddings = config.original_max_position_embeddings
        factor = config.max_position_embeddings / config.original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

    # Compute the inverse frequencies -- scaled based on the target sequence length
    if seq_len and seq_len > original_max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor


# fxl 4/21/25 understood 3/5
#       also, "inv_freq" is registerd and saved ni model. but will __init__ be excuted on coreml and produce cos_cached?
#      for torhc.jit.trace, __init__() is not executed; only forward() is traced. 
#       cf: LlamaRotaryEmbedding
#    cf: src/transformers/models/phimoe/modeling_phimoe.py   PhimoeRotaryEmbedding
class PhimoeRotaryEmbedding(nn.Module):
    """Rotary positional embeddings for LLaMA model."""
    
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings     # fxl: for precomputed rope factors? (although rope can handle unlimited seq len)
        self.base = config.rope_theta
                
        # fxl: here we only support "longrope", as in phimoe
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            self.short_mscale = config.rope_scaling.get("short_mscale")
            self.long_mscale = config.rope_scaling.get("long_mscale")
        else:
            self.rope_type = "default"
        # self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        assert(self.rope_type == "longrope"), f"Unsupported rope_type: {self.rope_type}. Only 'longrope' is supported."

        # Generate and cache the inverse frequency buffer
        mscale = None       # fxl: mscale - "attention scaling factor" ("magnitude scale"?, used to scale the magnitude. 
        if config.rope_scaling:
            mscale = (
                self.long_mscale
                if self.max_position_embeddings > config.rope_scaling["original_max_position_embeddings"]
                else self.short_mscale
            )
        #  fxl: 'creates a vector of frequencies used to compute the angular rotation at each position'
        inv_freq, attention_scaling = _compute_longrope_parameters(config, TEST_DEVICE, self.max_position_embeddings)
        # fxl: will use long_mscale or short_mscale .. which basically have same values??
        mscale = attention_scaling if mscale is None else mscale
        self.register_buffer("inv_freq", inv_freq)   # need this??

        # Cache cos and sin values for positions
        t = torch.arange(self.max_position_embeddings, device=TEST_DEVICE).type_as(self.inv_freq) # fxl: a 1D tensor of token positions
        freqs = torch.outer(t, self.inv_freq)
        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)  #  <--- same as above 

        # fxl: NB: inv_freq are unique freqienceis, which are half of dim. eg dim=128, #inv_freq=64. 
        #      then cos_cached/sin_cached will be of shape [1, max_pos, dim=128]
        #  there are duplicates in cos_cached, sin_cached. 
        #      but,  first half of cos_emb and sin_emb apply to even dimensions
        emb = torch.cat((freqs, freqs), dim=-1)

        # Shape: [1, max_pos, head_dim] - consistent for both single token and batched    fxl: good desc, why dim0 is 1??
        self.cos_cached = (emb.cos() * mscale).view(1, self.max_position_embeddings, self.dim)
        self.sin_cached = (emb.sin() * mscale).view(1, self.max_position_embeddings, self.dim)        

        if ENABLE_DEBUG2:
            print(f"\n[TRACE] RotaryEmbedding initialized:")
            print(f"  dim: {self.dim}")
            print(f"  max_position_embeddings: {self.max_position_embeddings}")
            print(f"  base: {self.base}")
            print(f"  inv_freq shape: {self.inv_freq.shape}")
            print(f"  cos_cached shape: {self.cos_cached.shape}")
            print(f"  sin_cached shape: {self.sin_cached.shape}")
            print(f"  cos_cached[0,0,:5]: {self.cos_cached[0,0,:5].tolist()}")
            print(f"  sin_cached[0,0,:5]: {self.sin_cached[0,0,:5].tolist()}")

    def forward(self, x, seq_len=None):
        # Simply return the pre-computed values, converting to the correct dtype    fxl:??? b/c rope is not trainable?
        return self.cos_cached.to(dtype=x.dtype), self.sin_cached.to(dtype=x.dtype)

    # fxl: uncelar where the input/outpu tensdor is (re)layouted in half-and-half layout
    # .. not in use?? so not checked 
    def rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings to input tensor x."""
        # Ensure tensor is contiguous and get dimensions
        x = x.contiguous()
        half_dim = x.shape[-1] // 2
        
        # Split x into two halves for rotation
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        
        # Ensure cos and sin have the right dimensions
        if cos.dim() == 4:  # Single token case [1, 1, 1, head_dim]
            cos = cos[..., :half_dim]
            sin = sin[..., :half_dim]
            if ENABLE_DEBUG2:
                print(f"      cos_sliced shape: {cos.shape}")
                print(f"      sin_sliced shape: {sin.shape}")
                print(f"      cos_sliced first 5 values: {cos.flatten()[:5].tolist()}")
                print(f"      sin_sliced first 5 values: {sin.flatten()[:5].tolist()}")
        else:  # Batched case [1, seq_len, head_dim]
            cos = cos.unsqueeze(1)[..., :half_dim]  # Add head dimension
            sin = sin.unsqueeze(1)[..., :half_dim]  # Add head dimension
            if ENABLE_DEBUG2:
                print(f"      cos_sliced shape: {cos.shape}")
                print(f"      sin_sliced shape: {sin.shape}")
                print(f"      cos_sliced first 5 values: {cos.flatten()[:5].tolist()}")
                print(f"      sin_sliced first 5 values: {sin.flatten()[:5].tolist()}")

        # Apply rotation using complex number multiplication        # fxl: so coreml can even trace this??? does it have own impl???
        rotated = torch.cat([
            x1 * cos - x2 * sin,  # Real part       # fxl: elem 0 in a pair
            x2 * cos + x1 * sin   # Imaginary part  # fxl: elem 1 in a pair
        ], dim=-1)
        # fxl: NB: 'It’s only okay if the model expects the rotated values in this half-and-half layout'

        if ENABLE_DEBUG2:
            print(f"      rotated shape: {rotated.shape}")
            print(f"      rotated first 5 values: {rotated.flatten()[:5].tolist()}")

        return rotated.to(MODEL_DTYPE)

######################################################
# below:  MoE. directly copied from transformers 

# orig design, see comments below; overkill? unfriendly to coreml
def sparsemixer(scores, jitter_eps, training, top_k=2):
    """
    Sparse mixer function to select top-k experts and compute multipliers.
    Based on the paper: https://arxiv.org/pdf/2409.12136
    We first replace the TopK(·) function as random sampling of discrete variables
    in model training. Then, following Liu et al. (2023a) and Liu et al. (2023b), we apply Heun's
    third order method to approximate the expert routing gradient and construct a modified
    back-propagation to give a mathematically sound gradient estimation for expert routing.

    Args:
        scores (torch.Tensor): Input scores tensor.
        jitter_eps (float): Jitter epsilon for numerical stability.
        training (bool): Flag indicating if the model is in training mode.
        top_k (int): Number of top experts to select.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Multiplier and selected experts tensors.

    fxl: 
    - The top-1 and top-2 experts are not selected from a single top-k operation
    - Instead, the second expert is selected by masking the first and repeating the top-1 selection logic

    pros: completely masking out very low-scoring experts --- avoid noise; also compute gradients on 
        sufficently ranked, but unselected experts (for training) 
    XXX can check ggml MoE code
    """
    if top_k != 2:
        raise ValueError("top_k must be equal to 2")

    # first expert

    with torch.no_grad():
        # Compute mask for sparsity
        #    fxl: determine which logits are too far from the max, thus set to -inf to enforce sparsity 
        #           before softmax 
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    # Apply mask  (fxl: set "unquialifed" experts to -inf. 0 prob in softmax 
    masked_gates = scores.masked_fill(mask_logits_threshold, float("-inf"))
    # if training:
    if False:   # fxl
        selected_experts = (
            (
                masked_gates
                - torch.empty_like(masked_gates, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )
            .max(dim=-1)[1]
            .unsqueeze(-1)
        )  # Gumbel sampling, more robust than the multinomial method
    else:
        selected_experts = max_ind  # fxl: top1 

    # Compute scores for gradients  fxl: softmax once; fwd: only get top1's "multiplier"
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

    # if training:
    if False:   # fxl
        # Compute midpoint mask
        max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
        mask_for_one = torch.logical_or(
            selected_experts == max_ind,
            torch.rand_like(max_scores) > 0.75,  # Heun's third-order method
        )
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

        multiplier = MultiplierProcessor.apply(
            scores,
            multiplier_o,
            selected_experts,
            masked_gates,
            mask_for_one,
        )
    else:
        multiplier = multiplier_o

    # Masked out first expert  (below computes 2nd expert)
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float("-inf"),
    )
    with torch.no_grad():
        # Compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    # Apply mask (fxl: keep 2nd & other "qualified" experts)
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))
    # if training:
    if False:   # fxl
        selected_experts_top2 = (
            (
                masked_gates_top2
                - torch.empty_like(masked_gates_top2, memory_format=torch.legacy_contiguous_format)
                .exponential_()
                .log()
            )
            .max(dim=-1)[1]
            .unsqueeze(-1)
        )  # Gumbel sampling, more robust than the multinomial method
    else:
        selected_experts_top2 = max_ind
    # Compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)

    # if training:
    if False:   # fxl
        # Compute midpoint mask
        max_scores, max_ind = masked_gates_top2.max(dim=-1, keepdim=True)
        mask_for_one_top2 = torch.logical_or(
            selected_experts_top2 == max_ind,
            torch.rand_like(max_scores).uniform_() > 0.75,  # Heun's third-order method
        )
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one_top2 = torch.add(0.3333, mask_for_one_top2, alpha=0.6667).type_as(masked_gates_top2)

        multiplier_top2 = MultiplierProcessor.apply(
            scores,
            multiplier_top2_o,
            selected_experts_top2,
            masked_gates_top2,
            mask_for_one_top2,
        )
    else:
        multiplier_top2 = multiplier_top2_o

    # fxl: combine the 2nd expert with the first one (are multipliers normalized? no?
    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)

    # fxl: "multiplier" (for top1,2) does not have to add to 1??
    return (
        multiplier,
        selected_experts,
    )

# v1, simplified. geet rid of training computation (jitter, mask, etc)
#    just simply topk, with softmax as multiplier
def sparsemixer_simple(scores, top_k=2):
    
    top2_vals, top2_inds = scores.topk(top_k, dim=-1)
    top2_weights = torch.softmax(top2_vals, dim=-1)
    return (top2_weights, top2_inds)

# Copied from transformers.models.mixtral.modeling_mixtral.MixtralBlockSparseTop2MLP with Mixtral->Phimoe
# fxl: one instance -- an expert?  and it's Gated MLP (?
class PhimoeBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: PhimoeConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        # orig, using Linear
        # self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False, dtype=MODEL_DTYPE)  # fxl: up proj
        # self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False, dtype=MODEL_DTYPE)  # fxl: down proj
        # self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False, dtype=MODEL_DTYPE)  # learned gate

        # use conv2d... for ANE
        self.w1 = nn.Conv2d(self.hidden_dim, self.ffn_dim, kernel_size=1, bias=False, dtype=MODEL_DTYPE)  # fxl: up proj
        self.w2 = nn.Conv2d(self.ffn_dim, self.hidden_dim, kernel_size=1, bias=False, dtype=MODEL_DTYPE)  # fxl: down proj
        self.w3 = nn.Conv2d(self.hidden_dim, self.ffn_dim, kernel_size=1, bias=False, dtype=MODEL_DTYPE)  # learned gate

        self.act_fn = ACT2FN[config.hidden_act]

    # def forward(self, hidden_states):
    #     # fxl: -1: inferred bs, hidden_dim, H=1, W=1 for conv2d kernel size 
    #     hidden_states = hidden_states.view(-1, self.hidden_dim, 1, 1) 
    #     # fxl: below "*" is elementwise gating
    #     current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
    #     current_hidden_states = self.w2(current_hidden_states)
    #     return current_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape     # shape [bs, hidden_dim]
        # fxl: -1: inferred bs, hidden_dim, H=1, W=1 for conv2d kernel size 
        x = hidden_states.view(-1, self.hidden_dim, 1, 1)  # shape [bs, hidden_dim, 1, 1]
        # x = hidden_states.view(1, self.hidden_dim, -1, 1) # shape [1, hidden_dim, bs, 1]  # will cause coreml runtime error... why? "Always keep bs as first dimension?"
        # fxl: below "*" is elementwise gating
        out = self.act_fn(self.w1(x)) * self.w3(x)
        out = self.w2(out)
        out = out.view(*original_shape)
        return out
    
class PhimoeSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    """ fxl notes on above: 
    - When some experts are overloaded and others are underloaded, GPU utilization drops
    - it only computes outputs for the tokens actually routed to each expert
    - block-sparse execution: chunks of tokens for each expert are handled independently, and only if needed
    (use loops, see below)    
    (2) above means each token is routed to all experts?
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        
        # gating         fxl: router. must cast to fp16
        # self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False, dtype=MODEL_DTYPE)
        self.gate = nn.Conv2d(self.hidden_dim, self.num_experts, kernel_size=1, bias=False, dtype=MODEL_DTYPE)

        self.experts = nn.ModuleList([PhimoeBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise


    # origin design. uses tensors as indices to get top2, and then do in-place update index_add_()
    # unfriendly to coreml
    def forward0(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise
            )
        # fxl: flatten input to something like (batch_size * seq_len, hidden_dim)
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(
            # fxl:to appease conv2d (gate)
            #hidden_states.view(batch_size * sequence_length, hidden_dim, 1, 1)
            # conv2d expects (bs, input_chan=hidden_dim, H=len, W=1)
            hidden_states.permute(0,2,1).unsqueeze(-1)
        )  # output shape (bs, output_chan=num_experts, H=len, W=1)
        # remove the last dim, and permuate back 
        router_logits = router_logits.squeeze(-1).permute(0, 2, 1)  
        # after this, router_logits has shape [batch, seq, experts]  
        # router_logits = router_logits.view(batch_size * sequence_length, self.num_experts)
        router_logits = router_logits.view(-1, self.num_experts)
        # now router_logits shape [batch*seq, experts]
        
        # routing_weights, selected_experts = sparsemixer(        
        #     router_logits,
        #     jitter_eps=self.router_jitter_noise,
        #     training=self.training,
        # )
        routing_weights, selected_experts = sparsemixer_simple(router_logits)

        '''
        selected_experts.shape = (bs=64,topk=2)
        '''

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # shape: (num_experts=16, topk=2, bs=64)  this is so called "one hot"

        # Loop over all available experts in the model and perform the computation on each expert
        #   fxl: allow each expert to have different numbers of tokens
        #   fxl: XXX loop below cannot be traced, need better way; also, may split wrt MLP_UP_SPLIT MLP_DOWN_SPLIT 
        #           cf LlamaMLP
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # fxl: return row indices and col indices in which elements of expert_mask[expert_idx] are True
            #   idx: 0 or 1 (like rank)? top_x: the index of the tokens that routed to THIS expert (e.g. 0,2,4..
            #    ^^^ feels the names are revsered...???

            # fxl: .. when no tokens routed to THIS expert...
            # remove branch for tracing. coreml should be fine even if top_x is empty?
            # if top_x.shape[0] == 0:
            #     continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # fxl:   below hidden_states[None] adds a dummy batch dim, shape becomes (1, bs=64, hidden_dim), 
            #           then select tokens with "top_x" as indices, 
            #           then reshape to (bs=64, hidden_dim) for the current expert    
            #   fxl:  NB things like "routing_weights[top_x, idx]" is using 1D tensors (top_x, idx) as indices
            #           understood 3/5
            #      routing_weights[top_x, idx, None] --> shape (num_routed_tokens, 1), scalar weight bdcast across hidden_dim
            # if top_x.shape[0] != 0:
            #     breakpoint()
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)  # fxl: extract hidden states for all tokens routed to the current expert
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # current_state shape: (num_routed_tokens, hidden_dim)
            # current_hidden_states shape: (num_routed_tokens, hidden_dim)

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            #  fxl: accmulate hte current experts output  ... to final_hidden_states of all tokens 
            #       "For each token index in top_x, add the corresponding row from current_hidden_states into final_hidden_states at that same index — in place"
            # final_hidden_states.index_add_(dim=0, index=top_x, source=current_hidden_states.to(hidden_states.dtype))

            # CoreML-compatible (using one_hot + matmul)
            one_hot_mask = torch.nn.functional.one_hot(top_x, num_classes=final_hidden_states.size(0)).half()
            contributions = torch.matmul(one_hot_mask.T, current_hidden_states)
            final_hidden_states += contributions

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # return final_hidden_states, router_logits   # we dont need router logits
        return final_hidden_states

    # simplified fwd
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape        
        router_logits = self.gate(
            # conv2d expects (bs, input_chan=hidden_dim, H=len, W=1)
            hidden_states.permute(0,2,1).unsqueeze(-1)
        )  # output shape (bs, output_chan=num_experts, H=len, W=1)
        # remove the last dim, and permuate back 
        router_logits = router_logits.squeeze(-1).permute(0, 2, 1)  
        # after this, router_logits has shape [batch, seq, experts]  
        router_logits = router_logits.view(-1, self.num_experts) # shape [batch*seq, experts]

        # Simplified top-k routing
        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # topk_weights shape: (bs=64*seq_len, topk=2), topk_indices shape: (bs=64*seq_len, topk=2)
        
        # Normalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # fxl: flatten input to something like (batch_size * seq_len, hidden_dim)
        hidden_states = hidden_states.view(-1, hidden_dim)

        final_hidden_states = torch.zeros(
            batch_size * sequence_length, 
            hidden_dim, 
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )

        expert_mask = torch.zeros(
            self.num_experts,
            batch_size * sequence_length,
            dtype=torch.bool,
            device=hidden_states.device
        )        
        
        for expert_idx in range(self.num_experts):
            # mask set to True for ANY topk indices that match expert_idx (scalar). so mask is 1D (not 2D)
            mask = (topk_indices == expert_idx).any(dim=-1)   
            expert_mask[expert_idx] = mask
            # mask shape: [bs], type bool

        # Process through experts
        for expert_idx in range(self.num_experts):
            mask = expert_mask[expert_idx]
                
            expert_weights = torch.where(
                topk_indices == expert_idx,   # condition
                topk_weights,    # "input" 
                torch.zeros_like(topk_weights)    # "other"
            ).sum(dim=-1)[mask]         # mask shape [bs], True/False
            # fxl: "weights" as in "attention weights".... "sum" will reduce zero values
            # expert_weights shape: [num_routed_tokens]
            
            expert_out = self.experts[expert_idx](hidden_states[mask]).to(dtype=hidden_states.dtype)  # [tokens, hidden]
            expert_out = expert_out * expert_weights.unsqueeze(-1)  # res: [tokens, hidden]
            expert_out = expert_out.to(dtype=final_hidden_states.dtype)
            
            final_hidden_states[mask] += expert_out
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

# cf: src/transformers/models/phimoe/modeling_phimoe.py     PhimoeAttention
class PhimoeAttention(nn.Module):
    """Attention mechanism optimized for Apple Neural Engine."""
    
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:   # fxl why need layer indx?
            print(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        # fxl: below not in use? 
        # self.rope_theta = config.rope_theta
        # self.is_causal = True
        # self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                           f" and `num_heads`: {self.num_heads}).")

        # below, projection using conv2d... w optional bias (=True
        self.q_proj = nn.Conv2d(self.hidden_size, self.num_heads * self.head_dim, kernel_size=1, bias=self.config.attention_bias, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.k_proj = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim, kernel_size=1, bias=self.config.attention_bias, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.v_proj = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim, kernel_size=1, bias=self.config.attention_bias, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.config.attention_bias, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        
        # fxl: why is this per layer? they are all the same, right? shoudlnt' they be shared???
        #       (transformers uses a single rotary embedding for the entire model
        #      XXX maybe change this. non trivial mem
        self.rotary_emb = PhimoeRotaryEmbedding(config)
        # self.scaling_factor = torch.tensor(1.0 / math.sqrt(self.head_dim), dtype=MODEL_DTYPE, device=TEST_DEVICE)    # phi does not have this

    # # pretty much reshape to appease ANE, proj QKV & apply emb to QK    (need debugging)
    #     returned KV ready to save to kvcache (but dose not operate on kvcache
    # fxl: hidden_states embedding for 1 token (? bs always 1) 
    def get_new_kv_cache(self, hidden_states, current_pos, rotary_emb):
        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device
        
        if ENABLE_DEBUG2:
            print(f"\nSINGLE TOKEN - Input shapes:")
            print(f"  hidden_states: {hidden_states.shape}, current_pos: {current_pos}, q_len: {q_len}")
        
        # Project QKV and ensure MODEL_DTYPE
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        if ENABLE_DEBUG2:
            print(f"  After permute+unsqueeze: {hidden_states.shape}")
        # Perform projections with fixed dimensions like in the original working code
        query_states = self.q_proj(hidden_states).view(1, self.num_heads, 1, self.head_dim).to(MODEL_DTYPE)
        key_states = self.k_proj(hidden_states).view(1, self.num_key_value_heads, 1, self.head_dim).to(MODEL_DTYPE)
        value_states = self.v_proj(hidden_states).view(1, self.num_key_value_heads, 1, self.head_dim).to(MODEL_DTYPE)
        
        if ENABLE_DEBUG2:
            print(f"  After projection:")
            print(f"    query_states: {query_states.shape}")
            print(f"    key_states: {key_states.shape}")
            print(f"    value_states: {value_states.shape}")
        
        # Use provided rotary embeddings
        cos, sin = rotary_emb
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if ENABLE_DEBUG2:
            print(f"  After applying rotary:")
            print(f"    query_states: {query_states.shape}")
            print(f"    key_states: {key_states.shape}")
            print(f"    value_states: {value_states.shape}")

        return query_states, key_states, value_states

    # cf above 
    # below: pretty much reshape to appease ANE, proj & apply emb 
    #       also cf: PhimoeAttention.forward()
    # "current_pos" not in use?? 
    def get_new_kv_cache_prefill(self, hidden_states, current_pos, rotary_emb, batch_size):
        """Get new key-value cache entries optimized for prefilling with batched tokens."""
        _, batch, _ = hidden_states.shape # [1, BATCH, hidden_size=2048]
        device = hidden_states.device
        
        # fxl: batch -- subseq length
        if ENABLE_DEBUG3:
            print(f"\nPREFILL - Input shapes:")
            print(f"  hidden_states: {hidden_states.shape}, current_pos: {current_pos}, batch: {batch}")
        
        # Project QKV and ensure MODEL_DTYPE - optimized for batch processing       this shape for ANE? XXX understand better 
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)  # [1, hidden_size, 1, batch]

        # Project all tokens at once using Conv2d
        query_states = self.q_proj(hidden_states)  # [1, num_heads * head_dim, 1, batch]
        key_states = self.k_proj(hidden_states)    # [1, num_kv_heads * head_dim, 1, batch]
        value_states = self.v_proj(hidden_states)  # [1, num_kv_heads * head_dim, 1, batch]

        # Reshape to final dimensions
        query_states = query_states.view(1, self.num_heads, self.head_dim, batch).permute(0, 1, 3, 2)  # [1, num_heads, batch, head_dim]
        key_states = key_states.view(1, self.num_key_value_heads, self.head_dim, batch).permute(0, 1, 3, 2)  # [1, num_kv_heads, batch, head_dim]
        value_states = value_states.view(1, self.num_key_value_heads, self.head_dim, batch).permute(0, 1, 3, 2)  # [1, num_kv_heads, batch, head_dim]

        # Get rotary embeddings for all positions at once       fxl: rotary_emb already cut into batch
        #  ((.... previous shape 1, bs, 1, headdim weird. 
        cos, sin = rotary_emb
        cos = cos.permute(0, 2, 1, 3)  # [1, 1, batch, head_dim]
        sin = sin.permute(0, 2, 1, 3)  # [1, 1, batch, head_dim]

        # Apply rotary embeddings to all positions at once
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        return query_states.to(MODEL_DTYPE), key_states.to(MODEL_DTYPE), value_states.to(MODEL_DTYPE)

    # fxl: below not in use??? (instead forward_regular, forward_prefill)
    '''
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,      # e.g. [0, 1, 2, 3, 4...]
        current_pos: int,
        IN_PREFILL: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with support for both single token and prefill modes."""
        batch_size, seq_length, _ = hidden_states.shape
        
        print(f"\n[DEBUG] Attention Forward:")
        print(f"  Input shapes:")
        print(f"    hidden_states: {hidden_states.shape}")
        print(f"    attention_mask: {attention_mask.shape if attention_mask is not None else None}")
        print(f"    position_ids: {position_ids.shape if position_ids is not None else None}")
        print(f"    current_pos: {current_pos}")
        print(f"    IN_PREFILL: {IN_PREFILL}")
        
        assert current_pos is not None, "current_pos must be provided"
        assert position_ids is not None, "position_ids must be provided"
        assert attention_mask is not None, "attention_mask must be provided"

        # Generate position IDs if not provided
        # Get rotary embeddings      return precomputed cos/sin values -- XXX check this also trace llama code, seq_len current_pos ignored? why ??
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_length, current_pos=current_pos)
        
        # Get KV states based on mode  (fxl: proj and apply rope..
        if IN_PREFILL:
            query_states, key_states, value_states = self.get_new_kv_cache_prefill(
                hidden_states, current_pos, (cos, sin), batch_size=seq_length
            )
        else:
            query_states, key_states, value_states = self.get_new_kv_cache(
                hidden_states, current_pos, (cos, sin)
            )
        
        # Update KV cache if provided
        if kwargs.get('kv_cache_layer', None) is not None:
            key_cache, value_cache = kwargs['kv_cache_layer']
            print(f"  KV Cache shapes:")
            print(f"    key_cache: {key_cache.shape}")
            print(f"    value_cache: {value_cache.shape}")
            print(f"    key_states: {key_states.shape}")
            print(f"    value_states: {value_states.shape}")
            print(f"    current_pos: {current_pos}")
            print(f"    seq_length: {seq_length}")
            
            if current_pos is not None:
                # Update at current position
                print(f"  Updating cache at position {current_pos}")
                print(f"    Update slice - key_cache[:, {current_pos}:{current_pos + seq_length}]")     # fxl: this <--- a subseq
                print(f"    key_states shape: {key_states.shape}")
                print(f"    value_states shape: {value_states.shape}")
                
                # Reshape key_states and value_states to match cache dimensions
                key_states_reshaped = key_states.squeeze(0)  # Remove batch dimension
                value_states_reshaped = value_states.squeeze(0)  # Remove batch dimension
                
                print(f"    After reshape:")
                print(f"      key_states_reshaped: {key_states_reshaped.shape}")
                print(f"      value_states_reshaped: {value_states_reshaped.shape}")
                
                # Update cache
                key_cache[:, current_pos:current_pos + seq_length] = key_states_reshaped
                value_cache[:, current_pos:current_pos + seq_length] = value_states_reshaped
                
                # Use full cache for attention
                key_states = key_cache[:, :current_pos + seq_length].unsqueeze(0)  # Add batch dimension back
                value_states = value_cache[:, :current_pos + seq_length].unsqueeze(0)  # Add batch dimension back
                
                print(f"  After cache update:")
                print(f"    key_states: {key_states.shape}")
                print(f"    value_states: {value_states.shape}")
        
        # Compute scaled dot-product attention
        attention_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling_factor
        
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        
        attention_weights = self.ANE_softmax(attention_weights, dim=-1)
        attention_output = torch.matmul(attention_weights, value_states)
        
        print(f"  Attention computation shapes:")
        print(f"    attention_weights: {attention_weights.shape}")
        print(f"    attention_output: {attention_output.shape}")
        
        # Reshape for output projection
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, self.hidden_size)
        attention_output = attention_output.permute(0, 2, 1).unsqueeze(2)
        
        # Project output using Conv2D
        output = self.o_proj(attention_output.squeeze(2).permute(0, 2, 1))
        print(f"  Final output shape: {output.shape}\n")
        
        return output
    '''

    # fxl: below quite stanard softmax?? maybe for ease of tracing?  "Optimized softmax for batch_size=1" but also used in prefil
    #   in use by forward_regular forward_prefill
    def ANE_softmax(self, x, dim=-1):
        #return F.softmax(x, dim=dim)
        
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        x = x - x_max
        exp_x = torch.exp(x)
        softmax_output = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
        return softmax_output
        
    # fxl: multiply q/k states with (precomputed) cos/sin values ...?
    # dindt check in detail, assume correct
    def apply_rotary_pos_emb(self, q_states, k_states, cos, sin):
        """
        Applies rotary position embeddings to both q_states and k_states.
        """
        if ENABLE_VALUES:
            print(f"\n[DEBUG] apply_rotary_pos_emb input:")
            print(f"  q_states shape: {q_states.shape}")
            print(f"  k_states shape: {k_states.shape}")
            print(f"  cos shape: {cos.shape}")
            print(f"  sin shape: {sin.shape}")
            print(f"  q_states first 5: {q_states[0,0,0,:5].tolist()}")
            print(f"  k_states first 5: {k_states[0,0,0,:5].tolist()}")
        
        def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            # fxl: cos shape [1,1,bs=64,head_dim=128]
            x = x.contiguous()
            half_dim = x.shape[-1] // 2
            
            x1 = x[..., :half_dim]    # fxl: 1st half of input embedding
            x2 = x[..., half_dim:]      # fxl: 2nd half of input embedding...
            
            if ENABLE_VALUES:
                print(f"\n[DEBUG] rotate:")
                print(f"  x1 first 5: {x1[0,0,0,:5].tolist()}")
                print(f"  x2 first 5: {x2[0,0,0,:5].tolist()}")
            
            if cos.dim() == 4:
                cos = cos[..., :half_dim]    # fxl: only take the first half ? (there's redudancy?
                sin = sin[..., :half_dim]
            else:
                cos = cos.unsqueeze(1)[..., :half_dim]
                sin = sin.unsqueeze(1)[..., :half_dim]
            
            # NB: each "pair" channel is "half_dim" apart; not adjacent. that seems ok
            rotated = torch.cat([
                x1 * cos - x2 * sin,
                x2 * cos + x1 * sin
            ], dim=-1)
            
            
            return rotated.to(MODEL_DTYPE)
        
        q_rotated = rotate(q_states, cos, sin)
        k_rotated = rotate(k_states, cos, sin)
                
        return q_rotated, k_rotated

    # fxl: 1 token gen. use kvcache. Q (query_states) already projected
    #   "kv_cache_layer" -- kvcache for this layer (??
    def forward_regular(self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None):
        """Forward pass for single token generation."""
        bsz, q_len, _ = hidden_states.shape
                
        # Get KV cache
        K_layer_cache, V_layer_cache = kv_cache_layer
        
        # Slice only the first CONTEXT_LENGTH from the cache (as cache is being filled incrementally
        #       XXX how would this slice be lowered to coreml IR? it might be compabiel with ANE, b/c start/end are static?
        K_layer_cache = K_layer_cache[..., :self.config.context_length, :]
        V_layer_cache = V_layer_cache[..., :self.config.context_length, :]
        
        # Repeat KV for multi-head attention
        #   (this, and transpose below, might require copy on ANE???? 
        key_states = self.repeat_kv(K_layer_cache, self.num_key_value_groups)
        value_states = self.repeat_kv(V_layer_cache, self.num_key_value_groups)

        # Compute attention using optimized path for batch_size=1
        
        # Compute attention scores directly without einsum    ( ... scaled dotproduct. ok to use matmul directly - not conv2d??
        # attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling_factor
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)    # cf modeling_phimoe.py forward()
        
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask[:, :, :self.config.context_length]
        
        if ENABLE_VALUES:
            print(f"[TRACE]  SINGLE attn_weights first 10={attn_weights[0, -1, 0:10, 0:10].tolist()}")
            print(f"[TRACE]  q_len={q_len}")

        # Optimized softmax for batch_size=1
        attn_weights = self.ANE_softmax(attn_weights, dim=-1)
        
        # Compute attention output directly without einsum  fxl: XXX means what?? 
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape before projecting: [1, heads, q_len, head_dim] -> [1, q_len, heads*head_dim]
        if ENABLE_DEBUG2:
            print(f"[TRACE]  B4 reshape attn_output shape={attn_output.shape}")
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Project output
        attn_output = self.o_proj(attn_output)
        
        return attn_output

    # fxl: prefill gen, use kvcache. Q (query_states) already projected.  
    #       similar to above, but use einsum for attn_weights and attn_output (sgemm... why??
    def forward_prefill(self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None):
        """Forward pass for prefill mode"""
        bsz, q_len, _ = hidden_states.shape
                
        # Get KV cache
        K_layer_cache, V_layer_cache = kv_cache_layer
        
        # Slice only up to CONTEXT_LENGTH from the cache
        K_layer_cache = K_layer_cache[..., :self.config.context_length, :]
        V_layer_cache = V_layer_cache[..., :self.config.context_length, :]
        
        if ENABLE_DEBUG3:  
            print("[forward_prefill.0] K_layer_cache.shape=", K_layer_cache.shape)
        # fxl: eg [forward_prefill.0] K_layer_cache.shape= torch.Size([num_kv_heads=8, len=512, head_dim=128])

        # Repeat KV for multi-head attention
        key_states = self.repeat_kv(K_layer_cache, self.num_key_value_groups)
        value_states = self.repeat_kv(V_layer_cache, self.num_key_value_groups)
        
        if ENABLE_DEBUG3:  
            print("[forward_prefill.1] hidden_states.shape=", hidden_states.shape)
            print("[forward_prefill.1] query_states.shape=", query_states.shape)
            print("[forward_prefill.1] key_states.shape=", key_states.shape)
            print("[forward_prefill.1] value_states.shape=", value_states.shape)
        '''
        fxl: NB this attends to all tokens in len (eg 512, masked) for each token in the batch (64)
            sounds like lots of wasted computation ????
        eg
        [forward_prefill.1] hidden_states.shape= torch.Size([1, bs=64, dim=4096])
        [forward_prefill.1] query_states.shape= torch.Size([1, #q_heads=32, bs=64, head_dim 128])
        [forward_prefill.1] key_states.shape= torch.Size([1, #k_heads(repated)=32, len=512, head_dim=128])
        [forward_prefill.1] value_states.shape= torch.Size([1, 32, 512, 128])
        '''
        # Compute scaled dot-product attention      fxl: understadn better: why use einsum rather than matmul?
        # bhqd: bs, heads, q_len, head_dim, bhkd: bs, heads, k_len, head_dim   
        #           -> bhqk: bs, heads, q_len, k_len (i.e. per-head attn scores
        #  ---> einsum transposes keystates automatically....
        # attn_weights = torch.einsum('bhqd,bhkd->bhqk', query_states, key_states) * self.scaling_factor
        #   (einsum - supported in MIL IR
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', query_states, key_states) / math.sqrt(self.head_dim)    # cf modeling_phimoe.py forward()

        if causal_mask is not None:
            if ENABLE_DEBUG3:  
                print("[forward_prefill.2] causal_mask.shape=", causal_mask.shape)
                print("[forward_prefill.2] attn_weights.shape=", attn_weights.shape)
            attn_weights = attn_weights + causal_mask[:, :, :self.config.context_length]
        
        ''' eg 
        [forward_prefill.2] causal_mask.shape= torch.Size([1, 1, bs=64, 512])
        [forward_prefill.2] attn_weights.shape= torch.Size([1, #heads(repeated)32, bs=64, len=512])
        '''

        if ENABLE_VALUES:
            print(f"[forward_prefill]  BATCH attn_weights first 10={attn_weights[0, -1, 0:10, 0:10].tolist()}")
        
        attn_weights = self.ANE_softmax(attn_weights, dim=-1)
        # bhkq: per head attn scores, bhkd: per head values (# of values == # of keys)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value_states)  # [batch=1, heads=32, q_len=1, head_dim=64]
        
        # Reshape before projecting: [batch, heads, q_len, head_dim] -> [batch, q_len, heads*head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Project output
        attn_output = self.o_proj(attn_output)
        
        return attn_output

    # fxl: is the following copy-based on ANE? (i guess wont matter b/c data vol is small?
    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads n_rep times, while keeping the batch dimension intact.
        Input shape: (num_key_value_heads, seqlen, head_dim)
        Output shape: (batch=1, num_attention_heads, seqlen, head_dim)
        """
        x = x.unsqueeze(1)  # Shape: (num_kv_heads, 1, seq_len, head_dim)
        x = x.repeat(1, n_rep, 1, 1)  # Shape: (num_kv_heads, n_rep, seq_len, head_dim)
        x = x.view(1, -1, x.size(-2), x.size(-1))  # Shape: (1, num_kv_heads * n_rep, seq_len, head_dim)
        return x

class PhimoeDecoderLayer(nn.Module):
    """Transformer decoder layer for Phi."""

    def __init__(self, config, layer_idx: int, use_ane_norm=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx      # why needed?
        self.self_attn = PhimoeAttention(config, layer_idx)
        self.block_sparse_moe = PhimoeSparseMoeBlock(config)
        
        # Use ANE_NORM if enabled, otherwise use RMSNorm
        if use_ane_norm:        # not implemented ... 
            self.input_layernorm = LayerNormANE(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = LayerNormANE(config.hidden_size, eps=config.rms_norm_eps)
        else:  # use standard layernorm (not llamaRMSnorm) - fxl
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
            self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)

    # fxl: TBD XXX used??    
    '''
    def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                    kv_cache_layer=None, current_pos=None, IN_PREFILL=False):
        assert False, "PhimoeDecoderLayer.forward() not implemented"
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Pass the layer's KV cache to attention, Class Attention
        if IN_PREFILL:
            hidden_states = self.self_attn.forward_prefill(
                hidden_states=hidden_states,
                query_states=query_states,
                causal_mask=causal_mask,
                kv_cache_layer=kv_cache_layer
            )
        else:
            hidden_states = self.self_attn.forward_regular(
                hidden_states=hidden_states,
                query_states=query_states,
                causal_mask=causal_mask,
                kv_cache_layer=kv_cache_layer
            )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    '''

def get_kv_cache_idx(layer_idx, num_layers, num_groups=1):
    """Helper function to get KV cache indices."""
    layers_per_group = num_layers // num_groups
    group_idx = layer_idx // layers_per_group
    layer_in_group_idx = layer_idx % layers_per_group
    return group_idx, layer_in_group_idx, layers_per_group

class PhimoeModel(BaseModel):
    """Phi Moe model implementation.
    fxl: these are transformer blocks; no embedding, no LM head
    """

    def __init__(self, config, use_ane_norm=False, model_path=None):
        super().__init__(config)
        self.use_ane_norm = use_ane_norm
        self.model_path = model_path
        

        # Initialize layers
        self.layers = nn.ModuleList([
            PhimoeDecoderLayer(config, layer_idx=i, use_ane_norm=use_ane_norm) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Initialize normalization
        if use_ane_norm:
            self.norm = LayerNormANE(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
            
        # Initialize KV cache with MODEL_DTYPE
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # fxl: note the shape, lowest dim is head_dim, then len BEFORE # of heads
        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            cache_size = (
                2 * config.num_hidden_layers,
                config.num_key_value_heads,
                self.config.state_length,
                self.head_dim
            )
            self.register_buffer("kv_cache_0", torch.zeros(cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE))
            if ENABLE_DEBUG:
                print(f"Initialized unified KV kv_cache_0 with shape: {self.kv_cache_0.shape}")
        else:
            layers_per_group = config.num_hidden_layers
            for i in range(config.num_hidden_layers):
                cache_size = (
                    2 * layers_per_group,
                    config.num_key_value_heads,
                    self.config.state_length,
                    self.head_dim
                )
                self.register_buffer(
                    f"kv_cache_{i}", 
                    torch.zeros(cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE)
                )

    def get_rotary_embeddings_s(self, current_pos):
        """Get rotary embeddings for the current position"""
        if ENABLE_VALUES:
            print(f"\n[DEBUG] get_rotary_embeddings_s:")
            print(f"  current_pos: {current_pos}")
        
        # fxl: all layer have same rotary embedding? 
        #    below indexing: suppose sin_cached has shape [1(?), max_context=128k, dim=128]
        #      then it takes a "pos" out and reshape to 4D tensor (1,1,1,128)
        #         first 3dims force to be 1; last dim=-1 means torch shall infer size
        #      (1,1,1,128) is to be bdcast??
        sin = self.layers[0].self_attn.rotary_emb.sin_cached[:, current_pos].view(1, 1, 1, -1)
        cos = self.layers[0].self_attn.rotary_emb.cos_cached[:, current_pos].view(1, 1, 1, -1)
        if ENABLE_VALUES:
            print(f"  cos shape: {cos.shape}")
            print(f"  sin shape: {sin.shape}")
            print(f"  cos first 5: {cos[0,0,0,:5].tolist()}")
            print(f"  sin first 5: {sin[0,0,0,:5].tolist()}")
        
        return cos, sin

    def get_rotary_embedding_prefill(self, positions):
        """Get rotary embeddings for a sequence of positions.
        Args:
            positions: Tensor of position indices
        Returns:
            Tuple of (cos, sin) tensors with shape [1, seq_len, 1, head_dim]   fxl: why this shape <<<<<???
        """
        # Get rotary embeddings from the first attention layer
        rotary_emb = self.layers[0].self_attn.rotary_emb
        
        # Get embeddings for the range of positions directly
        seq_len = positions.size(0)
        cos = rotary_emb.cos_cached[:, positions].view(1, seq_len, 1, rotary_emb.dim)
        sin = rotary_emb.sin_cached[:, positions].view(1, seq_len, 1, rotary_emb.dim)
        
        if ENABLE_VALUES:
            print(f"cos shape: {cos.shape}")
            print(f"sin shape: {sin.shape}")

            print(f"[TRACE] get_rotary_embedding_prefill Batched rotary from pos {positions[0]}:")
            print(f"  cos shape: {cos.shape}, values[0,:5]: {cos[0,0,0,:5].tolist()}")
            print(f"  sin shape: {sin.shape}, values[0,:5]: {sin[0,0,0,:5].tolist()}")

        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    # fxl: 4/17/25 the basic idea seems: npu cannot process a long seq. therefore, cut into small subseq. 
    #   as such, prefill cannot be done in one shot, instead it must process each subseq. and "current_pos"
    #   is the starting position of the subseq (btw, not in use??). there's a KVcache  maintained across subseqs. and attention is computed
    #   and computed across the subseq    (understood 2/5
    def process_layer_prefill(self, layer_idx, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, layer_offset):
        """Process a single transformer layer in prefill mode"""
        layer = self.layers[layer_idx]
        batch_size = position_ids.shape[0]

        if ENABLE_VALUES:
            print("BATCH------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} Input hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")
        normalized_states = layer.input_layernorm(hidden_states)

        if ENABLE_VALUES:
            print("BATCH------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} after input_layernorm hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")
        
        if ENABLE_DEBUG3:
            print(f"position_ids={position_ids}")
            print(f"position_ids.shape={position_ids.shape}")
            print(f"rotary_emb.shape={rotary_emb[0].shape}")
            # eg. rotary_emb.shape=torch.Size([1, len=64, 1, headdim=128])
            print("[process_layer_prefill] causal_mask.shape=", causal_mask.shape)
        # e.g causal_mask.shape= torch.Size([1, 1, bs=64, len=512]) bs is the slice len. 

        # Get query, key and value states for prefill           fxl: proj the new tokens (with rope
        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
            normalized_states,
            current_pos,
            rotary_emb,
            batch_size
        )

        # Get group indices   (fxl: group means layer group, as a way to partition a model
        group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(layer_idx, self.config.num_hidden_layers)

        # Get the combined KV cache for this group
        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            kv_cache = getattr(self, "kv_cache_0")
        else:
            kv_cache = getattr(self, f"kv_cache_{group_idx}")

        key_idx = layer_in_group_idx
        value_idx = layer_in_group_idx + layers_per_group       # fxl: b/c kv cache is 2x the layer size (kcache, then vcache)

        # Store the full sequence length in prefill mode       (fxl: append to kvcache?
        seq_length = key_states.shape[2]  # Get actual sequence length
        kv_cache[key_idx:key_idx + 1, :, current_pos:current_pos + seq_length, :] = key_states
        kv_cache[value_idx:value_idx + 1, :, current_pos:current_pos + seq_length, :] = value_states
        
        # Get the key and value states for this layer from the merged cache (fxl: extract past kvcache? (separate as k,v)
        key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
        value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)

        # Run attention with the updated KV cache
        attn_output = layer.self_attn.forward_prefill(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
        )

        hidden_states = hidden_states + attn_output

        # Skip MLP for last layer in prefill mode since output isn't used
        is_last_layer = (layer_idx == len(self.layers) - 1)
        if not is_last_layer:
            # Add post-attention normalization and MLP
            post_attn = layer.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + layer.block_sparse_moe(post_attn)
        else:
            print("Skipping MLP for last layer in prefill mode")

        if ENABLE_VALUES:
            print("BATCH------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} Output hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")

        return hidden_states

    def process_layer_regular(self, layer_idx, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, layer_offset):
        """Process a single transformer layer in regular (non-prefill) mode"""
        layer = self.layers[layer_idx]
        batch_size = position_ids.shape[0]

        if ENABLE_VALUES:
            print("SINGLE------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} Input hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")

        if ENABLE_DEBUG2:
            print(f"normalized_states.shape={hidden_states.shape}")

        normalized_states = layer.input_layernorm(hidden_states)
        if ENABLE_DEBUG2:
            print(f"normalized_states.shape={normalized_states.shape}")

        if ENABLE_VALUES:
            print("SINGLE------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} after input_layernorm hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")
        
        if ENABLE_DEBUG2:
            print(f"position_ids={position_ids}")
            print(f"position_ids.shape={position_ids.shape}")
            print(f"rotary_emb.shape={rotary_emb[0].shape}")

        # Get query, key and value states for regular processing (fxl: project 
        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache(
            normalized_states,
            current_pos,
            rotary_emb
        )

        # Get group indices
        group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(layer_idx, self.config.num_hidden_layers)

        # Get the combined KV cache for this group
        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            kv_cache = getattr(self, "kv_cache_0")
        else:
            kv_cache = getattr(self, f"kv_cache_{group_idx}")

        key_idx = layer_in_group_idx
        value_idx = layer_in_group_idx + layers_per_group

        # For non-prefill, store single position
        kv_cache[key_idx:key_idx + 1, :, current_pos:current_pos + 1, :] = key_states
        kv_cache[value_idx:value_idx + 1, :, current_pos:current_pos + 1, :] = value_states
        
        # Get the key and value states for this layer from the merged cache
        key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
        value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)

        # Run attention with the updated KV cache
        attn_output = layer.self_attn.forward_regular(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
        )

        hidden_states = hidden_states + attn_output

        # Add post-attention normalization and MLP
        post_attn = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.block_sparse_moe(post_attn)

        if ENABLE_VALUES:
            print("SINGLE------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} Output hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")
        return hidden_states

    def process_layer(self, layer_idx, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, layer_offset, IN_PREFILL=False):
        """Process a single transformer layer, delegating to the appropriate mode-specific implementation"""
        if IN_PREFILL:
           return self.process_layer_prefill(layer_idx, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, layer_offset)
        else:
            return self.process_layer_regular(layer_idx, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, layer_offset)

    def process_layers(self, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, start_layer=0, end_layer=None, IN_PREFILL=False):
        """Process a range of transformer layers"""
        if end_layer is None:
            end_layer = len(self.layers)

        layer_offset = 0
        if not ENABLE_UNIFIED_CACHE:
            layer_offset = start_layer

        for i in range(start_layer, end_layer):
            hidden_states = self.process_layer(
                i, hidden_states,  position_ids,
                causal_mask, current_pos, rotary_emb, layer_offset, IN_PREFILL
            )
            #if TEST_ANE > 0 and i >= TEST_ANE:
            #    break
        return hidden_states

    def forward(self, hidden_states, position_ids=None, causal_mask=None, current_pos=None,
                start_layer=0, end_layer=None, IN_PREFILL=False):
        """
        Forward pass with support for partial layer execution.
        
        Args:
            hidden_states: Input tensor
            update_mask: Mask for KV cache updates
            position_ids: Position IDs
            causal_mask: Causal attention mask
            current_pos: Current position in the sequence
            start_layer: Start processing from this layer (inclusive)
            end_layer: End processing at this layer (exclusive)
        """
        if ENABLE_DEBUG2:
            print(f"PhimoeModel.forward - hidden_states shape: {hidden_states.shape}")

        # Get rotary embeddings
        if IN_PREFILL:      # fxl: rope for a sequence of positions (will slice)
            rotary_emb = self.get_rotary_embedding_prefill(position_ids)
        else:
            rotary_emb = self.get_rotary_embeddings_s(current_pos)

        # Process layers
        hidden_states = self.process_layers(
            hidden_states, position_ids, causal_mask,
            current_pos, rotary_emb, start_layer, end_layer, IN_PREFILL=IN_PREFILL,
        )
        if ENABLE_VALUES:
            print(f"phimoeModel.forward - hidden_states last 10: {hidden_states[-1, -1, -10:].tolist()}")

        # Apply final normalization if this is the last block
        if end_layer is None or end_layer == len(self.layers):
            if IN_PREFILL:
                print("Skipping final normalization for prefill, data not used!")
                # return first batch only to mimizie memory usage and avoid optimization!
                return hidden_states[:,0:1,:]
            else:
                if ENABLE_VALUES:
                    print(f"phimoeModel.forward b4 self.norm hidden_states last 10: {hidden_states[-1, -1, -10:].tolist()}")
                hidden_states = self.norm(hidden_states)
                if ENABLE_VALUES:
                    print(f"phimoeModel.forward AFTER self.norm hiddhistoryen_states last 10: {hidden_states[-1, -1, -10:].tolist()}")

        return hidden_states

    #LlamaModel.forward_prefill    fxl -- not in use??
    '''
    def forward_prefill(self, hidden_states, position_ids=None, causal_mask=None, current_pos=None, start_layer=None, end_layer=None):
        """
        Forward pass for prefilling KV cache
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Get rotary embeddings
        rotary_emb = self.get_rotary_embedding_prefill(position_ids)
        
        # Process layers within the specified range if provided
        if start_layer is not None and end_layer is not None:
            hidden_states = self.process_layers(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                rotary_emb=rotary_emb,
                start_layer=start_layer,
                end_layer=end_layer,
                IN_PREFILL=True
            )
        else:
            # Process all layers for non-split mode
            hidden_states = self.process_layers(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                rotary_emb=rotary_emb,
                IN_PREFILL=True
            )
        
        # Apply final normalization if this is the last block

        if end_layer is None or end_layer == len(self.layers):
            if ENABLE_VALUES:
                print(f"phimoeModel.forward b4 self.norm hidden_states last 10: {hidden_states[-1, -1, -10:].tolist()}")
            
            hidden_states = self.norm(hidden_states)
            
            if ENABLE_VALUES:
                print(f"phimoeModel.forward AFTER self.norm hidden_states last 10: {hidden_states[-1, -1, -10:].tolist()}")

        return hidden_states
    '''

# fxl: not in use??
'''
def stable_l2_norm(x, eps):
    """Compute stable L2 norm optimized for ANE.
    
    Args:
        x: Input tensor
        eps: Small value to prevent division by zero
    
    Returns:
        Normalized tensor and scale factor
    """
    # Find maximum absolute value for scaling
    max_val = x.abs().max(axis=-1, keepdim=True).values
    max_val = torch.clamp(max_val, min=eps)
    
    # Scale input to prevent overflow
    xscaled = x / max_val
    
    # Compute L2 norm on scaled values
    scaled_norm = torch.linalg.norm(xscaled, dim=-1, keepdim=True)
    scaled_norm = torch.clamp(scaled_norm, min=eps)
    
    return x / scaled_norm, max_val
'''

# fxl: this wraps around Phimoe Model and adds lm_head embed
#      also responsible for loading model weights 
class PhimoeForCausalLM(nn.Module):
    """Phi Moe model with causal language modeling head."""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, use_ane_norm=False, enable_coreml=False):
        super().__init__()
        self.config = config
        self.enable_coreml = enable_coreml
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(TEST_DEVICE)
        self.model = PhimoeModel(config, use_ane_norm=use_ane_norm).to(TEST_DEVICE)
        
        # Initialize lm_head as Conv2d for ANE optimization  fxl: <--- this, interesting
        if ENABLE_CONV2D:
            if ENABLE_VACAB_SPLIT8:
                self.lm_head8_1 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_2 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_3 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_4 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_5 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_6 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_7 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_8 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                print("Created lm_head8_1 through lm_head8_8")
            elif ENABLE_VACAB_SPLIT:
                self.lm_head2_1 = nn.Conv2d(config.hidden_size, config.vocab_size//2, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head2_2 = nn.Conv2d(config.hidden_size, config.vocab_size//2, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                print("Created lm_head2_1 and lm_head2_2")
            else:
                self.lm_head1 = nn.Conv2d(config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                print("Created lm_head1")
        else:
            # Use linear head
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
            print("Created linear lm_head")

    def load_pretrained_weights(self, model_path, **kwargs):
        """Load pretrained weights for both the base model and embeddings."""
        '''
        fxl: loads weights from .safetensors files and maps them to a model that may have some reshaped or renamed layers
        '''

        print("Loading pretrained weights...")
        
        # load a list of safe tensor files ....  into a flat dict...
        file_dict = {}
        for file in tqdm(os.listdir(model_path)):
            if file.endswith(".safetensors"):
                file_dict.update(safetensors.torch.load_file(os.path.join(model_path, file)))
        # file_dict -- flat dictionary mapping parameter names to tensors

        # Handle lm_head weight
        lm_head_present = False
        embed_tokens_key = None
        for k, v in file_dict.items():
            if k == "lm_head.weight":
                lm_head_present = True
            if "embed_tokens.weight" in k:
                embed_tokens_key = k

        if not lm_head_present:
            print("lm_head.weight not found in the model file dictionary")
            if embed_tokens_key:
                print(f"Using {embed_tokens_key} for lm_head.weight")
                file_dict['lm_head.weight'] = file_dict[embed_tokens_key].clone()
            else:
                print("embed_tokens.weight not found. Unable to set lm_head.weight")
                return False

        # Filter and reshape weights   (filter --> selects a subset of weights... lm_head etc
        # below, split emb/head weights (as "splits"), and create new keys for them.... save back to "filtered_state_dict"
        filtered_state_dict = {}
        for k, v in file_dict.items():
            if k == "model.embed_tokens.weight":
                print(f"Loading {k} with shape {v.shape}")
                filtered_state_dict["embed_tokens.weight"] = v  # Keep original dtype
                print(f"Moving model.embed_tokens.weight to embed_tokens.weight")
            elif k == "lm_head.weight":
                if ENABLE_CONV2D:
                    reshaped_weight = v.view(v.shape[0], v.shape[1], 1, 1)
                    if ENABLE_VACAB_SPLIT8:
                        vocab_split = self.config.vocab_size // 8
                        splits = torch.split(reshaped_weight, vocab_split)    # fxl: split a tensor into smaller parts
                        for i, split in enumerate(splits):
                            filtered_state_dict[f"lm_head8_{i+1}.weight"] = split
                            print(f"Split lm_head weight into lm_head8_{i+1}.weight with shape {split.shape}")
                    elif ENABLE_VACAB_SPLIT:
                        vocab_split = self.config.vocab_size // 2
                        split1, split2 = torch.split(reshaped_weight, [vocab_split, self.config.vocab_size - vocab_split])
                        filtered_state_dict["lm_head2_1.weight"] = split1
                        filtered_state_dict["lm_head2_2.weight"] = split2
                    else:
                        filtered_state_dict["lm_head1.weight"] = reshaped_weight
                else:
                    filtered_state_dict["lm_head.weight"] = v

        # Load filtered weights
        # fxl: "missing_keys is a list of str containing any keys that are expected"  https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
        #  below, load "filtered_state_dict" into model ("self") 
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)

        if missing_keys:
            print(f"(after loading emb/head) Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"(after loading emb/head) Unexpected keys: {unexpected_keys}")

        # Load weights for the base model (base: transformer blocvks 
        #       fxl add dims, make weights 4D (but keep o_proj 2D)... for ANE optimization? (like it or coreml expects 4 dims?)
        base_filtered_dict = {}
        for k, v in file_dict.items():
            if k.startswith("model."):
                new_key = k.replace("model.", "")
                if "layers." in new_key:
                    # Handle attention weights
                    if 'self_attn' in new_key and 'weight' in new_key:
                        if 'o_proj' in new_key:
                            base_filtered_dict[new_key] = v
                            if "layers.0" in new_key:   #only print once. 
                                print(f"Keeping o_proj weights as 2D: {new_key} shape {v.shape}... (& more layers...")
                        else:
                            reshaped_weight = v.view(v.shape[0], v.shape[1], 1, 1)
                            base_filtered_dict[new_key] = reshaped_weight
                            if "layers.0" in new_key:   #only print once.
                                print(f"Reshaped {new_key} from {v.shape} to {reshaped_weight.shape}... (& more layers...")
                    # MoE experts... eg 'model.layers.1.block_sparse_moe.experts.5.w1.weight'
                    elif 'block_sparse_moe.experts.' in new_key and 'weight' in new_key:
                        reshaped_weight = v.view(v.shape[0], v.shape[1], 1, 1)
                        base_filtered_dict[new_key] = reshaped_weight
                        if "layers.0" in new_key:   #only print once. 
                            print(f"Reshaped MoE weight {new_key} from {v.shape} to {reshaped_weight.shape}... (& more layers...")
                    # MoE router eg 'model.layers.0.block_sparse_moe.gate.weight'
                    elif "block_sparse_moe.gate.weight" in new_key:
                        reshaped_weight = v.view(v.shape[0], v.shape[1], 1, 1)
                        base_filtered_dict[new_key] = reshaped_weight
                        print(f"Reshaped MoE router weight {new_key} from {v.shape} to {reshaped_weight.shape}")
                    else:
                        base_filtered_dict[new_key] = v
                elif new_key == "norm.weight" or new_key == "norm.bias":  # phi has layernorm with has bias, not RMS
                    base_filtered_dict[new_key] = v

        # Load base model weights    (<--- will load base_filtered_dict from model ("self"
        missing_keys, unexpected_keys = self.model.load_state_dict(base_filtered_dict, strict=False)

        # Filter out expected missing keys
        expected_missing = ['kv_cache_0']  # KV cache buffer is initialized separately
        expected_missing.extend([f'layers.{i}.self_attn.rotary_emb.inv_freq' for i in range(self.config.num_hidden_layers)])
        
        # "actual_missing" -- unexpected keys in model 
        actual_missing = [k for k in missing_keys if k not in expected_missing]
        # if not actual_missing and not unexpected_keys:   # fxl <-- this is correct, but for testing we may load fewer layers
        if not actual_missing:
            print("Pretrained weights loaded successfully")
            # fxl: these tensors (?) are "expected", but not provided as part of models. so they were initialized as part of model construction
            if missing_keys:
                print("Note: The following expected buffers were initialized:")
                for k in missing_keys:
                    print(f"  - {k}")
            return True
        else:
            print("Pretrained weights loaded with some issues:")
            if actual_missing:
                print(f"(actual) Missing keys: {actual_missing}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            return False

    # fxl: only used for testing 
    def prefill_kv_cache(self, input_ids, position_ids, start_pos, causal_mask):
        """
        Pre-fills KV cache for a batch of tokens starting from start_pos.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_length]
            start_pos: Starting position in the KV cache
            causal_mask: Causal attention mask
            
        Returns:
            None (updates KV cache in-place)
        """
        #print(f"[DEBUG] Prefill phase started. start_pos={start_pos}")

        batch_size, seq_length = input_ids.shape
        
        # Create position IDs for this batch - each token should get its correct position
        
        if ENABLE_DEBUG3:       
            print(f"[DEBUG] Position IDs: {position_ids.tolist()}")
        
        
        # Get embeddings and run through model
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.to(MODEL_DTYPE)

        # Get correct causal mask for the sequence
        # For prefill, each token should attend to all previous tokens in the sequence
        if causal_mask is not None:
            # Take the full sequence slice of causal mask
            causal_mask_prefill = causal_mask[:, :, :seq_length, :]
            #print(f"[DEBUG] Using causal mask shape: {causal_mask_prefill.shape}")
        else:
            causal_mask_prefill = None
        
        # Process through model to update KV cache
        with torch.no_grad():
            self.model.forward_prefill(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask_prefill,
                current_pos=start_pos
            )

    # fxl: XXX handle this? -- "this model may need to switch between short and long rope, invalidating the cache in the process"
    def forward(
        self,
        input_ids: torch.LongTensor,
        update_mask: torch.FloatTensor,   # fxl: not in use??
        position_ids: torch.LongTensor,
        current_pos: int,
        causal_mask: torch.Tensor,
        IN_PREFILL: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for causal language modeling."""
        if ENABLE_DEBUG:
            print(f"phimoeForCausalLM::forward called with input_ids: {input_ids.shape}, update_mask: {update_mask.shape}, position_ids: {position_ids.shape}, causal_mask: {causal_mask.shape}, current_pos: {current_pos}")
        # Get embeddings

        # Assert input_ids is a 2D tensor
        assert len(input_ids.shape) == 2, f"input_ids must be a 2D tensor, got shape {input_ids.shape}"
        if not IN_PREFILL:
            assert len(position_ids.shape) == 1, f"position_ids must be a 1D tensor for Inference, got shape {position_ids.shape}"
        else:
            assert position_ids.shape[-1] == input_ids.shape[-1], f"position_ids last dimension should match input_ids for Prefill, got shape {position_ids.shape} and input_ids shape {input_ids.shape}"

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.to(MODEL_DTYPE)

        if ENABLE_VALUES:
            print(f"phimoeForCausalLM::embed_tokens weight dtype: {self.embed_tokens.weight.dtype}")
            print(f"phimoeForCausalLM::embed_tokens input_idss (values):{input_ids.tolist()}")
            print(f"phimoeForCausalLM::embed_tokens model input hidden states (first 16 values):{hidden_states[0,0,:16].tolist()}")

        # Process through transformer layers
        hidden_states = self.model(
            hidden_states=hidden_states,
            position_ids=position_ids,
            current_pos=current_pos,
            causal_mask=causal_mask,  # Pass causal_mask through
            start_layer=0,
            end_layer=None,
            IN_PREFILL=IN_PREFILL,  # Added trailing comma
        )
        
        # Project to vocabulary using appropriate head
        if ENABLE_CONV2D:
            # Reshape for Conv2d and ensure float16
            hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
            
            if ENABLE_VACAB_SPLIT8:
                # Use 8-way split head   (fxl: so ANE favors smaller projection??
                logits1 = self.lm_head8_1(hidden_states).squeeze(2).transpose(1, 2)
                logits2 = self.lm_head8_2(hidden_states).squeeze(2).transpose(1, 2)
                logits3 = self.lm_head8_3(hidden_states).squeeze(2).transpose(1, 2)
                logits4 = self.lm_head8_4(hidden_states).squeeze(2).transpose(1, 2)
                logits5 = self.lm_head8_5(hidden_states).squeeze(2).transpose(1, 2)
                logits6 = self.lm_head8_6(hidden_states).squeeze(2).transpose(1, 2)
                logits7 = self.lm_head8_7(hidden_states).squeeze(2).transpose(1, 2)
                logits8 = self.lm_head8_8(hidden_states).squeeze(2).transpose(1, 2)
                
                if self.enable_coreml and ENABLE_LOGITS2:
                    return logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8
                else:
                    logits = torch.cat([logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8], dim=2)
            
            elif ENABLE_VACAB_SPLIT:
                # Use 2-way split head
                logits1 = self.lm_head2_1(hidden_states).squeeze(2).transpose(1, 2)
                logits2 = self.lm_head2_2(hidden_states).squeeze(2).transpose(1, 2)
                
                if self.enable_coreml and ENABLE_LOGITS2:
                    return logits1, logits2
                
                logits = torch.cat([logits1, logits2], dim=2)
            
            else:
                # Use single head
                logits = self.lm_head1(hidden_states).squeeze(2).transpose(1, 2)
        else:
            # Use linear head
            logits = self.lm_head(hidden_states)
        
        return logits

class PhimoeConverter(BaseConverter):
    """Handles Phimoe model conversion to CoreML."""

    def __init__(self, config, model_path=None, use_ane_norm=False):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.use_ane_norm = use_ane_norm
        # Initialize model with enable_coreml=True for CoreML conversion
        self.model = PhimoeForCausalLM(config, use_ane_norm=use_ane_norm, enable_coreml=True)
        
        if False and model_path:
            self.model.model.load_pretrained_weights(
                model_path,
                enable_conv2d=True,
                mlp_up_split=1,
                mlp_down_split=1
            )

    def convert(self):
        self.preprocess()
        # Phimoe model needs special handling before CoreML conversion
        coreml_model = self.convert_to_coreml(self.model)
        self.postprocess()
        return coreml_model

    def convert_to_coreml(self, model):
        """Convert Phimoe model using CoreMLTools."""
        return ct.convert(model)  # Placeholder for actual logic
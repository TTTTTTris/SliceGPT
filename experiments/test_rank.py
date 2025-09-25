#%%
import torch
import torch.nn as nn

def compute_rank_slicegpt(ratio):
    bi_score = torch.load('/home/yifanyang/TransformerCompression/bi/wikitext2/llama_7b/VO_false/sparsity_score_{ratio}.pt'.format(ratio=ratio))
    print(bi_score, torch.mean(bi_score))
    size_attn = 4096 * 4096
    size_mlp = 4096* 11008
    size_org = (size_attn * 4 + size_mlp * 2) * len(bi_score)
    size_total = 0
    for num in range(len(bi_score) - 1):
        score_1 = bi_score[num]
        score_2 = bi_score[num + 1]
        size_total += (size_attn * 4 + size_mlp) * score_1 + size_mlp * score_2
        # adapters
        size_total += score_1 * 4096 * score_1 * 4096
        size_total += score_1 * 4096 * score_2 * 4096
    size_total += (size_attn * 4 + size_mlp) * bi_score[-1] + size_mlp
    print('result ratio', size_total / size_org)

compute_rank_slicegpt(0.2)

#%%
def compute_uniform_slicegpt(ratio):
    size_attn = 4096 * 4096
    size_mlp = 4096 * 11008
    size_org = (size_attn * 4 + size_mlp * 2) * 32 + 32000 * 4096 * 2
    size_total = 32000 * 4096 * (1 + ratio)
    print(size_total)
    for num in range(31):
        size_total += (size_attn * 4 + size_mlp * 2) * ratio
        # adapters
        size_total += ratio * 4096 * ratio * 4096 * 2
    size_total += (size_attn * 4 + size_mlp) * ratio + size_mlp
    size_total += ratio * 4096 * ratio * 4096 + ratio * 4096 * 4096
    print('result ratio', 1 - size_total / size_org)
    print(size_total / 1024 / 1024 / 1024)

compute_uniform_slicegpt(0.8)
compute_uniform_slicegpt(0.7)
compute_uniform_slicegpt(0.6)
compute_uniform_slicegpt(0.55)
compute_uniform_slicegpt(0.45)
compute_uniform_slicegpt(0.35)
print(sum(p.numel() for p in model.parameters()) / 1024 / 1024 / 1024)

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
# cuda device
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = torch.load('./out_slicegpt/llama-2-7b-uniform/wikitext2/uniform/Llama-2-7b-hf_0.4')

print(model)

#%%
# for i in range(32):
#     print(model.model.layers[i].attn_shortcut_Q.shape)
#     print(model.model.layers[i].mlp_shortcut_Q.shape)

for name, para in model.named_parameters():
    print(name, para.numel())
# print(sum(p.numel() for p in model.parameters()))

#%%
new_embedding_dimension = int(0.6 * 4096)
# round (down) to the nearest multiple of round_interval
new_embedding_dimension -= new_embedding_dimension % 1
print(new_embedding_dimension, 4096 * 0.6)
print(4096 * new_embedding_dimension - 10059776)
# print(int(4096 * 11008 * 0.6) - 27035648)
# print(32000 * 4096 - 131072000)

#%%
import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# model = torch.load('./out_slicegpt/llama-2-7b-uniform/wikitext2/Llama-2-7b-hf_0.4')
model = torch.load('./out_models/llama-2-7b-uniform/no_QK/Llama-2-7b-hf_88')

#%%
print(model.model.layers[0].mlp.gate_proj)
print(model.model.layers[0].mlp.gate_proj.weight.shape)
# print(model.model.layers[0].mlp.down_proj.weight.shape)

#%%

import math

def compute_llama_decoder_flops(
    batch_size,
    seq_len,
    hidden_dim,
    intermediate_dim,
    num_heads,
    use_kv_cache=True,
    ratio=1.0
):
    """
    Compute FLOPs for one LLaMA decoder block.
    
    Args:
        batch_size: batch size
        seq_len: current sequence length (including cached tokens)
        hidden_dim: hidden size (e.g., 4096)
        intermediate_dim: MLP intermediate size (e.g., 11008)
        num_heads: number of attention heads (e.g., 32)
        use_kv_cache: whether to assume KV cache is used
    Returns:
        total_flops: estimated FLOPs
    """

    # --- Attention ---
    # Q: (batch_size, seq_len, hidden_dim) x (hidden_dim, hidden_dim) => 2 * batch_size * seq_len * hidden_dim * hidden_dim
    flops_q = 2 * batch_size * seq_len * hidden_dim * hidden_dim

    # K,V: (batch_size, 1, hidden_dim) x (hidden_dim, hidden_dim) if using kv_cache
    if use_kv_cache:
        flops_kv = 2 * batch_size * 1 * hidden_dim * hidden_dim * (ratio + 1) # K and V separately
    else:
        flops_kv = 2 * batch_size * seq_len * hidden_dim * hidden_dim * (ratio + 1)  # recompute K,V every time
    
    # Attention score: Q (batch, seq_len, head_dim) x K^T (batch, head_dim, seq_len)
    head_dim = hidden_dim // num_heads
    flops_attn_scores = batch_size * num_heads * seq_len * seq_len * head_dim * 2  # matmul: 2 flops per multiply-add

    # Softmax: usually ~5 flops per element for numerical stability, but let's use 5
    flops_softmax = batch_size * num_heads * seq_len * seq_len * 5

    # Attention weighted value: (batch, seq_len, seq_len) x (batch, seq_len, head_dim)
    flops_attn_weighted_v = batch_size * num_heads * seq_len * seq_len * head_dim * 2 * ratio

    # Attention output projection
    flops_attn_out = 2 * batch_size * seq_len * hidden_dim * hidden_dim * ratio

    # --- MLP ---
    # First Linear: (batch_size, seq_len, hidden_dim) x (hidden_dim, intermediate_dim)
    flops_mlp_fc1 = 2 * batch_size * seq_len * hidden_dim * intermediate_dim * ratio

    # Activation (SiLU): around 4 flops per element (multiply, sigmoid, multiply)
    flops_activation = batch_size * seq_len * intermediate_dim * 4 * ratio

    # Second Linear: (batch_size, seq_len, intermediate_dim) x (intermediate_dim, hidden_dim)
    flops_mlp_fc2 = 2 * batch_size * seq_len * intermediate_dim * hidden_dim * ratio

    # --- LayerNorms (approximate as 5 flops per element * 2 norms) ---
    flops_layernorm = batch_size * seq_len * hidden_dim * 5 * 2

    # --- Total ---
    total_flops = (
        flops_q
        + flops_kv
        + flops_attn_scores
        + flops_softmax
        + flops_attn_weighted_v
        + flops_attn_out
        + flops_mlp_fc1
        + flops_activation
        + flops_mlp_fc2
        + flops_layernorm
    )

    return total_flops

def compute_llama_model_flops(
    batch_size,
    seq_len,
    hidden_dim,
    intermediate_dim,
    num_heads,
    num_layers,
    use_kv_cache=True,
    ratio=1.0
):
    """
    Compute FLOPs for the entire LLaMA model.
    
    Args:
        batch_size: batch size
        seq_len: current sequence length (including cached tokens)
        hidden_dim: hidden size (e.g., 4096)
        intermediate_dim: MLP intermediate size (e.g., 11008)
        num_heads: number of attention heads (e.g., 32)
        num_layers: number of decoder blocks
        use_kv_cache: whether to assume KV cache is used
    Returns:
        total_flops: estimated FLOPs
    """
    flops_per_block = compute_llama_decoder_flops(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        use_kv_cache=use_kv_cache,
        ratio=ratio
    )
    
    total_flops = flops_per_block * num_layers
    return total_flops

# Example usage:
if __name__ == "__main__":
    batch_size = 256
    seq_len = 256  # e.g., total length including cache
    hidden_dim = 4096
    intermediate_dim = 11008
    num_heads = 32
    num_layers = 32  # Number of decoder blocks
    ratio = 1.0
    n_tokens = 256
    flops_list = []
    for n_tokens in [64, 128, 256, 512]:
        for ratio in [0.4, 0.52, 0.64, 0.76, 0.88, 1.0]:
            print('ratio', ratio)
            # Compute FLOPs for each token
            flops = 0
            # Compute FLOPs for each token
            for i in range(n_tokens):
                use_kv_cache = True if i > 0 else False
                flops += compute_llama_model_flops(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,  # Number of decoder blocks
                    use_kv_cache=use_kv_cache,
                    ratio=ratio
                )

                num_layers = 32  # Number of decoder blocks
            flops_list.append(flops)
            print(f"Estimated FLOPs: {flops / 1e9:.2f} GFLOPs")

import matplotlib.pyplot as plt
import numpy as np

# flops_list = [flops / flops_list[-1] for flops in flops_list]
flops_array = np.array(flops_list)
flops_list = flops_array.reshape(-1, 6)
for i in range(flops_list.shape[0]):
    flops_list[i] = flops_list[i][-1] / flops_list[i]
    print(flops_list[i])
    plt.plot(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), flops_list[i], label=f"n_tokens={2**(i+6)}")
plt.xlabel('Ratio')
plt.ylabel('FLOPs (GFLOPs)')
plt.title('FLOPs vs Ratio')
plt.legend()
plt.grid()
plt.show()

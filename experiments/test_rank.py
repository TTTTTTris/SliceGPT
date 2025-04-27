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
model = torch.load('/raid0-data/yifan/out_slicegpt/llama-2-7b-uniform/wikitext2/uniform/Llama-2-7b-hf_0.4')

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

# model = torch.load('/raid0-data/yifan/out_slicegpt/llama-2-7b-uniform/wikitext2/Llama-2-7b-hf_0.4')
model = torch.load('/raid0-data/yifan/out_models/llama-2-7b-uniform/no_QK/Llama-2-7b-hf_88')

#%%
print(model.model.layers[0].mlp.gate_proj)
print(model.model.layers[0].mlp.gate_proj.weight.shape)
# print(model.model.layers[0].mlp.down_proj.weight.shape)


import torch
from lavis.models.blip2_models.Qformer import BertSelfAttention
"""
Baseline pruning strategies
"""
def position_based_baseline(query_tokens, downsample=1):
    """
    Pruning strategy based on position 
    - (only keeps first K tokens in the sequence)
    """
    # Baseline strategy to prune query tokens
    # (batch_dim, sequence_dim, hidden_dim)
    seq_len = query_tokens.shape[1]
    reduced_seq_len = seq_len // downsample
    return query_tokens[:, :reduced_seq_len, :]

def magnitude_based_baseline(query_tokens, downsample=1):
    """
    Pruning strategy based on magnitude (l2-norm)
    - (only keeps top k large tokens in the sequence)
    """
    seq_len = query_tokens.shape[1]
    reduced_seq_len = seq_len // downsample
    seleced_seq_batch = []
    # Calculate magnitude
    l2_norms = torch.norm(query_tokens, dim=2)
    for i in range(query_tokens.shape[0]):
        # Find topk token magnitude in each sequence
        _, indices = torch.topk(l2_norms[i, :], k=reduced_seq_len)
        # Maintain its original order
        sorted_indices, _ = torch.sort(indices)
        seleced_seq_batch.append(query_tokens[i, sorted_indices, :])
    reduced_tensor = torch.stack(seleced_seq_batch, dim=0)
    return reduced_tensor

"""
Our pruning strategy based on token-wise importance
"""
def importance_pruning(query_tokens, downsample=1, Qformer=None):
    """
    Pruning strategy based on cross-attention
    """
    seq_len = query_tokens.shape[1]
    reduced_seq_len = seq_len // downsample
    all_attn_probs = []
    # Set true to use hansong's method and false to use our method
    hansong_flag = False
    for _, layer in Qformer.named_modules():
        if isinstance(layer, BertSelfAttention) and layer.is_cross_attention == (not hansong_flag):
            # print(f'{name}, attention_probs={layer.attn_probs}, attn_probs_shape={layer.attn_probs.shape}')
            all_attn_probs.append(layer.attn_probs)

    ALL_LAYERS_FLAG = False
    # (layer_dim, batch_dim, head_dim, query_dim, image_dim)
    if ALL_LAYERS_FLAG:
        all_attn_probs = torch.stack(all_attn_probs, dim=0)
    else:
        all_attn_probs = all_attn_probs[-1] 

    # (batch_dim, query_dim)
    importance = torch.zeros((query_tokens.shape[0],query_tokens.shape[1]), device = 'cuda:0')
    for i in range(all_attn_probs.shape[-1]): #loop through image patch
        if ALL_LAYERS_FLAG:
            r = torch.argsort(torch.argsort(all_attn_probs[:,:,:,:,i])) #for one image, token importance value
        else:
            r = torch.argsort(torch.argsort(all_attn_probs[:,:,:,i])) #for one image, token importance value
        if ALL_LAYERS_FLAG:
            importance += torch.sum(r, dim=(0,2)) # accumulate through all image patches
        else:
            importance += torch.sum(r, dim=(1)) # accumulate through all image patches

    selected_seq_batch = []
    # Loop batch dimensin
    for i in range(query_tokens.shape[0]):
        _, indices = torch.topk(importance[i,:], k=reduced_seq_len)
        # Maintain its original order
        sorted_indices, _ = torch.sort(indices)
        # print(f"\tSorted_indices: {sorted_indices}")
        selected_seq_batch.append(query_tokens[i, sorted_indices, :])

    reduced_tensor = torch.stack(selected_seq_batch, dim=0)
    return reduced_tensor


import torch
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
def importance_pruning(query_tokens,
                       downsample=1, all_attn_scores=None):
    """
    Pruning strategy based on cross-attention
    """
    print(type(cross_attentions), len(cross_attentions))
    for i in range(len(cross_attentions)):
        if isinstance(cross_attentions[i],tuple):
            print(f"dim {i}:",len(cross_attentions[i]))

        elif isinstance(cross_attentions[i],torch.Tensor):
            print(f"dim {i}:",cross_attentions[i].shape)
    #importance_score = [item.sum(0).sum(1) for item in all_attn_scores]
    # print(f"size {0}",cross_attentions[0].shape)





    # print(cross_attentions)
    raise RuntimeError("Not implemented")
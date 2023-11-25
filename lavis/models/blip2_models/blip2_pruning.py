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
                       cross_attentions=None,
                       downsample=1):
    """
    Pruning strategy based on cross-attention
    """
    # len(cross_attentions) == 12 (N_LAYERS in q-former)
    print(type(cross_attentions), len(cross_attentions))
    # Shape (32, 12, 32, 257)
    #        (Batch dim, head dim, query_tokens, image_tokens)
    print(cross_attentions[0].shape)
    raise RuntimeError("Not implemented")

def self_attention_pruning(query_tokens,
                          self_attentions,
                          downsample=1):
    DEBUG = False
    """
    Pruning based on self attention proabbility
    """
    seq_len = query_tokens.shape[1]
    reduced_seq_len = seq_len // downsample
    # (layer_dim, batch_dim, head_dim, query_dim, key_dim)
    self_attentions = torch.stack(self_attentions, dim=0)
    sum_self_attentions = torch.sum(self_attentions, dim=(0, 2, 3))

    if DEBUG:
        for sample_id in range(sum_self_attentions.shape[0]):
            print(f"sample_id={sample_id}, sum_self_attentions={sum_self_attentions[sample_id, :]}")
    # Select topK tokens for each sample
    selected_seq_batch = []
    for i in range(query_tokens.shape[0]):
        _, indices = torch.topk(sum_self_attentions[i, :], k=reduced_seq_len)
        # Maintain in its original order
        sorted_indices, _ = torch.sort(indices)
        selected_seq_batch.append(query_tokens[i, sorted_indices, :])
    
    reduced_tensor = torch.stack(selected_seq_batch, dim=0)
    return reduced_tensor

def cross_attention_pruning(query_tokens,
                            cross_attentions,
                            downsample=1):
    DEBUG = True
    seq_len = query_tokens.shape[1]
    reduced_seq_len = seq_len // downsample
    # Get cross attnetions
    cross_attentions = list(filter(lambda cross_attention: isinstance(cross_attention, torch.Tensor), cross_attentions))
    cross_attentions = torch.stack(cross_attentions, dim=0)
    # Expected cross attentions shape (layer_dim, bs_dim, head_dim, query_dim, key_dim)
    # Reduce along layer dimension and head dimension
    cross_attentions_reduced = torch.sum(cross_attentions, dim=(0, 2))
    # Expected cross_attention_reduced shape (bs_dim, query_dim, key_dim)
    selected_seq_batch = []
    for i in range(query_tokens.shape[0]):
        cross_attention_sample = cross_attentions_reduced[i, :, :]
        # Higher prob => lower rank
        query_rankings = torch.argsort(cross_attention_sample, dim=0, descending=True) 
        # Average along image path dimension
        query_rankings = torch.mean(query_rankings.float(), dim=1)
        # Find indicies with smallest ranks
        _, indices = torch.topk(query_rankings, k=reduced_seq_len, largest=False)
        # Maintain in its original order
        sorted_indices, _ = torch.sort(indices)
        selected_seq_batch.append(query_tokens[i, sorted_indices, :])

    reduced_tensor = torch.stack(selected_seq_batch, dim=0)
    return reduced_tensor

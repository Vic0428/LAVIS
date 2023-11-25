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
    sum_self_attentions = []
    for layer_id, self_attention in enumerate(self_attentions):
        assert(isinstance(self_attention, torch.Tensor) == True)
        # Self_attnetion shape: (batch_size_dim, heads_dim, query_dim, key_dim)
        self_attention_sum = torch.sum(self_attention, dim=(1, 2))
        sum_self_attentions.append(self_attention_sum)

    # (layer_dim, batch_dim, key_dim)
    sum_self_attentions = torch.stack(sum_self_attentions, dim=0)
    # Let's take a sum to determine the accumulate across layers
    # (batch_dim. key_dim)
    sum_self_attentions = torch.sum(sum_self_attentions, dim=0)
    if DEBUG:
        for sample_id in range(sum_self_attentions.shape[0]):
            print(f"sample_id={sample_id}, sum_self_attentions={sum_self_attentions[sample_id, :]}")
    # Select topK tokens for each sample
    seleced_seq_batch = []
    for i in range(query_tokens.shape[0]):
        _, indices = torch.topk(sum_self_attentions[i, :], k=reduced_seq_len)
        # Maintain in its original order
        sorted_indices, _ = torch.sort(indices)
        seleced_seq_batch.append(query_tokens[i, sorted_indices, :])
    
    reduced_tensor = torch.stack(seleced_seq_batch, dim=0)
    return reduced_tensor
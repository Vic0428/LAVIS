import torch
from lavis.models.blip2_models.blip2_cs242 import init_logger

pruning_logger = init_logger(__name__)

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
    pruning_logger.info("Apply self_attention pruning")
    ALL_LAYERS = True
    """
    Pruning based on self attention probability
    """
    seq_len = query_tokens.shape[1]
    reduced_seq_len = seq_len // downsample
    # (layer_dim, batch_dim, head_dim, query_dim, key_dim)
    self_attentions = torch.stack(self_attentions, dim=0)
    if ALL_LAYERS:
        pruning_logger.info("\tPruning based on all self_attention layers")
        sum_self_attentions = torch.sum(self_attentions, dim=(0, 2, 3))
    else:
        pruning_logger.info("\tPruning based on the last self_attention layers")
        sum_self_attentions = torch.sum(self_attentions[-1], dim=(1, 2))

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
    pruning_logger.debug("Apply cross_attention pruning")
    ALL_LAYERS = False
    batch_sz, seq_len, _ = query_tokens.shape
    reduced_seq_len = seq_len // downsample
    # Get cross attentions
    cross_attentions = list(filter(lambda cross_attention: isinstance(cross_attention, torch.Tensor), cross_attentions))
    if ALL_LAYERS:
        pruning_logger.debug("\tPruning based on all cross attention layers")
        # Expected cross attentions shape (layer_dim, bs_dim, head_dim, query_dim, key_dim)
        cross_attentions = torch.stack(cross_attentions, dim=0)
        # Reduce along layer dimension and head dimension
        cross_attentions_reduced = torch.sum(cross_attentions, dim=0)
    else:
        pruning_logger.debug("\tPruning based on the last cross attention layers")
        cross_attentions_reduced = cross_attentions[-1]

    # Expected cross_attention_reduced shape (bs_dim, head_dim, query_dim, key_dim)
    selected_seq_batch = []
    for i in range(batch_sz):
        # Shape: (head_dim, query_dim, key_dim)
        cross_attention_sample = cross_attentions_reduced[i]
        # alt_query_rankings = torch.zeros((seq_len)).to('cuda')
        # for j in range(cross_attention_sample.shape[2]):
            # r = torch.argsort(torch.argsort(cross_attention_sample[:, :, j]))
            # alt_query_rankings += torch.sum(r, dim=0)
        # # Each (head, image_token) scores query_tokens along the query dimension
        query_rankings = torch.argsort(torch.argsort(cross_attention_sample, dim=1), dim=1)
        query_rankings = torch.sum(query_rankings, dim=(0, 2))
        # assert(query_rankings.equal(query_rankings) == True)

        # Vote: Aggregate score across (head_dim, key_dim)
        # query_rankings = torch.sum(query_rankings, dim=(0, 2))
        _, indices = torch.topk(query_rankings, k=reduced_seq_len)
        # Maintain in its original order
        sorted_indices, _ = torch.sort(indices)
        selected_seq_batch.append(query_tokens[i, sorted_indices, :])

    reduced_tensor = torch.stack(selected_seq_batch, dim=0)
    return reduced_tensor

def cross_attention_pruning_with_image_weight(query_tokens,
                                              cross_attentions,
                                              vit_self_attentions,
                                              downsample=1):
    pruning_logger.debug("Apply cross_attention pruning with image weight")
   
    batch_sz, seq_len, _ = query_tokens.shape
    reduced_seq_len = seq_len // downsample
    # Get cross attentions
    cross_attentions = list(filter(lambda cross_attention: isinstance(cross_attention, torch.Tensor), cross_attentions))

    pruning_logger.debug("\tPruning based on the last cross attention layers")
    cross_attentions_reduced = cross_attentions[-1]

    # Get image weight
    # (layer_dim, batch_dim, head_dim, query_dim, key_dim)
    vit_self_attentions = torch.stack(vit_self_attentions, dim=0)
    image_scores = torch.sum(vit_self_attentions[-1], dim=(1, 2)) # shape (batch_dim, key_dim)
    image_weight = torch.nn.functional.softmax(image_scores, dim=1)
    
    # Expected cross_attention_reduced shape (bs_dim, head_dim, query_dim, key_dim)
    selected_seq_batch = []
    for i in range(batch_sz):
        # Shape: (head_dim, query_dim, key_dim)
        cross_attention_sample = cross_attentions_reduced[i]
        
        # Each (head, image_token) scores query_tokens along the query dimension
        query_rankings = torch.argsort(torch.argsort(cross_attention_sample, dim=1), dim=1)
        query_rankings = torch.sum(query_rankings, dim=(0, 2))

        # Apply image_weight to the query_rankings
        weighted_query_rankings = query_rankings * image_weight[i]

        # Vote: Aggregate score across (head_dim, key_dim)
        _, indices = torch.topk(weighted_query_rankings, k=reduced_seq_len)
        
        # Maintain in its original order
        sorted_indices, _ = torch.sort(indices)
        selected_seq_batch.append(query_tokens[i, sorted_indices, :])

    reduced_tensor = torch.stack(selected_seq_batch, dim=0)
    return reduced_tensor

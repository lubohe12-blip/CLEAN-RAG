import torch
import torch.nn.functional as F
 
def SupConHardLoss(model_emb, temp, n_pos):
    '''
    return the SupCon-Hard loss
    features:  
        model output embedding, dimension [bsz, n_all, out_dim], 
        where bsz is batchsize, 
        n_all is anchor, pos, neg (n_all = 1 + n_pos + n_neg)
        and out_dim is embedding dimension
    temp:
        temperature     
    n_pos:
        number of positive examples per anchor
    '''
    # l2 normalize every embedding
    features = F.normalize(model_emb, dim=-1, p=2)
    # features_T is [bsz, outdim, n_all], for performing batch dot product
    features_T = torch.transpose(features, 1, 2)
    # anchor is the first embedding 
    anchor = features[:, 0]
    # anchor is the first embedding 
    anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T)/temp 
    # anchor_dot_features now [bsz, n_all], contains 
    anchor_dot_features = anchor_dot_features.squeeze(1)
    # deduct by max logits, which will be 1/temp since features are L2 normalized 
    logits = anchor_dot_features - 1/temp
    # the exp(z_i dot z_a) excludes the dot product between itself
    # exp_logits is of size [bsz, n_pos+n_neg]
    exp_logits = torch.exp(logits[:, 1:])
    exp_logits_sum = n_pos * torch.log(exp_logits.sum(1)) # size [bsz], scale by n_pos
    pos_logits_sum = logits[:, 1:n_pos+1].sum(1) #sum over all (anchor dot pos)
    log_prob = (pos_logits_sum - exp_logits_sum)/n_pos
    loss = - log_prob.mean()
    return loss    


def _compute_batch_centers(embeddings, labels):
    labels = labels.to(device=embeddings.device, dtype=torch.long).view(-1)
    unique_labels, inverse_indices, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True)
    center_sums = torch.zeros(
        unique_labels.size(0),
        embeddings.size(1),
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    center_sums.index_add_(0, inverse_indices, embeddings)
    centers = center_sums / counts.unsqueeze(1).to(dtype=embeddings.dtype)
    return unique_labels, inverse_indices, counts, centers


def compute_gaussian_well_loss(embeddings, labels, sigma):
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for Gaussian well regularization.")

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [B, D].")

    labels = labels.to(device=embeddings.device, dtype=torch.long).view(-1)
    if labels.numel() != embeddings.size(0):
        raise ValueError("labels must have the same batch size as embeddings.")

    _, inverse_indices, counts, centers = _compute_batch_centers(embeddings, labels)
    valid_sample_mask = counts[inverse_indices] >= 2
    valid_sample_count = int(valid_sample_mask.sum().item())
    valid_class_count = int((counts >= 2).sum().item())

    if valid_sample_count == 0:
        zero = embeddings.new_zeros(())
        return zero, {
            "valid_sample_count": 0,
            "valid_class_count": valid_class_count,
        }

    valid_embeddings = embeddings[valid_sample_mask]
    valid_centers = centers[inverse_indices[valid_sample_mask]]
    squared_distances = (valid_embeddings - valid_centers).pow(2).sum(dim=1)
    sigma_sq = embeddings.new_tensor(float(sigma) ** 2)
    loss = 1.0 - torch.exp(-squared_distances / sigma_sq)
    return loss.mean(), {
        "valid_sample_count": valid_sample_count,
        "valid_class_count": valid_class_count,
    }


def compute_embedding_compactness_stats(embeddings, labels):
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [B, D].")

    labels = labels.to(device=embeddings.device, dtype=torch.long).view(-1)
    if labels.numel() != embeddings.size(0):
        raise ValueError("labels must have the same batch size as embeddings.")

    unique_labels, inverse_indices, counts, centers = _compute_batch_centers(
        embeddings, labels)
    valid_sample_mask = counts[inverse_indices] >= 2
    valid_class_mask = counts >= 2

    stats = {
        "intra_center_dist_sum": 0.0,
        "intra_center_dist_count": 0,
        "intra_pairwise_dist_sum": 0.0,
        "intra_pairwise_dist_count": 0,
        "nearest_negative_center_dist_sum": 0.0,
        "nearest_negative_center_dist_count": 0,
        "valid_sample_count": int(valid_sample_mask.sum().item()),
        "valid_class_count": int(valid_class_mask.sum().item()),
    }

    if stats["valid_sample_count"] > 0:
        valid_embeddings = embeddings[valid_sample_mask]
        valid_centers = centers[inverse_indices[valid_sample_mask]]
        intra_center_dist = (valid_embeddings - valid_centers).norm(dim=1, p=2)
        stats["intra_center_dist_sum"] = float(intra_center_dist.sum().item())
        stats["intra_center_dist_count"] = int(intra_center_dist.numel())

    if unique_labels.numel() > 1:
        center_distances = torch.cdist(embeddings, centers, p=2)
        center_distances.scatter_(
            1,
            inverse_indices.unsqueeze(1),
            float("inf"),
        )
        nearest_negative_center_dist = center_distances.min(dim=1).values
        finite_mask = torch.isfinite(nearest_negative_center_dist)
        if finite_mask.any():
            valid_nearest = nearest_negative_center_dist[finite_mask]
            stats["nearest_negative_center_dist_sum"] = float(valid_nearest.sum().item())
            stats["nearest_negative_center_dist_count"] = int(valid_nearest.numel())

    for class_index, count in enumerate(counts.tolist()):
        if count < 2:
            continue
        class_embeddings = embeddings[inverse_indices == class_index]
        pairwise_distances = torch.pdist(class_embeddings, p=2)
        if pairwise_distances.numel() == 0:
            continue
        stats["intra_pairwise_dist_sum"] += float(pairwise_distances.sum().item())
        stats["intra_pairwise_dist_count"] += int(pairwise_distances.numel())

    return stats

# ==================================================
# INFLUENCER RECOMMENDATION ENGINE
# ==================================================

@torch.no_grad()
def recommend_influencers(model, data, brand_idx, top_k=10, device='cuda'):
    """Recommend top-k influencers for a brand."""
    model.eval()
    data = data.to(device)

    # Get embeddings
    z_dict = model(data.x_dict, data.edge_index_dict)

    # Brand embedding
    brand_emb = z_dict['brand'][brand_idx].unsqueeze(0)

    # All influencer embeddings
    inf_embs = z_dict['influencer']

    # Compute match scores
    brand_emb_expanded = brand_emb.expand(inf_embs.size(0), -1)
    scores = model.predict_match(brand_emb_expanded, inf_embs).squeeze()
    scores = torch.sigmoid(scores)

    # Get top-k
    top_scores, top_indices = torch.topk(scores, k=top_k)

    return top_indices.cpu().numpy(), top_scores.cpu().numpy()

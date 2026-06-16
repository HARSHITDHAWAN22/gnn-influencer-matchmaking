# ==================================================
# NEGATIVE SAMPLING FOR LINK PREDICTION
# ==================================================

def negative_sampling(num_brands, num_influencers, num_samples, device):
    """Generate negative samples for training."""
    neg_brand = torch.randint(0, num_brands, (num_samples,), device=device)
    neg_inf = torch.randint(0, num_influencers, (num_samples,), device=device)
    return torch.stack([neg_brand, neg_inf], dim=0)




# ============================================================================
# RECOMMENDATION FUNCTION
# ============================================================================

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

# ============================================================================
# TEST RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("TESTING RECOMMENDATIONS")
print("="*80)

# Get recommendations for first 3 brands
for brand_idx in range(min(3, len(brands))):
    print(f"\n{'='*80}")
    print(f"BRAND: {brands.iloc[brand_idx]['brand_name']}")
    print(f"Industry: {brands.iloc[brand_idx]['industry']}")
    print(f"Language: {brands.iloc[brand_idx].get('language_preference', 'N/A')}")
    print(f"Budget Range: {brands.iloc[brand_idx].get('campaign_budget_range', 'N/A')}")
    print(f"{'='*80}")

    top_inf, scores = recommend_influencers(model, graph_data, brand_idx, top_k=10, device=device)

    print(f"\nTop 10 Recommended Influencers:\n")

    for rank, (inf_idx, score) in enumerate(zip(top_inf, scores), 1):
        inf = influencers_data.iloc[inf_idx]

        print(f"Rank {rank:2d} | Match Score: {score:.4f}")
        print(f"  Username: {inf.get('username', 'N/A')}")
        print(f"  Followers: {inf.get('followers', 0):,.0f}")
        print(f"  Engagement: {inf.get('ENGAGEMENT_RATE', 0):.4f}")
        print(f"  Content Quality: {inf.get('Content_quality_score', 0):.4f}")
        print(f"  Price: ₹{inf.get('price_per_post', 0):,.0f}")

        cat_cols = [col for col in influencers_data.columns if col.startswith('cat_')]
        active_cats = [col.replace('cat_', '').replace('_', ' ')
                      for col in cat_cols if inf.get(col, 0) == 1]
        if active_cats:
            print(f"  Categories: {', '.join(active_cats[:3])}")
        print()

print("\ MATCHMAKING MODEL READY!")
print("="*80)


# ==================================================
# MATCH SCORE INTERPRETATION
# ==================================================

# Match Score Guidelines:
# Score >= 0.40 : Excellent Match
# Score >= 0.35 : Very Good Match
# Score >= 0.30 : Good Match
#
# Higher scores indicate stronger compatibility between
# brand requirements and influencer characteristics.



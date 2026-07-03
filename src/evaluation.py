from config.config import (
    TOP_K_RECOMMENDATIONS,
    EXPORT_TOP_K,
    EXPORT_PROGRESS_INTERVAL,
    RECOMMENDATION_OUTPUT_FILE,
    EXCELLENT_MATCH_SCORE,
    VERY_GOOD_MATCH_SCORE,
    GOOD_MATCH_SCORE,
)

# ============================================================================
# RECOMMENDATION FUNCTION
# ============================================================================

@torch.no_grad()
def recommend_influencers(model, data, brand_idx, top_k=TOP_K_RECOMMENDATIONS, device='cuda'):
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

    top_inf, scores = recommend_influencers(model, graph_data, brand_idx, top_k=TOP_K_RECOMMENDATIONS, device=device)

    print(f"\nTop {TOP_K_RECOMMENDATIONS} Recommended Influencers:\n")

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
# Score >= EXCELLENT_MATCH_SCORE : Excellent Match
# Score >= VERY_GOOD_MATCH_SCORE : Very Good Match
# Score >= GOOD_MATCH_SCORE : Good Match
#
# Higher scores indicate stronger compatibility between
# brand requirements and influencer characteristics.


# ==================================================
# RECOMMENDATION EXPORT PIPELINE
# ==================================================

def export_all_recommendations(model, data, brands, influencers_data,
                                output_file=RECOMMENDATION_OUTPUT_FILE,
                                top_k=EXPORT_TOP_K, device='cuda'):
    """Export recommendations for all brands to CSV."""
    print(f"\n{'='*80}")
    print("EXPORTING ALL RECOMMENDATIONS")
    print(f"{'='*80}\n")

    recommendations = []

    for brand_idx in range(len(brands)):
        if brand_idx % EXPORT_PROGRESS_INTERVAL == 0:
            print(f"Processing brand {brand_idx}/{len(brands)}...")

        brand_row = brands.iloc[brand_idx]
        top_inf, scores = recommend_influencers(model, data, brand_idx, top_k, device)

        for rank, (inf_idx, score) in enumerate(zip(top_inf, scores), 1):
            inf_row = influencers_data.iloc[inf_idx]

            recommendations.append({
                'brand_idx': brand_idx,
                'brand_name': brand_row.get('brand_name', ''),
                'brand_industry': brand_row.get('industry', ''),
                'brand_language': brand_row.get('language_preference', ''),
                'brand_budget': brand_row.get('campaign_budget_range', ''),
                'influencer_idx': inf_idx,
                'influencer_username': inf_row.get('username', ''),
                'influencer_followers': inf_row.get('followers', 0),
                'influencer_engagement': inf_row.get('ENGAGEMENT_RATE', 0),
                'influencer_quality': inf_row.get('Content_quality_score', 0),
                'influencer_price': inf_row.get('price_per_post', 0),
                'rank': rank,
                'match_score': score
            })

    # Create DataFrame and save
    df = pd.DataFrame(recommendations)
    df.to_csv(output_file, index=False)

    print(f"\n Recommendations exported to: {output_file}")
    print(f"Total recommendations: {len(df)}")
    print(f"\nSample recommendations:")
    print(df.head(10))

    return df


# ==================================================
# GENERATING FINAL RECOMMENDATION REPORTS
# ==================================================

# Export recommendations
recommendations_df = export_all_recommendations(
    model, graph_data, brands, influencers_data,
    output_file=RECOMMENDATION_OUTPUT_FILE,
    top_k=EXPORT_TOP_K,
    device=device
)



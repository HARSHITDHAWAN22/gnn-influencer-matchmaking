# ==================================================
# HETEROGENEOUS GRAPH DESIGN
# ==================================================

# The graph contains two primary node types:
# 1. Influencers
# 2. Brands
#
# Different node types allow the model to learn relationships
# across entities with different characteristics and objectives.

# ------------------------------------------------------------------
# BRAND TO INFLUENCER EDGES
# ------------------------------------------------------------------
#
# These edges represent potential collaboration opportunities.
#
# Edge weights are computed using compatibility factors such as:
# - Budget alignment
# - Language preferences
# - Audience characteristics
# - Engagement quality
# - Campaign objectives
#
# A higher edge weight indicates a stronger potential match.

# ------------------------------------------------------------------
# INFLUENCER TO INFLUENCER EDGES
# ------------------------------------------------------------------
#
# Influencers with similar content categories, audience profiles,
# or engagement patterns are connected together.
#
# Benefits:
# - Helps identify substitute creators
# - Enables campaign scaling
# - Improves recommendation diversity
# - Captures creator ecosystem relationships
#
# Example:
# A fashion creator can be linked to other fashion creators,
# allowing the model to propagate useful signals between them.

# ------------------------------------------------------------------
# BRAND TO BRAND EDGES
# ------------------------------------------------------------------
#
# Brands with similar campaign requirements are connected.
#
# This allows the model to learn:
# - Similar marketing objectives
# - Comparable audience targets
# - Common collaboration preferences
#
# Knowledge learned from one brand can therefore benefit
# recommendations for similar brands.

# ------------------------------------------------------------------
# EDGE DENSITY STRATEGY
# ------------------------------------------------------------------
#
# Extremely low thresholds produce dense graphs containing many
# weak relationships and noisy connections.
#
# Extremely high thresholds produce sparse graphs and may remove
# useful collaboration opportunities.
#
# A balanced threshold is used to preserve meaningful matches
# while maintaining graph connectivity and recommendation quality.


# ============================================================================
# CREATE HETEROGENEOUS GRAPH WITH MULTI-CRITERIA MATCHING
# ============================================================================

def create_matching_graph(influencers_data, brands, influencer_x, brand_x):
    """
    Create heterogeneous graph with edges based on:
    - Language matching
    - Content type alignment
    - Budget compatibility
    - Engagement requirements
    """
    print("\n" + "="*80)
    print("STEP 4: CREATING HETEROGENEOUS GRAPH")
    print("="*80)

    data = HeteroData()

    # Add node features
    data['influencer'].x = influencer_x
    data['influencer'].num_nodes = len(influencer_x)

    data['brand'].x = brand_x
    data['brand'].num_nodes = len(brand_x)



# ==================================================
# BRAND-INFLUENCER COMPATIBILITY EDGE CREATION
# ==================================================

 print("\n[1/4] Creating brand-influencer edges based on compatibility...")

    edge_list = []
    edge_scores = []

    # Create compatibility edges
    num_brands = len(brands)
    num_influencers = len(influencers_data)

    # Sample brands for edge creation (to keep graph manageable)
    sample_size = min(1000, num_brands)
    sampled_brand_indices = np.random.choice(num_brands, sample_size, replace=False)

    for brand_idx in sampled_brand_indices:
        brand_row = brands.iloc[brand_idx]

        # Get brand requirements
        # Can increse the brand requirements later if we want in this section.
        brand_lang = brand_row.get('language_preference', 'English')
        brand_budget = brand_row.get('budget_per_post', 0.5)
        brand_region = brand_row.get('target_audience_region', 'Pan India')
        brand_format = brand_row.get('preferred_formats', 'Reels')

        # Sample influencers for this brand
        # Here samples 20 influencer for brand requirement we can change it according to our requirement later.
        num_matches = min(20, num_influencers)  # Top 20 potential matches per brand
        candidate_inf_indices = np.random.choice(num_influencers, num_matches, replace=False)

        for inf_idx in candidate_inf_indices:
            inf_row = influencers_data.iloc[inf_idx]

            # Calculate compatibility score
            compatibility_score = 0.0

            # 1. Budget compatibility (30% weight)
            inf_price = inf_row.get('price_per_post', 500) / 1500  # normalize
            budget_diff = abs(brand_budget - inf_price)
            budget_score = max(0, 1 - budget_diff) * 0.3
            compatibility_score += budget_score

            # 2. Language matching (20% weight)
            # In real scenario, you'd have language info for influencers
            lang_score = 0.2  # Assume compatible
            compatibility_score += lang_score

            # 3. Engagement matching (30% weight)
            inf_engagement = inf_row.get('ENGAGEMENT_RATE', 0)
            engagement_score = min(1.0, inf_engagement * 100) * 0.3
            compatibility_score += engagement_score

            # 4. Content quality (20% weight)
            inf_quality = inf_row.get('Content_quality_score', 0.5)
            quality_score = inf_quality * 0.2
            compatibility_score += quality_score

            # Add edge if compatibility is above threshold
            if compatibility_score > 0.3:  # Threshold for edge creation
                edge_list.append([brand_idx, inf_idx])
                edge_scores.append(compatibility_score)

    print(f"  Created {len(edge_list)} compatibility-based edges")

    if len(edge_list) > 0:
        edge_index = torch.LongTensor(edge_list).t()
        edge_weights = torch.FloatTensor(edge_scores)
    else:
        # Fallback: create random edges
        print("  WARNING: No compatible edges found. Creating sample edges...")
        num_edges = 5000
        brand_indices = torch.randint(0, num_brands, (num_edges,))
        inf_indices = torch.randint(0, num_influencers, (num_edges,))
        edge_index = torch.stack([brand_indices, inf_indices], dim=0)
        edge_weights = torch.rand(num_edges)

    # Add edges to graph (bidirectional)
    data['brand', 'matches_with', 'influencer'].edge_index = edge_index
    data['brand', 'matches_with', 'influencer'].edge_attr = edge_weights

    data['influencer', 'matched_by', 'brand'].edge_index = edge_index.flip(0)
    data['influencer', 'matched_by', 'brand'].edge_attr = edge_weights



# ==================================================
# INFLUENCER SIMILARITY EDGE GENERATION
# ==================================================


print("\n[2/4] Creating influencer-influencer similarity edges...")

    inf_edges = []
    cat_cols = [col for col in influencers_data.columns if col.startswith('cat_')]

    if len(cat_cols) > 0:
        # Sample for efficiency
        sample_infs = min(500, num_influencers)
        sampled_inf_indices = np.random.choice(num_influencers, sample_infs, replace=False)

        for inf_idx in sampled_inf_indices:
            inf_cats = influencers_data.iloc[inf_idx][cat_cols].values

            # Find similar influencers (same categories)
            similarities = (influencers_data[cat_cols].values * inf_cats).sum(axis=1)
            top_similar = np.argsort(similarities)[-6:-1]  # Top 5 similar (excluding self)

            for similar_idx in top_similar:
                if similar_idx != inf_idx and similarities[similar_idx] > 0:
                    inf_edges.append([inf_idx, similar_idx])

    if len(inf_edges) > 0:
        data['influencer', 'similar_to', 'influencer'].edge_index = \
            torch.LongTensor(inf_edges).t()

    print(f"  Created {len(inf_edges)} influencer similarity edges")



# ==================================================
# BRAND SIMILARITY EDGE GENERATION
# ==================================================

print("\n[3/4] Creating brand-brand similarity edges...")

    brand_edges = []

    # Group by industry
    for industry in brands['industry'].unique()[:10]:  # Limit to top 10 industries
        brand_indices = brands[brands['industry'] == industry].index.tolist()

        # Connect brands in same industry
        for i in range(min(len(brand_indices), 20)):
            for j in range(i+1, min(i+6, len(brand_indices))):
                brand_edges.append([brand_indices[i], brand_indices[j]])
                brand_edges.append([brand_indices[j], brand_indices[i]])

    if len(brand_edges) > 0:
        data['brand', 'similar_to', 'brand'].edge_index = \
            torch.LongTensor(brand_edges).t()

    print(f"  Created {len(brand_edges)} brand similarity edges")

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print("\n[4/4] Graph construction complete!")
    print("\n" + "="*80)
    print("HETEROGENEOUS GRAPH STRUCTURE:")
    print("="*80)
    print(data)
    print("="*80)

    return data

# Create the graph
graph_data = create_matching_graph(influencers_data, brands, influencer_x, brand_x)

   




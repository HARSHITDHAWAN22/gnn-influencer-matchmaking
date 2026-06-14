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

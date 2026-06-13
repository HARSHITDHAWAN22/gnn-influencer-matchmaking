# ==================================================
# INFLUENCER FEATURE ENGINEERING
# ==================================================

ef prepare_complete_node_features(influencers_data, brands,
                                    influencer_text_embs, brand_text_embs):
    """
    Combine numerical features with text embeddings for rich node representations.
    This includes language, engagement, content type, and all dataset features.
    """
    print("\n" + "="*80)
    print("STEP 3: PREPARING COMPLETE NODE FEATURES")
    print("="*80)

# ==================================================
# INFLUENCER FEATURE ENGINEERING
# ==================================================
    print("\n[1/3] Preparing influencer features...")

    # Numerical features (already normalized in your preprocessing)
    influencer_numerical_cols = [
        'followers', 'ENGAGEMENT_RATE', 'AVG_LIKES_COUNT_10',
        'AVG_COMMENTS_COUNT_10', 'Content_quality_score',
        'Consistency_score', 'GROWTH_RATE', 'influencer_score',
        'ROI', 'male_pct', 'female_pct', 'price_per_post'
    ]

# Get available columns (in case some are missing)
    available_cols = [col for col in influencer_numerical_cols
                     if col in influencers_data.columns]

    influencer_numerical = influencers_data[available_cols].fillna(0).values

    # Add category features (one-hot encoded)
    cat_cols = [col for col in influencers_data.columns if col.startswith('cat_')]
    if len(cat_cols) > 0:
        category_features = influencers_data[cat_cols].fillna(0).values
        influencer_numerical = np.hstack([influencer_numerical, category_features])

    # Combine numerical + text embeddings
    influencer_text_np = influencer_text_embs.cpu().numpy()
    influencer_features = np.hstack([influencer_numerical, influencer_text_np])

    print(f"  Numerical features: {influencer_numerical.shape}")
    print(f"  Text embeddings: {influencer_text_np.shape}")
    print(f"  Combined features: {influencer_features.shape}")


    # Get available columns (in case some are missing)
    available_cols = [col for col in influencer_numerical_cols


# ==================================================
# BRAND FEATURE ENGINEERING
# ==================================================
  
brand_numerical_cols = [
        'kpi_target', 'budget_total', 'budget_per_post',
        'target_age_min', 'target_age_max'
    ]

    brand_numerical = brands[brand_numerical_cols].fillna(0).values

    # Encode categorical features
    categorical_brand_cols = [
        'industry', 'language_preference', 'preferred_platform',
        'target_audience_region', 'engagement_priority', 'preferred_formats'
    ]

    # One-hot encode categorical columns
    for col in categorical_brand_cols:
        if col in brands.columns:
            le = LabelEncoder()
            encoded = le.fit_transform(brands[col].fillna('Unknown'))
            # Normalize to [0, 1]
            encoded_norm = encoded / (encoded.max() + 1)
            brand_numerical = np.hstack([brand_numerical, encoded_norm.reshape(-1, 1)])

    # Combine numerical + text embeddings
    brand_text_np = brand_text_embs.cpu().numpy()
    brand_features = np.hstack([brand_numerical, brand_text_np])

    print(f"  Numerical + categorical: {brand_numerical.shape}")
    print(f"  Text embeddings: {brand_text_np.shape}")
    print(f"  Combined features: {brand_features.shape}")
if col in influencers_data.columns]


# ==================================================
# FEATURE TENSOR GENERATION
# ==================================================

rint("\n[3/3] Converting to PyTorch tensors...")

    influencer_x = torch.FloatTensor(influencer_features)
    brand_x = torch.FloatTensor(brand_features)

    print(f"\n✅ Feature preparation complete!")
    print(f"  Influencer tensor: {influencer_x.shape}")
    print(f"  Brand tensor: {brand_x.shape}")

    return influencer_x, brand_x

# Prepare features
influencer_x, brand_x = prepare_complete_node_features(
    influencers_data, brands, influencer_text_embs, brand_text_embs
)


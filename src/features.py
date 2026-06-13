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

    # -----------------------------------------------------------------------
    # INFLUENCER FEATURES
    # -----------------------------------------------------------------------
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


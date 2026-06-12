# ==================================================
# TEXT EMBEDDINGS FOR LANGUAGE AND CONTENT MATCHING
# ==================================================

# Initialize Sentence Transformer for text embeddings
@torch.no_grad()
#fucntioon which takes input from influencer and brand dataset.
def create_text_embeddings(influencers_data, brands):
    """
    Create text embeddings for language, content, and categorical features.
    This enables semantic matching between brands and influencers.
    """
    print("\n[1/4] Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & efficient model



# ==================================================
# INFLUENCER TEXT FEATURE ENGINEERING
# ==================================================


 # Combine bio and description for influencer text representation
    influencer_texts = []
    for idx, row in influencers_data.iterrows():
        bio = str(row.get('bio', ''))
        desc = str(row.get('description', ''))
        location = str(row.get('primary_location', ''))

        # Get active categories
        cat_cols = [col for col in influencers_data.columns if col.startswith('cat_')]
        categories = [col.replace('cat_', '').replace('_', ' ')
                     for col in cat_cols if row.get(col, 0) == 1]
        cat_text = ', '.join(categories[:3]) if categories else 'general'

        # Combine all text features(all text in 1 string)
        combined_text = f"{bio} {desc} Location: {location} Categories: {cat_text}"
        influencer_texts.append(combined_text)

    # Generate embeddings
    influencer_text_embs = model.encode(influencer_texts,
                                        batch_size=128,
                                        show_progress_bar=True,
                                        convert_to_tensor=True)


# ==================================================
# BRAND TEXT FEATURE ENGINEERING
# ==================================================


  brand_texts = []
    for idx, row in brands.iterrows():
        brand_name = str(row.get('brand_name', ''))
        industry = str(row.get('industry', ''))
        values = str(row.get('brand_values', ''))
        lang = str(row.get('language_preference', ''))
        region = str(row.get('target_audience_region', ''))
        format_pref = str(row.get('preferred_formats', ''))

        # Combine all brand features
        combined_text = f"{brand_name} Industry: {industry} Values: {values} Language: {lang} Region: {region} Format: {format_pref}"
        brand_texts.append(combined_text)

    # Generate embeddings
    brand_text_embs = model.encode(brand_texts,
                                    batch_size=128,
                                    show_progress_bar=True,
                                    convert_to_tensor=True)

    print(f"\n[4/4] Text embeddings created!")
    print(f"  Influencer embeddings: {influencer_text_embs.shape}")
    print(f"  Brand embeddings: {brand_text_embs.shape}")

    return influencer_text_embs, brand_text_embs

# Generate text embeddings
influencer_text_embs, brand_text_embs = create_text_embeddings(influencers_data, brands)

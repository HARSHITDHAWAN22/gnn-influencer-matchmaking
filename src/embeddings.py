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

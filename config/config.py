# ==================================================
# PROJECT CONFIGURATION
# ==================================================

# Random Seed
RANDOM_SEED = 42

# ==================================================
# MODEL
# ==================================================

HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64

# ==================================================
# TRAINING
# ==================================================

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

NUM_EPOCHS = 100
PATIENCE = 15

# ==================================================
# DATA SPLIT
# ==================================================

VALIDATION_RATIO = 0.10
TEST_RATIO = 0.20

# ==================================================
# EVALUATION
# ==================================================

EVALUATION_INTERVAL = 5


# ==================================================
# RANDOMNESS
# ==================================================

RANDOM_SEED = 42

# ==================================================
# BASIC MODEL
# ==================================================

HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64

# ==================================================
# ADVANCED MODEL
# ==================================================

ADV_HIDDEN_CHANNELS = 256
ADV_OUT_CHANNELS = 128

ATTENTION_HEADS = 4

# ==================================================
# DROPOUT
# ==================================================

DROPOUT_LOW = 0.20
DROPOUT_MEDIUM = 0.30

# ==================================================
# TRAINING
# ==================================================

LEARNING_RATE = 0.001
ADV_LEARNING_RATE = 0.0003

WEIGHT_DECAY = 1e-5
ADV_WEIGHT_DECAY = 5e-5

NUM_EPOCHS = 100
ADV_NUM_EPOCHS = 200

PATIENCE = 15
ADV_PATIENCE = 25

# ==================================================
# DATA SPLIT
# ==================================================

VALIDATION_RATIO = 0.10
TEST_RATIO = 0.20

# ==================================================
# EVALUATION
# ==================================================

EVALUATION_INTERVAL = 5

# ==================================================
# FOCAL LOSS
# ==================================================

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.5
LABEL_SMOOTHING = 0.10

# ==================================================
# MIXUP
# ==================================================

MIXUP_ALPHA = 0.20
MIXUP_START_EPOCH = 20

# ==================================================
# HARD NEGATIVE MINING
# ==================================================

HARD_NEGATIVE_START_EPOCH = 15
NEGATIVE_CANDIDATE_MULTIPLIER = 5

# ==================================================
# LOSS
# ==================================================

MARGIN = 0.40
MARGIN_LOSS_WEIGHT = 0.10

# ==================================================
# GRADIENT CLIPPING
# ==================================================

MAX_GRAD_NORM = 1.0

# ==================================================
# LR SCHEDULER
# ==================================================

COSINE_T0 = 50
COSINE_T_MULT = 2

# ==================================================
# MODEL PATHS
# ==================================================

BEST_MODEL_PATH = "best_matchmaking_model.pt"
BEST_ADV_MODEL_PATH = "best_advanced_model.pt"


# ==================================================
# GRAPH CONSTRUCTION
# ==================================================

MAX_BRAND_SAMPLE_SIZE = 1000
MAX_INFLUENCER_MATCHES = 20
MAX_INFLUENCER_SIMILARITY_SAMPLE = 500

# ==================================================
# COMPATIBILITY SCORING
# ==================================================

DEFAULT_BRAND_BUDGET = 0.5
DEFAULT_INFLUENCER_PRICE = 500
PRICE_NORMALIZATION_FACTOR = 1500

BUDGET_WEIGHT = 0.30
LANGUAGE_WEIGHT = 0.20
ENGAGEMENT_WEIGHT = 0.30
QUALITY_WEIGHT = 0.20

DEFAULT_CONTENT_QUALITY = 0.5

COMPATIBILITY_THRESHOLD = 0.30

# ==================================================
# SIMILARITY GRAPH
# ==================================================

TOP_SIMILAR_INFLUENCERS = 5
MAX_BRAND_INDUSTRIES = 10
MAX_BRANDS_PER_INDUSTRY = 20
MAX_BRAND_CONNECTIONS = 5

# ==================================================
# FALLBACK GRAPH
# ==================================================

FALLBACK_EDGE_COUNT = 5000



# ==================================================
# RECOMMENDATION
# ==================================================

TOP_K_RECOMMENDATIONS = 10
EXPORT_TOP_K = 20
EXPORT_PROGRESS_INTERVAL = 100

RECOMMENDATION_OUTPUT_FILE = "matchmaking_recommendations.csv"

# ==================================================
# MATCH SCORE INTERPRETATION
# ==================================================

EXCELLENT_MATCH_SCORE = 0.40
VERY_GOOD_MATCH_SCORE = 0.35
GOOD_MATCH_SCORE = 0.30


# ==================================================
# EVALUATION
# ==================================================

DEFAULT_THRESHOLD = 0.50

THRESHOLD_SEARCH_START = 0.01
THRESHOLD_SEARCH_END = 0.99
THRESHOLD_SEARCH_STEPS = 99

CONFUSION_MATRIX_FIGSIZE = (18, 5)
ROC_CURVE_FIGSIZE = (8, 6)
PR_CURVE_FIGSIZE = (8, 6)
METRICS_BAR_FIGSIZE = (10, 6)

PLOT_Y_LIMIT = 1.05

# ==================================================
# MODEL CHECKPOINTS
# ==================================================

BEST_ADV_MODEL_PATH = "best_advanced_model.pt"


# ==================================================
# HYBRID MODEL (GNN + XGBOOST + LIGHTGBM)
# ==================================================

NEGATIVE_SAMPLES_PER_POSITIVE = 1

# ==================================================
# XGBOOST
# ==================================================

XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 7
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.80
XGB_COLSAMPLE_BYTREE = 0.80
XGB_EVAL_METRIC = "logloss"
XGB_TREE_METHOD = "hist"

# ==================================================
# LIGHTGBM
# ==================================================

LGBM_N_ESTIMATORS = 500
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES = 64
LGBM_SUBSAMPLE = 0.80
LGBM_COLSAMPLE_BYTREE = 0.80

# ==================================================
# HYBRID EVALUATION
# ==================================================

HYBRID_THRESHOLD = 0.50

HYBRID_COMPARISON_FIGSIZE = (10, 6)

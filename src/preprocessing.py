# =====================================
# IMPORTING LIBRARIES
# =====================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
!pip install torch torch-geometric
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
import torch.nn.functional as F


# ==================================================
# LOADING INFLUENCER AND BRAND DATASETS
# ==================================================

import pandas as pd
import requests
import io

# Replace with your actual URLs and token
url_influencers = "https://raw.githubusercontent.com/HARSHITDHAWAN22/AI--MATHCMAKING-MODEL./main/data/insta_dataset_with_labels_v1.csv"
url_brands = "https://raw.githubusercontent.com/HARSHITDHAWAN22/AI--MATHCMAKING-MODEL./main/data/brand_dataset_india_4500_v2.csv"
token = ""

headers = {'Authorization': f'token {token}'}

# Load influencers CSV
response_influencers = requests.get(url_influencers, headers=headers)
response_influencers.raise_for_status()  # Raises HTTPError if 4xx/5xx

influencers = pd.read_csv(io.StringIO(response_influencers.text))

# Load brands CSV
response_brands = requests.get(url_brands, headers=headers)
response_brands.raise_for_status()

brands = pd.read_csv(io.StringIO(response_brands.text))


# influencer_file_path = 'influencers.csv' # Removed the line that reads a local file
# influencers_data = pd.read_csv(influencer_file_path) # Removed the line that reads a local file

influencers_data = influencers.copy() # Use the DataFrame already loaded

# Handling Missing Values and Infinite Values
numeric_cols_influencers = influencers_data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols_influencers = influencers_data.select_dtypes(include=['object']).columns

for col in numeric_cols_influencers:
    # Replace infinite values with NaN
    influencers_data[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    influencers_data[col].fillna(influencers_data[col].median(), inplace=True)

for col in categorical_cols_influencers:
    influencers_data[col].fillna(influencers_data[col].mode()[0] if not influencers_data[col].mode().empty else 'Unknown', inplace=True)

# Normalize important numeric columns, update column names as per influencers dataset
cols_to_normalize_influencers = ['kpi_target', 'kpi_actual', 'performance_ratio'] # Corrected column names based on influencers DataFrame

scaler_inf = MinMaxScaler()
influencers_data[cols_to_normalize_influencers] = scaler_inf.fit_transform(influencers_data[cols_to_normalize_influencers])

# Preview processed influencers data
print("Preprocessed Influencers Data Preview:")
display(influencers_data.head()) # Use display for better output


# ==================================================
# LOCATION-WISE ENGAGEMENT ANALYSIS
# ==================================================

avg_likes = influencers_data.groupby('primary_location')['likes'].mean().sort_values()
plt.figure(figsize=(12,6))
avg_likes.plot(kind='bar', color='coral')
plt.title('Average Likes by Primary Location')
plt.ylabel('Average Likes')
plt.show()


# ==================================================
# CATEGORY-WISE INFLUENCER ANALYSIS
# ==================================================

print("Available columns in influencers_data:")
print(influencers_data.columns.tolist())


# ==================================================
# FOLLOWER COUNT VS PRICE-PER-POST ANALYSIS
# ==================================================

plt.figure(figsize=(12, 8))
sns.scatterplot(x='followers', y='price_per_post', hue='cat_fashion___style', data=influencers_data)
plt.title('Followers vs Price Per Post by Fashion & Style Category (Actual Values)')
plt.xlabel('Followers (Actual Count)')
plt.ylabel('Price Per Post (Actual $)')
plt.ticklabel_format(style='plain')  # Prevents scientific notation
plt.show()


# ==================================================
# BRAND DATA CLEANING AND PREPROCESSING
# ==================================================

numeric_cols_brand = brands.select_dtypes(include=['int64', 'float64']).columns
categorical_cols_brand = brands.select_dtypes(include=['object']).columns

for col in numeric_cols_brand:
    brands[col].fillna(brands[col].median(), inplace=True)

for col in categorical_cols_brand:
    brands[col].fillna(brands[col].mode()[0] if not brands[col].mode().empty else 'Unknown', inplace=True)

# Normalize important numeric columns
cols_to_normalize_brand = ['kpi_target', 'budget_total', 'budget_per_post', 'target_age_min', 'target_age_max']

scaler = MinMaxScaler()
brands[cols_to_normalize_brand] = scaler.fit_transform(brands[cols_to_normalize_brand])

# Preview processed brand data
print("Preprocessed Brand Data Preview:")
print(brands.head())


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# List of columns to normalize for influencer score calculation (example columns)
columns_to_normalize = ['followers', 'ENGAGEMENT_RATE', 'AVG_LIKES_COUNT_10', 'AVG_COMMENTS_COUNT_10']

# Normalize specified columns individually using MinMaxScaler
scaler = MinMaxScaler()
influencers_data[columns_to_normalize] = scaler.fit_transform(influencers_data[columns_to_normalize])

# Example of creating a composite influencer score from normalized columns
# You can define weights as per importance or use equal weight average
weights = {'followers': 0.4, 'ENGAGEMENT_RATE': 0.3, 'AVG_LIKES_COUNT_10': 0.2, 'AVG_COMMENTS_COUNT_10': 0.1}

influencers_data['influencer_score'] = sum(influencers_data[col] * weight for col, weight in weights.items())


# Preview the dataframe with the scores
print(influencers_data[['influencer_score'] + columns_to_normalize].head())


# ==================================================
# FEATURE SCALING FOR MACHINE LEARNING MODELS
# ==================================================

columns_to_normalize = [
    'ENGAGEMENT_RATE', 'GROWTH_RATE', 'Content_quality_score',
    'Consistency_score', 'audience_engagement_score_',
     'male_pct', 'female_pct'
]

# Fill missing values with median to avoid issues during scaling
for col in columns_to_normalize:
   # influencers_data[col].fillna(influencers_data[col].median(), inplace=True) this wwil not work in pandas 3 so , new syntax below
    influencers_data[col] = influencers_data[col].fillna(influencers_data[col].median())

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the columns, updating in the original dataframe
influencers_data[columns_to_normalize] = scaler.fit_transform(influencers_data[columns_to_normalize])


# ==================================================
# ROI MODIFICATION, SCALING AND NORMALIZATION
# ==================================================

# ROI values can vary significantly and may contain extreme or invalid values.
# To improve model training stability, ROI is transformed and normalized
# into a standard range of 0 to 1 using MinMax Scaling.
# This ensures all ROI values are comparable and prevents large variations
# from negatively affecting the machine learning model.

from sklearn.preprocessing import MinMaxScaler

value_metric = 'kpi_actual'       # Value measure column
cost_metric = 'price_per_post'    # Cost measure column

# Clean cost column if object type
if influencers_data[cost_metric].dtype == 'object':
    influencers_data[cost_metric] = influencers_data[cost_metric].astype(str).str.replace('[₹,]', '', regex=True).astype(float)

epsilon = 1e-6  # small number to avoid zero division

# Calculate ratio-based ROI
roi_ratio = influencers_data[value_metric] / (influencers_data[cost_metric] + epsilon)

# Replace infinite or NaN values
roi_ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
roi_ratio.fillna(0, inplace=True)

# Scale ROI between 0 and 1 for easier comparison
scaler = MinMaxScaler(feature_range=(0, 1))
influencers_data['ROI'] = scaler.fit_transform(roi_ratio.values.reshape(-1, 1))

print("Influencers Data with Ratio-based Normalized ROI:")
print(influencers_data[['brand_id', value_metric, cost_metric, 'ROI']].head())

# Calculate the average of the ROI column
average_roi = influencers_data['ROI'].mean()

# Print the average ROI
print(f"The average ROI is: {average_roi}")

# Preview the normalized columns
print(influencers_data[columns_to_normalize].head())


# ==================================================
# CORRELATION ANALYSIS OF NUMERICAL FEATURES
# ==================================================

# Select only numeric columns for correlation calculation
numeric_influencers_data = influencers_data.select_dtypes(include=[np.number])

plt.figure(figsize=(14,12))
sns.heatmap(numeric_influencers_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# ==================================================
# OUTLIER DETECTION USING INTERQUARTILE RANGE (IQR)
# ==================================================


# Dictionary to store outlier indices for each column
outliers = {}

# Identify numeric columns in influencers_data
numeric_cols = influencers_data.select_dtypes(include=['int64', 'float64']).columns

# IQR based outlier detection
for col in numeric_cols:
    Q1 = influencers_data[col].quantile(0.25)
    Q3 = influencers_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Find outlier indices
    col_outliers = influencers_data[(influencers_data[col] < lower_bound) | (influencers_data[col] > upper_bound)].index
    outliers[col] = col_outliers.tolist()
    print(f"Column '{col}' has {len(col_outliers)} outliers.")

# Example: display outliers in a specific column
example_col = numeric_cols[0]  # just the first numeric col
print(f"Outliers in '{example_col}':")
print(influencers_data.loc[outliers[example_col], [example_col]])

# Optionally, you can choose how to handle/remove these outliers later

# ==================================================
# LOG TRANSFORMATION OF SKEWED FEATURES
# ==================================================

influencers_data['sid_profile_log'] = np.log1p(influencers_data['sid_profile'].clip(lower=0))
influencers_data['following_log'] = np.log1p(influencers_data['following'].clip(lower=0))

# Preview before and after stats for confirmation
print("Before and After Log Transformation")
print(influencers_data[['sid_profile', 'sid_profile_log', 'following', 'following_log']].head())


# ==================================================
# WINSORIZATION FOR OUTLIER CAPPING AND DATA STABILITY
# ==================================================

# Winsorization is applied at the 5th and 95th percentiles.
# This preserves data diversity while reducing the influence
# of extreme values that may negatively affect model training.
# Important outlier information is retained instead of removing
# observations completely.


columns_to_winsorize = [
    'likes', 'following', 'followers', 'num_posts', 'price_per_post',
    'kpi_target', 'kpi_actual', 'ROI', 'AVG_LIKES_COUNT_10', 'AVG_COMMENTS_COUNT_10',
    'ENGAGEMENT_RATE', 'GROWTH_RATE', 'average_collabration_score'
]

# Apply capping at 5% and 95% percentiles
lower_quantile = 0.05
upper_quantile = 0.95

for col in columns_to_winsorize:
    lower_bound = influencers_data[col].quantile(lower_quantile)
    upper_bound = influencers_data[col].quantile(upper_quantile)
    influencers_data[col] = influencers_data[col].clip(lower=lower_bound, upper=upper_bound)
    print(f"Capped column '{col}' between {lower_bound:.4f} and {upper_bound:.4f}")

# Preview to confirm
print("\nWinsorized Influencers Data Preview (descriptive statistics):")
print(influencers_data[columns_to_winsorize].describe())


# ==================================================
# MODIFIED Z-SCORE BASED OUTLIER DETECTION AND ANALYSIS
# ==================================================

# Outliers are identified using the Modified Z-Score method,
# which is more robust to skewed distributions and extreme values
# than traditional Z-Score analysis.

def modified_z_score(series):
    median_val = np.median(series)
    mad = np.median(np.abs(series - median_val))
    if mad == 0:
        return np.zeros(len(series))
    return 0.6745 * (series - median_val) / mad

# Columns to apply Modified Z-Score
cols_to_check = ['sid_profile', 'following', 'followers', 'price_per_post']

# Threshold for outlier detection (usually 3.5)
threshold = 3.5

outlier_indices = {}

for col in cols_to_check:
    mod_z_scores = modified_z_score(influencers_data[col])
    outliers = influencers_data[np.abs(mod_z_scores) > threshold].index
    outlier_indices[col] = outliers.tolist()
    print(f"Column '{col}' has {len(outliers)} outliers detected by Modified Z-Score.")

# Example: show the first few outliers for 'sid_profile'
print("Example outliers for 'sid_profile':")
print(influencers_data.loc[outlier_indices['sid_profile'], ['sid_profile']])


# ==================================================
# INFLUENCER SCORE DISTRIBUTION ANALYSIS
# ==================================================

# Visualizing the distribution of influencer scores
# to understand score spread, density and potential
# concentration patterns across influencers.

plt.figure(figsize=(10,6))
sns.histplot(influencers_data['influencer_score_scaled_back'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Influencer Score')
plt.xlabel('Influencer Score')
plt.ylabel('Frequency')
plt.xlim(0, 3)  # Set x-axis limits from 0 to 3
plt.show()

print(influencers_data['influencer_score'])

verage_score = influencers_data['influencer_score'].mean()
print(f'Average influencer_score: {average_score}')

sns.pairplot(influencers_data[['followers', 'likes', 'price_per_post', 'ENGAGEMENT_RATE']].sample(500))
plt.suptitle('Pairplot of Key Influencer Metrics', y=1.02)
plt.show()


# ==================================================
# BRAND DATA EXPLORATION AND QUALITY ANALYSIS
# ==================================================

brands.describe()

plt.figure(figsize=(8,5))
sns.histplot(brands['budget_per_post'], bins=40, kde=True, color='teal')
plt.title('Distribution of Budget Per Post')
plt.xlabel('Budget Per Post')
plt.ylabel('Frequency')
plt.show()

print(brands.isnull().sum())


# ==================================================
# BRAND FEATURE NORMALIZATION FOR SIMILARITY MATCHING
# ==================================================

# Select columns to normalize for similarity (adjust if you want more)
columns_to_normalize = [
    'kpi_target',
    'budget_total',
    'budget_per_post',
    'target_age_min',
    'target_age_max'
]

# Fit and transform using sklearn MinMaxScaler (same as influencers)
scaler = MinMaxScaler()
brands[columns_to_normalize] = scaler.fit_transform(brands[columns_to_normalize])

# Show result
print(brands[columns_to_normalize].describe())

 Columns to check
columns_to_check = ['kpi_target', 'budget_total', 'budget_per_post', 'target_age_min', 'target_age_max']

# Z-score threshold (commonly 3)
z_thresh = 3

for col in columns_to_check:
    col_zscore = (brands[col] - brands[col].mean()) / brands[col].std()
    outliers = brands[np.abs(col_zscore) > z_thresh]
    print(f"Column '{col}' has {outliers.shape[0]} outliers using Z-score method.")
    # Optionally, cap or remove outliers:
    # brands[col] = np.where(np.abs(col_zscore) > z_thresh,
    #                             brands[col].median(),
    #                             brands[col])



# ==================================================
# Z-SCORE BASED OUTLIER ANALYSIS FOR BRAND FEATURES
# ==================================================

# Comparing Z-Score based outlier detection with previous
# approaches to evaluate its effectiveness on brand features.

from scipy.stats import zscore

cols = ['budget_total', 'budget_per_post']
z_thresh = 3

for col in cols:
    z_scores = zscore(brands[col])
    brands = brands[(abs(z_scores) < z_thresh)]

# Columns to check
columns_to_check = ['kpi_target', 'budget_total', 'budget_per_post', 'target_age_min', 'target_age_max']

# Z-score threshold (commonly 3)
z_thresh = 3

for col in columns_to_check:
    col_zscore = (brands[col] - brands[col].mean()) / brands[col].std()
    outliers = brands[np.abs(col_zscore) > z_thresh]
    print(f"Column '{col}' has {outliers.shape[0]} outliers using Z-score method.")
    # Optionally, cap or remove outliers:
    # brands[col] = np.where(np.abs(col_zscore) > z_thresh,
    #                             brands[col].median(),
    #                             brands[col])


# ==================================================
# INDUSTRY-WISE KPI TARGET ANALYSIS
# ==================================================

#LEARNING: WINZORIZATION CAN INCREASE OUTLIER COUNT IN SOME CASES; Z-SCORE PROVIDED BETTER RESULTS DURING EXPERIMENTATION

plt.figure(figsize=(10,6))
sns.boxplot(x='industry', y='kpi_target', data=brands)
plt.title('KPI Target by Industry')
plt.xlabel('Industry')
plt.ylabel('KPI Target')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,8))
corr = brands.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Brand Features')
plt.show()













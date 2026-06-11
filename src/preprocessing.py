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



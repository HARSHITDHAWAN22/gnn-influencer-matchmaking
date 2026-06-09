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

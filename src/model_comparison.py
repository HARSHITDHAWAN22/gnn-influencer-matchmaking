# =====================================================================
# HYBRID GNN + XGBOOST / LIGHTGBM COMPARISON
# =====================================================================

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from xgboost import XGBClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

# ---------------------------------------------------------------------
# 1️ Extract GNN-based embeddings for brand–influencer pairs
# ---------------------------------------------------------------------
@torch.no_grad()
def build_pair_dataset(model, data, edge_index, device, n_negatives=1):
    model.eval()
    data = data.to(device)
    z = model(data.x_dict, data.edge_index_dict)

    b, i = edge_index
    valid = (b < z['brand'].size(0)) & (i < z['influencer'].size(0))
    b, i = b[valid], i[valid]

    # Positive (true matches)
    pos_b, pos_i = z['brand'][b], z['influencer'][i]
    pos_pairs = torch.cat([pos_b, pos_i], dim=1)
    pos_labels = torch.ones(pos_pairs.size(0), dtype=torch.int64)

    # Negative (random pairs)
    neg_b = torch.randint(0, z['brand'].size(0), (pos_pairs.size(0)*n_negatives,), device=device)
    neg_i = torch.randint(0, z['influencer'].size(0), (pos_pairs.size(0)*n_negatives,), device=device)
    neg_b, neg_i = z['brand'][neg_b], z['influencer'][neg_i]
    neg_pairs = torch.cat([neg_b, neg_i], dim=1)
    neg_labels = torch.zeros(neg_pairs.size(0), dtype=torch.int64)

    # Merge
    X = torch.cat([pos_pairs, neg_pairs], dim=0).cpu().numpy()
    y = torch.cat([pos_labels, neg_labels], dim=0).cpu().numpy()
    return X, y

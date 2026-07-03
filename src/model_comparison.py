from config.config import (
    NEGATIVE_SAMPLES_PER_POSITIVE,
    HYBRID_THRESHOLD,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_EVAL_METRIC,
    XGB_TREE_METHOD,
    LGBM_N_ESTIMATORS,
    LGBM_LEARNING_RATE,
    LGBM_NUM_LEAVES,
    LGBM_SUBSAMPLE,
    LGBM_COLSAMPLE_BYTREE,
    HYBRID_COMPARISON_FIGSIZE,
    ROC_CURVE_FIGSIZE,
    PR_CURVE_FIGSIZE,
    PLOT_Y_LIMIT,
)

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
def build_pair_dataset(model, data, edge_index, device, n_negatives=NEGATIVE_SAMPLES_PER_POSITIVE):
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


# ---------------------------------------------------------------------
# 2️ Build datasets from the GNN
# ---------------------------------------------------------------------
X_train, y_train = build_pair_dataset(model, graph_data, train_e, device)
X_val,   y_val   = build_pair_dataset(model, graph_data, val_e, device)
X_test,  y_test  = build_pair_dataset(model, graph_data, test_e, device)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------
# 3️ Train XGBoost
# ---------------------------------------------------------------------
xgb = XGBClassifier(
    n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LEARNING_RATE,
    subsample=XGB_SUBSAMPLE, colsample_bytree=XGB_COLSAMPLE_BYTREE,
    eval_metric=XGB_EVAL_METRIC, tree_method=XGB_TREE_METHOD, random_state=42
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred_prob_xgb = xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_prob_xgb >= HYBRID_THRESHOLD).astype(int)

# ---------------------------------------------------------------------
# 4️ Train LightGBM
# ---------------------------------------------------------------------
lgbm = lgb.LGBMClassifier(
    n_estimators=LGBM_N_ESTIMATORS, learning_rate=LGBM_LEARNING_RATE, num_leaves=LGBM_NUM_LEAVES,
    subsample=LGBM_SUBSAMPLE, colsample_bytree=LGBM_COLSAMPLE_BYTREE, random_state=42
)
lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])

y_pred_prob_lgb = lgbm.predict_proba(X_test)[:, 1]
y_pred_lgb = (y_pred_prob_lgb >= HYBRID_THRESHOLD).astype(int)


# ---------------------------------------------------------------------
# 5️ Collect all metrics
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob, name):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    acc = np.mean(y_true == y_pred)
    return {'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'AUC': auc}

metrics = []
metrics.append(compute_metrics(test_res['y_true'], (test_res['y_prob'] >= HYBRID_THRESHOLD).astype(int), test_res['y_prob'], 'GNN'))
metrics.append(compute_metrics(y_test, y_pred_xgb, y_pred_prob_xgb, 'GNN + XGBoost'))
metrics.append(compute_metrics(y_test, y_pred_lgb, y_pred_prob_lgb, 'GNN + LightGBM'))

metrics_df = pd.DataFrame(metrics)
print("\n=== Final Model Comparison ===")
print(metrics_df.to_string(index=False))


# ---------------------------------------------------------------------
# 6️ Visualization
# ---------------------------------------------------------------------
plt.figure(figsize=HYBRID_COMPARISON_FIGSIZE)
sns.barplot(data=metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Metric', y='Score', hue='Model')
plt.title("Performance Comparison: GNN vs XGBoost vs LightGBM", fontsize=14, weight='bold')
plt.ylim(0, PLOT_Y_LIMIT)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# --- ROC Curves ---
plt.figure(figsize=ROC_CURVE_FIGSIZE)
for (name, y_true, y_prob) in [
    ('GNN', test_res['y_true'], test_res['y_prob']),
    ('XGBoost', y_test, y_pred_prob_xgb),
    ('LightGBM', y_test, y_pred_prob_lgb)
]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.3f})")
plt.plot([0,1], [0,1], '--', color='gray')
plt.title("ROC Curves", fontsize=14, weight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --- Precision–Recall Curves ---
plt.figure(figsize=PR_CURVE_FIGSIZE)
for (name, y_true, y_prob) in [
    ('GNN', test_res['y_true'], test_res['y_prob']),
    ('XGBoost', y_test, y_pred_prob_xgb),
    ('LightGBM', y_test, y_pred_prob_lgb)
]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, precision, label=f"{name} (AP={roc_auc_score(y_true, y_prob):.3f})")
plt.title("Precision–Recall Curves", fontsize=14, weight='bold')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


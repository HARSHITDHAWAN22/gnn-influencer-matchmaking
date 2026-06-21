import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# Replace these names with the variables you use in the notebook:
# y_true_prob / y_pred_prob = predicted probabilities for positive class (match)
# y_true = ground-truth binary labels (1=match, 0=no-match)
# If you currently compute pos = torch.sigmoid(...).cpu().numpy() use that as y_pred_prob

# Example: if your code uses `pos` as probabilities and `neg` as negatives for eval:
# y_pred_prob = np.concatenate([pos.cpu().numpy(), neg_pred.cpu().numpy()])  # adapt as required
# y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg_pred))])

# If you already have y_pred_prob and y_true, skip above and use them.

print("DEBUG: types and shapes")
print("  y_pred_prob type:", type(y_pred_prob), "shape:", getattr(y_pred_prob, 'shape', None))
print("  y_true type:", type(y_true), "shape:", getattr(y_true, 'shape', None))
print("  unique y_true values:", np.unique(y_true, return_counts=True))

# Thresholding
thr = 0.5
y_pred = (y_pred_prob >= thr).astype(int)

print("\nCounts:")
(unique, counts) = np.unique(y_pred, return_counts=True)
print("  Predicted label counts:", dict(zip(unique, counts)))
(unique, counts) = np.unique(y_true, return_counts=True)
print("  True label counts:", dict(zip(unique, counts)))

# Confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix (rows=true, cols=pred):\n", cm)
print("\nclassification_report:\n", classification_report(y_true, y_pred, digits=4, zero_division=0))

# More robust metric (shows where division-by-zero occurs)
prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
print("\nprecision_recall_fscore_support per-class:\n", list(zip(prec, rec, f1, sup)))


# ====================================================================
# FIXED EVALUATION FUNCTION
# ====================================================================

@torch.no_grad()
def evaluate(model, data, edge_index, device, threshold=0.5):
    model.eval()
    data, edge_index = data.to(device), edge_index.to(device)

    # Get embeddings
    z = model(data.x_dict, data.edge_index_dict)

    b, i = edge_index
    valid = (b < z['brand'].size(0)) & (i < z['influencer'].size(0))
    b, i = b[valid], i[valid]

    # Positive edges (true matches)
    pos = torch.sigmoid(model.predict_match(z['brand'][b], z['influencer'][i]).squeeze())

    # Negative sampling (random non-match pairs)
    n = pos.size(0)
    neg_b = torch.randint(0, z['brand'].size(0), (n,), device=device)
    neg_i = torch.randint(0, z['influencer'].size(0), (n,), device=device)
    neg = torch.sigmoid(model.predict_match(z['brand'][neg_b], z['influencer'][neg_i]).squeeze())

    # Combine
    y_prob = torch.cat([pos, neg]).detach().cpu().numpy().astype(np.float64)
    y_true = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])

    # --- Metrics ---
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    rpt = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)

    # Handle missing class keys gracefully
    precision = rpt.get('1', {}).get('precision', 0.0)
    recall = rpt.get('1', {}).get('recall', 0.0)
    f1 = rpt.get('1', {}).get('f1-score', 0.0)

    # Debug sanity check
    print("\n=== DEBUG METRICS ===")
    print(f"y_true unique: {np.unique(y_true, return_counts=True)}")
    print(f"y_pred unique: {np.unique(y_pred, return_counts=True)}")
    print("Confusion matrix:\n", cm)
    print("Classification report summary:", {k: rpt[k] for k in ['0','1'] if k in rpt})

    return {
        'auc': auc, 'ap': ap, 'accuracy': acc,
        'precision': precision, 'recall': recall, 'f1': f1,
        'cm': cm, 'y_true': y_true, 'y_prob': y_prob
    }

# ====================================================================
# LOAD MODEL & DATA
# ====================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
graph_data = create_matching_graph(influencers_data, brands, influencer_x, brand_x)
graph_data, train_e, val_e, test_e = split_edges(graph_data)

model = FixedAdvancedModel(hidden_channels=256, out_channels=128, metadata=graph_data.metadata()).to(device)
model.load_state_dict(torch.load('best_advanced_model.pt', map_location=device))



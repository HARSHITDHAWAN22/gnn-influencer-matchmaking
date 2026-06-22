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



# ====================================================================
# SWEEP THRESHOLD ON VALIDATION SET
# ====================================================================

best_thresh, best_acc = 0.5, 0.0
for t in np.linspace(0.01, 0.99, 99):
    res = evaluate(model, graph_data, val_e, device, threshold=t)
    if res['accuracy'] > best_acc:
        best_acc, best_thresh = res['accuracy'], t

print(f"\nBest threshold found: {best_thresh:.2f} (Validation Accuracy: {best_acc:.4f})")

# ====================================================================
# FINAL EVALUATION
# ====================================================================

print("\n--- Train ---")
train_res = evaluate(model, graph_data, train_e, device, best_thresh)

print("\n--- Validation ---")
val_res = evaluate(model, graph_data, val_e, device, best_thresh)

print("\n--- Test ---")
test_res = evaluate(model, graph_data, test_e, device, best_thresh)

# ====================================================================
# SUMMARY TABLE
# ====================================================================

summary = pd.DataFrame([
    ['Train', train_res['auc'], train_res['ap'], train_res['accuracy'], train_res['precision'], train_res['recall'], train_res['f1']],
    ['Validation', val_res['auc'], val_res['ap'], val_res['accuracy'], val_res['precision'], val_res['recall'], val_res['f1']],
    ['Test', test_res['auc'], test_res['ap'], test_res['accuracy'], test_res['precision'], test_res['recall'], test_res['f1']]
], columns=['Split', 'AUC-ROC', 'AvgPrecision', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

print("\n" + "="*80)
print(summary.to_string(index=False))
print("="*80)


# ====================================================================
# VISUALIZATION SECTION
# ====================================================================

# --- 1️ Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
splits = ['Train', 'Validation', 'Test']
results = [train_res, val_res, test_res]

for ax, name, res in zip(axes, splits, results):
    cm = res['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{name} Confusion Matrix', fontsize=13, weight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')


# --- 2️ ROC Curves ---
plt.figure(figsize=(8, 6))
for name, res in zip(splits, results):
    fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("ROC Curves", fontsize=14, weight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# --- 3️ Precision–Recall Curves ---
plt.figure(figsize=(8, 6))
for name, res in zip(splits, results):
    precision, recall, _ = precision_recall_curve(res['y_true'], res['y_prob'])
    plt.plot(recall, precision, label=f"{name} (AP={res['ap']:.3f})")
plt.title("Precision–Recall Curves", fontsize=14, weight='bold')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.tight_layout()
plt.show()


 #--- ️4 Bar Chart of Final Metrics ---
metric_df = pd.DataFrame([
    ['Train', train_res['accuracy'], train_res['precision'], train_res['recall'], train_res['f1']],
    ['Validation', val_res['accuracy'], val_res['precision'], val_res['recall'], val_res['f1']],
    ['Test', test_res['accuracy'], test_res['precision'], test_res['recall'], test_res['f1']]
], columns=['Split', 'Accuracy', 'Precision', 'Recall', 'F1'])

metric_df.set_index('Split').plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Model Performance Comparison", fontsize=14, weight='bold')
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()




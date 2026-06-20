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


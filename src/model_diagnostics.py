# ============================================================================
# COMPLETE DIAGNOSIS + FIX FOR 60% ACCURACY
# Run this to understand WHY accuracy is low and HOW to fix it
# ============================================================================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

print("="*80)
print("DIAGNOSIS: WHY IS ACCURACY ONLY 60%?")
print("="*80)

# ============================================================================
# STEP 1: DIAGNOSE THE PROBLEM
# ============================================================================

@torch.no_grad()
def diagnose_model_issues(model, data, edge_index, device):
    """
    Detailed diagnosis of why accuracy is low.
    """
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTIC TESTS")
    print("="*80)

    model.eval()
    data = data.to(device)
    edge_index = edge_index.to(device)

    # Get predictions
    z_dict = model(data.x_dict, data.edge_index_dict)

    # Positive samples
    brand_emb = z_dict['brand'][edge_index[0]]
    inf_emb = z_dict['influencer'][edge_index[1]]
    pos_pred = torch.sigmoid(model.predict_match(brand_emb, inf_emb).squeeze())

    # Negative samples
    neg_edge_index = torch.stack([
        torch.randint(0, data['brand'].num_nodes, (edge_index.size(1),), device=device),
        torch.randint(0, data['influencer'].num_nodes, (edge_index.size(1),), device=device)
    ], dim=0)

    neg_brand_emb = z_dict['brand'][neg_edge_index[0]]
    neg_inf_emb = z_dict['influencer'][neg_edge_index[1]]
    neg_pred = torch.sigmoid(model.predict_match(neg_brand_emb, neg_inf_emb).squeeze())

    # Combine
    y_pred_prob = torch.cat([pos_pred, neg_pred]).cpu().numpy()
    y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])

    # ========================================================================
    # DIAGNOSTIC 1: Score Distribution Analysis
    # ========================================================================
    print("\n" + "-"*80)
    print("DIAGNOSTIC 1: SCORE DISTRIBUTION ANALYSIS")
    print("-"*80)

    pos_scores = y_pred_prob[y_true == 1]
    neg_scores = y_pred_prob[y_true == 0]

    print(f"\nPositive Samples (Should Match):")
    print(f"  Mean:   {pos_scores.mean():.4f}")
    print(f"  Median: {np.median(pos_scores):.4f}")
    print(f"  Std:    {pos_scores.std():.4f}")
    print(f"  Min:    {pos_scores.min():.4f}")
    print(f"  Max:    {pos_scores.max():.4f}")

    print(f"\nNegative Samples (Should NOT Match):")
    print(f"  Mean:   {neg_scores.mean():.4f}")
    print(f"  Median: {np.median(neg_scores):.4f}")
    print(f"  Std:    {neg_scores.std():.4f}")
    print(f"  Min:    {neg_scores.min():.4f}")
    print(f"  Max:    {neg_scores.max():.4f}")

    separation = pos_scores.mean() - neg_scores.mean()
    print(f"\nScore Separation: {separation:.4f}")

    # Diagnosis
    print("\n DIAGNOSIS:")
    if separation < 0.15:
        print("   CRITICAL: Scores are almost identical!")
        print("     Model CANNOT distinguish matches from non-matches")
        print("     ROOT CAUSE: Poor training or bad features")
    elif separation < 0.25:
        print("    WARNING: Separation is too low")
        print("     Model struggles to distinguish")
        print("     ROOT CAUSE: Weak training signal")
    else:
        print("  Separation is OK (but accuracy still low)")
        print("     ROOT CAUSE: Threshold or prediction issue")

    # ========================================================================
    # DIAGNOSTIC 2: Prediction Pattern Analysis
    # ========================================================================
    print("\n" + "-"*80)
    print("DIAGNOSTIC 2: PREDICTION PATTERN ANALYSIS")
    print("-"*80)

    # Count predictions at different thresholds
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_pred_prob > threshold).astype(int)
        num_positive_pred = y_pred.sum()
        num_negative_pred = len(y_pred) - num_positive_pred
        accuracy = (y_pred == y_true).mean()

        print(f"\nThreshold {threshold:.1f}:")
        print(f"  Predicted Match:     {num_positive_pred:6d} ({num_positive_pred/len(y_pred)*100:.1f}%)")
        print(f"  Predicted No-Match:  {num_negative_pred:6d} ({num_negative_pred/len(y_pred)*100:.1f}%)")
        print(f"  Accuracy:            {accuracy:.4f}")

    # Check if model is predicting all one class
    y_pred_05 = (y_pred_prob > 0.5).astype(int)
    unique_preds = np.unique(y_pred_05)

    print("\n DIAGNOSIS:")
    if len(unique_preds) == 1:
        print(f"   CRITICAL: Model predicts ONLY class {unique_preds[0]}!")
        if unique_preds[0] == 0:
            print("     All predictions = No Match")
            print("     ROOT CAUSE: Scores too low OR threshold too high")
        else:
            print("     All predictions = Match")
            print("     ROOT CAUSE: Scores too high OR threshold too low")
    else:
        print("   Model predicts both classes")



    # ========================================================================
    # DIAGNOSTIC 3: Confusion Matrix Analysis
    # ========================================================================
    print("\n" + "-"*80)
    print("DIAGNOSTIC 3: CONFUSION MATRIX ANALYSIS")
    print("-"*80)

    cm = confusion_matrix(y_true, y_pred_05)

    print(f"\n              Predicted")
    print(f"           No Match  |  Match")
    print(f"Actual  --------------------------")
    print(f"No Match   {cm[0,0]:6d}   | {cm[0,1]:6d}")
    print(f"Match      {cm[1,0]:6d}   | {cm[1,1]:6d}")

    tn, fp, fn, tp = cm.ravel()

    print(f"\nBreakdown:")
    print(f"  True Negatives:  {tn:,} ")
    print(f"  False Positives: {fp:,} ")
    print(f"  False Negatives: {fn:,} ")
    print(f"  True Positives:  {tp:,} ")

    print("\n📊 DIAGNOSIS:")
    if tp == 0:
        print("   CRITICAL: Model NEVER predicts Match correctly!")
        print("     ROOT CAUSE: Threshold too high or scores too low")
    elif fn > tp * 2:
        print("    WARNING: Model misses most true matches")
        print("     ROOT CAUSE: Low recall - threshold too high")
    elif fp > tn * 0.5:
        print("    WARNING: Too many false positives")
        print("     ROOT CAUSE: Low precision - threshold too low")
    else:
        print("   Predictions somewhat balanced")

    # ========================================================================
    # DIAGNOSTIC 4: Edge Quality Check
    # ========================================================================
    print("\n" + "-"*80)
    print("DIAGNOSTIC 4: TRAINING EDGE QUALITY")
    print("-"*80)

    num_edges = edge_index.size(1)
    num_brands = data['brand'].num_nodes
    num_influencers = data['influencer'].num_nodes

    print(f"\nGraph Statistics:")
    print(f"  Number of Brands:      {num_brands:,}")
    print(f"  Number of Influencers: {num_influencers:,}")
    print(f"  Number of Edges:       {num_edges:,}")
    print(f"  Avg Edges per Brand:   {num_edges/num_brands:.1f}")

    print("\n DIAGNOSIS:")
    if num_edges / num_brands < 5:
        print("    WARNING: Very few edges per brand")
        print("     ROOT CAUSE: Insufficient training signal")
    elif num_edges / num_brands > 100:
        print("    WARNING: Too many edges (possibly random)")
        print("     ROOT CAUSE: Noisy training data")
    else:
        print("   Edge count seems reasonable")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    issues = []

    if separation < 0.15:
        issues.append(" CRITICAL: No score separation")
    if len(unique_preds) == 1:
        issues.append(" CRITICAL: Predicting only one class")
    if tp == 0:
        issues.append(" CRITICAL: Zero true positives")
    if separation < 0.25:
        issues.append("  Low score separation")
    if num_edges / num_brands < 5:
        issues.append("  Too few training edges")

    if len(issues) == 0:
        print("\n No critical issues found")
        print("   Problem likely: threshold or minor tuning needed")
    else:
        print("\n ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")

    return {
        'pos_mean': pos_scores.mean(),
        'neg_mean': neg_scores.mean(),
        'separation': separation,
        'num_unique_preds': len(unique_preds),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'issues': issues
    }

# ============================================================================
# RUN DIAGNOSIS
# ============================================================================

print("\n🔍 Running comprehensive diagnosis on your model...")
diagnosis = diagnose_model_issues(model, graph_data, test_edge, device)

    

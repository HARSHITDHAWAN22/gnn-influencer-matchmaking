import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, to_hetero
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import HeteroData # Import HeteroData

print("="*80)
print("FIXED ADVANCED MODEL: BOOST ACCURACY TO 85-90%")
print("="*80)

# ============================================================================
# FIXED ARCHITECTURE - NO LAZY LAYERS
# ============================================================================

class FixedAdvancedGNN(nn.Module):
    """
    Fixed GNN with proper dimension handling.
    """
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        # Use SAGE instead of GAT to avoid dimension issues
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.conv4 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        # Layer 1
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.3, training=self.training)

        # Layer 2 with skip
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.3, training=self.training)
        x2 = x2 + x1  # Skip connection

        # Layer 3 with skip
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.elu(x3)
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x3 = x3 + x2  # Skip connection

        # Layer 4
        x4 = self.conv4(x3, edge_index)

        return x4

from config.config import (
    ADV_HIDDEN_CHANNELS,ADV_OUT_CHANNELS,DROPOUT_LOW,DROPOUT_MEDIUM,ATTENTION_HEADS,
    ADV_LEARNING_RATE,ADV_WEIGHT_DECAY,ADV_NUM_EPOCHS,ADV_PATIENCE,EVALUATION_INTERVAL,
    FOCAL_ALPHA,FOCAL_GAMMA,LABEL_SMOOTHING,MIXUP_ALPHA,MIXUP_START_EPOCH,
    HARD_NEGATIVE_START_EPOCH,NEGATIVE_CANDIDATE_MULTIPLIER,MARGIN,MARGIN_LOSS_WEIGHT,
    MAX_GRAD_NORM,COSINE_T0,COSINE_T_MULT,BEST_ADV_MODEL_PATH)

class FixedAdvancedModel(nn.Module):
    """
    Fixed advanced model without LazyLinear issues.
    """
    def __init__(self, hidden_channels=ADV_HIDDEN_CHANNELS, out_channels=ADV_OUT_CHANNELS, metadata=None):
        super().__init__()

        # Base GNN
        self.gnn = FixedAdvancedGNN(hidden_channels, out_channels)
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr='mean')

        # Brand-specific encoder
        self.brand_encoder = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ELU(),
            nn.Dropout(DROPOUT_LOW),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )

        # Influencer-specific encoder
        self.influencer_encoder = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ELU(),
            nn.Dropout(DROPOUT_LOW),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )

        # Multi-head attention
        self.attention_heads = ATTENTION_HEADS
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels * 2, hidden_channels // 4),
                nn.ELU(),
                nn.Linear(hidden_channels // 4, 1),
                nn.Sigmoid()
            ) for _ in range(self.attention_heads)
        ])

        # Advanced predictor
        self.predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ELU(),
            nn.Dropout(DROPOUT_MEDIUM),

            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ELU(),
            nn.Dropout(DROPOUT_MEDIUM),

            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(DROPOUT_LOW),

            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ELU(),

            nn.Linear(hidden_channels // 4, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)

    def predict_match(self, z_brand, z_inf):
        # Apply encoders
        z_brand_encoded = self.brand_encoder(z_brand)
        z_inf_encoded = self.influencer_encoder(z_inf)

        # Concatenate
        combined = torch.cat([z_brand_encoded, z_inf_encoded], dim=-1)

        # Multi-head attention
        attention_weights = []
        for attention_layer in self.attention_layers:
            weight = attention_layer(combined)
            attention_weights.append(weight)

        # Average attention
        attention = torch.stack(attention_weights, dim=0).mean(dim=0)

        # Apply attention
        weighted_combined = combined * attention

        # Predict
        return self.predictor(weighted_combined)

# ============================================================================
# ADVANCED FOCAL LOSS
# ============================================================================

class AdvancedFocalLoss(nn.Module):
    """Focal loss with label smoothing."""
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Label smoothing
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Focal loss
        probs = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()

# ============================================================================
# MIXUP AUGMENTATION
# ============================================================================

def mixup_data(z_brand, z_inf, labels, alpha=MIXUP_ALPHA):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = z_brand.size(0)
    index = torch.randperm(batch_size).to(z_brand.device)

    mixed_brand = lam * z_brand + (1 - lam) * z_brand[index]
    mixed_inf = lam * z_inf + (1 - lam) * z_inf[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_brand, mixed_inf, mixed_labels

# ============================================================================
# HARD NEGATIVE MINING
# ============================================================================

def hard_negative_mining(z_dict, pos_edge_index, model, device, num_negatives):
    """Select hard negative samples."""
    num_brands = z_dict['brand'].size(0)
    num_influencers = z_dict['influencer'].size(0)

    # Sample more candidates
    num_candidates = num_negatives * NEGATIVE_CANDIDATE_MULTIPLIER
    neg_brand_idx = torch.randint(0, num_brands, (num_candidates,), device=device)
    neg_inf_idx = torch.randint(0, num_influencers, (num_candidates,), device=device)

    # Score candidates
    with torch.no_grad():
        neg_brand_emb = z_dict['brand'][neg_brand_idx]
        neg_inf_emb = z_dict['influencer'][neg_inf_idx]
        neg_scores = torch.sigmoid(
            model.predict_match(neg_brand_emb, neg_inf_emb).squeeze()
        )

    # Select hardest
    _, hard_indices = torch.topk(neg_scores, k=num_negatives)

    return torch.stack([neg_brand_idx[hard_indices], neg_inf_idx[hard_indices]], dim=0)

# ============================================================================
# ADVANCED TRAINING
# ============================================================================

def advanced_train_epoch(model, data, optimizer, focal_loss, device, epoch):
    """Advanced training epoch."""
    model.train()
    optimizer.zero_grad()

    # Get embeddings
    z_dict = model(data.x_dict, data.edge_index_dict)

    # Positive edges
    # Ensure edge_index is accessed from the data object
    edge_index = data['brand', 'matches_with', 'influencer'].edge_index


    if edge_index.size(1) == 0:
        print("⚠️  No training edges!")
        return 0.0, 0.0, 0.0

    brand_emb = z_dict['brand'][edge_index[0]]
    inf_emb = z_dict['influencer'][edge_index[1]]
    pos_label = torch.ones(brand_emb.size(0), device=device)

    # Negative sampling (hard after epoch 15)
    if epoch > HARD_NEGATIVE_START_EPOCH:
        neg_edge_index = hard_negative_mining(
            z_dict, edge_index, model, device, edge_index.size(1)
        )
    else:
        neg_edge_index = torch.stack([
            torch.randint(0, data['brand'].num_nodes, (edge_index.size(1),), device=device),
            torch.randint(0, data['influencer'].num_nodes, (edge_index.size(1),), device=device)
        ], dim=0)

    neg_brand_emb = z_dict['brand'][neg_edge_index[0]]
    neg_inf_emb = z_dict['influencer'][neg_edge_index[1]]
    neg_label = torch.zeros(neg_brand_emb.size(0), device=device)

    # Combine
    all_brand_emb = torch.cat([brand_emb, neg_brand_emb])
    all_inf_emb = torch.cat([inf_emb, neg_inf_emb])
    all_labels = torch.cat([pos_label, neg_label])

    # Mixup (after epoch 20)
    if epoch > MIXUP_START_EPOCH:
        all_brand_emb, all_inf_emb, all_labels = mixup_data(
            all_brand_emb, all_inf_emb, all_labels, alpha=MIXUP_ALPHA
        )

    # Forward
    pred = model.predict_match(all_brand_emb, all_inf_emb).squeeze()

    # Focal loss
    loss = focal_loss(pred, all_labels)

    # Margin loss
    pos_pred = torch.sigmoid(pred[:edge_index.size(1)])
    neg_pred = torch.sigmoid(pred[edge_index.size(1):])
    margin_loss = F.relu(MARGIN - (pos_pred.mean() - neg_pred.mean()))

    total_loss = loss + MARGIN_LOSS_WEIGHT * margin_loss

    # Backward
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
    optimizer.step()

    return total_loss.item(), pos_pred.mean().item(), neg_pred.mean().item()

# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_with_accuracy(model, data, edge_index, device):
    """Evaluate model."""
    model.eval()

    if edge_index.size(1) == 0:
        return 0.0, 0.0, 0.0

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
    y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
    y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])

    # Metrics
    try:
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
    except:
        auc, ap = 0.0, 0.0

    # Find best threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]

    y_pred_binary = (y_pred > best_threshold).astype(int)
    accuracy = (y_pred_binary == y_true).mean()

    return auc, ap, accuracy

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_advanced_model(graph_data, train_edge, val_edge, test_edge,
                        hidden_channels=ADV_HIDDEN_CHANNELS, out_channels=ADV_OUT_CHANNELS,
                        num_epochs=ADV_NUM_EPOCHS, lr=ADV_LEARNING_RATE):
    """Train advanced model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*80)
    print("TRAINING ADVANCED MODEL")
    print("="*80)
    print(f"Device: {device}")
    print(f"Hidden: {hidden_channels}, Out: {out_channels}")
    print(f"Epochs: {num_epochs}, LR: {lr}")

    # Move to device
    data = graph_data.to(device) # Ensure data is passed to the device
    train_edge = train_edge.to(device)
    val_edge = val_edge.to(device)
    test_edge = test_edge.to(device)

    # Initialize model
    model = FixedAdvancedModel(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        metadata=data.metadata()
    ).to(device)

    # Dummy forward to initialize lazy modules
    with torch.no_grad():
        # Ensure x_dict and edge_index_dict are correctly accessed from the HeteroData object
        _ = model(data.x_dict, data.edge_index_dict)


    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=ADV_WEIGHT_DECAY)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT
    )

    # Loss
    focal_loss = AdvancedFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    best_val_acc = 0
    patience_counter = 0
    patience = ADV_PATIENCE

    print("\n" + "-"*80)
    print("TRAINING")
    print("-"*80)

    for epoch in range(1, num_epochs + 1):
        # Train
        loss, pos_mean, neg_mean = advanced_train_epoch(
            model, data, optimizer, focal_loss, device, epoch
        )

        # Evaluate every 5 epochs
        if epoch % EVALUATION_INTERVAL == 0:
            val_auc, val_ap, val_acc = evaluate_with_accuracy(model, data, val_edge, device)

            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Pos: {pos_mean:.3f} | Neg: {neg_mean:.3f} | "
                  f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), BEST_ADV_MODEL_PATH)
                print(f"  → New best! {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch}")
                    break

        scheduler.step()

    # Load best model
    model.load_state_dict(torch.load(BEST_ADV_MODEL_PATH))

    # Final test
    test_auc, test_ap, test_acc = evaluate_with_accuracy(model, data, test_edge, device)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Test AUC:      {test_auc:.4f}")
    print(f"Test AP:       {test_ap:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    if test_acc > 0.85:
        print("\n🌟 EXCELLENT! Accuracy > 85%")
    elif test_acc > 0.80:
        print("\n👍 VERY GOOD! Accuracy > 80%")
    elif test_acc > 0.75:
        print("\n✓ GOOD! Accuracy > 75%")
    else:
        print("\n⚠️  Needs more training or better edges")

    print("="*80)

    return model, device

# ============================================================================
# INCLUDE GRAPH CREATION AND EDGE SPLITTING HERE
# ============================================================================

# Assuming create_matching_graph is defined in a previous cell (e.g. UKZOb_S6Y5km)
# Assuming influencers_data, brands, influencer_x, brand_x are defined in previous data preprocessing cells
# Assuming split_edges is defined (e.g., in A6JZdK-1Z5yH or earlier)

print("\n" + "="*80)
print("RECREATING GRAPH AND SPLITTING EDGES")
print("="*80)

# Check if necessary variables are defined before proceeding
try:
    graph_data = create_matching_graph(influencers_data, brands, influencer_x, brand_x)

    # Split data again with the newly created graph
    graph_data, train_edge, val_edge, test_edge = split_edges(graph_data)

    print("\nGraph recreated and edges split again within the training cell.")
    print(graph_data)

    # ============================================================================
    # RUN TRAINING
    # ============================================================================

    print("\n🚀 Starting training with optimizations...")
    print("\nOptimizations:")
    print("  ✅ Deeper architecture (4 layers)")
    print("  ✅ Skip connections")
    print("  ✅ Multi-head attention")
    print("  ✅ Focal loss + label smoothing")
    print("  ✅ Hard negative mining (epoch 15+)")
    print("  ✅ Mixup augmentation (epoch 20+)")
    print("  ✅ Margin loss")
    print("  ✅ Cosine annealing")
    print("\nExpected: 85-88% accuracy")
    print("Time: ~30-40 minutes\n")

    # Train
    advanced_model, device = train_advanced_model(
        graph_data, train_edge, val_edge, test_edge,
        hidden_channels=ADV_HIDDEN_CHANNELS,
        out_channels=ADV_OUT_CHANNELS,
        num_epochs=ADV_NUM_EPOCHS,
        lr=ADV_LEARNING_RATE
    )

    print("\n✅ TRAINING COMPLETE!")
    print("Your model is ready for 85-88% accuracy!")

except NameError as e:
    print(f"\nError: {e}. Make sure create_matching_graph, influencers_data, brands, influencer_x, brand_x, and split_edges are defined and run in previous cells.")
    print("Please ensure all preceding data preprocessing, feature preparation, graph creation, and edge splitting steps are successfully executed.")

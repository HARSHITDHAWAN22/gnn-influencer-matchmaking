# ==================================================
# NEGATIVE SAMPLING FOR LINK PREDICTION
# ==================================================

def negative_sampling(num_brands, num_influencers, num_samples, device):
    """Generate negative samples for training."""
    neg_brand = torch.randint(0, num_brands, (num_samples,), device=device)
    neg_inf = torch.randint(0, num_influencers, (num_samples,), device=device)
    return torch.stack([neg_brand, neg_inf], dim=0)


# ==================================================
# TRAINING AND EVALUATION FUNCTIONS
# ==================================================

def train_epoch(model, data, optimizer, device):
    """Train model for one epoch."""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    z_dict = model(data.x_dict, data.edge_index_dict)

    # Get positive edges
    edge_index = data['brand', 'matches_with', 'influencer'].edge_index


# ==================================================
# MATCHMAKING GNN TRAINING PIPELINE
# ==================================================

def train_matchmaking_model(data, train_edge, val_edge, test_edge,
                            hidden_channels=128, out_channels=64,
                            num_epochs=100, lr=0.001):
    """Complete training pipeline."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"TRAINING MATCHMAKING GNN MODEL")
    print(f"{'='*80}")
    print(f"Device: {device}")

    # Move to device
    data = data.to(device)
    train_edge = train_edge.to(device)
    val_edge = val_edge.to(device)
    test_edge = test_edge.to(device)

    # Initialize model
    model = MatchmakingGNN(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        metadata=data.metadata()
    ).to(device)

    # Perform a dummy forward pass to initialize lazy layers
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Positive samples
    brand_emb = z_dict['brand'][edge_index[0]]
    inf_emb = z_dict['influencer'][edge_index[1]]
    pos_pred = model.predict_match(brand_emb, inf_emb).squeeze()
    pos_label = torch.ones(pos_pred.size(0), device=device)

    # Negative samples
    neg_edge_index = negative_sampling(
        data['brand'].num_nodes,
        data['influencer'].num_nodes,
        edge_index.size(1),
        device
    )

    neg_brand_emb = z_dict['brand'][neg_edge_index[0]]
    neg_inf_emb = z_dict['influencer'][neg_edge_index[1]]
    neg_pred = model.predict_match(neg_brand_emb, neg_inf_emb).squeeze()
    neg_label = torch.zeros(neg_pred.size(0), device=device)

    # Combine
    pred = torch.cat([pos_pred, neg_pred])
    label = torch.cat([pos_label, neg_label])

    # Loss
    loss = F.binary_cross_entropy_with_logits(pred, label)

    # Backward
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def evaluate(model, data, edge_index, device):
    """Evaluate model."""
    model.eval()

    # Forward pass
    z_dict = model(data.x_dict, data.edge_index_dict)

    # Positive samples
    brand_emb = z_dict['brand'][edge_index[0]]
    inf_emb = z_dict['influencer'][edge_index[1]]
    pos_pred = model.predict_match(brand_emb, inf_emb).squeeze()

    # Negative samples
    neg_edge_index = negative_sampling(
        data['brand'].num_nodes,
        data['influencer'].num_nodes,
        edge_index.size(1),
        device
    )

    neg_brand_emb = z_dict['brand'][neg_edge_index[0]]
    neg_inf_emb = z_dict['influencer'][neg_edge_index[1]]
    neg_pred = model.predict_match(neg_brand_emb, neg_inf_emb).squeeze()

    # Combine
    pred = torch.cat([pos_pred, neg_pred]).sigmoid().cpu().numpy()
    label = torch.cat([
        torch.ones(pos_pred.size(0)),
        torch.zeros(neg_pred.size(0))
    ]).cpu().numpy()

    # Metrics
    auc = roc_auc_score(label, pred)
    ap = average_precision_score(label, pred)

    return auc, ap

print("\n Training functions defined!")


# ==================================================
# TRAIN VALIDATION TEST EDGE SPLITTING
# ==================================================

def split_edges(data, val_ratio=0.1, test_ratio=0.2):
    """Split edges into train/val/test."""
    edge_index = data['brand', 'matches_with', 'influencer'].edge_index
    num_edges = edge_index.size(1)

    indices = torch.randperm(num_edges)

    val_size = int(num_edges * val_ratio)
    test_size = int(num_edges * test_ratio)
    train_size = num_edges - val_size - test_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

    train_edge = edge_index[:, train_idx]
    val_edge = edge_index[:, val_idx]
    test_edge = edge_index[:, test_idx]

    # Store in data object
    data['brand', 'matches_with', 'influencer'].edge_label_index = train_edge
    data['brand', 'matches_with', 'influencer'].edge_label = torch.ones(train_edge.size(1))


    # Add reverse edges for 'matched_by' relationship if not already present
    if ('influencer', 'matched_by', 'brand') not in data.edge_types:
        data['influencer', 'matched_by', 'brand'].edge_index = train_edge.flip(0)
        data['influencer', 'matched_by', 'brand'].edge_label_index = train_edge.flip(0)
        data['influencer', 'matched_by', 'brand'].edge_label = torch.ones(train_edge.size(1))

    print(f"\nEdge split:")
    print(f"  Train: {train_size}")
    print(f"  Val: {val_size}")
    print(f"  Test: {test_size}")

    return data, train_edge, val_edge, test_edge

# Split data
graph_data, train_edge, val_edge, test_edge = split_edges(graph_data)


# ==================================================
# MODEL TRAINING AND EARLY STOPPING
# ==================================================


    # Training loop
    best_val_auc = 0
    patience = 15
    patience_counter = 0

    print(f"\n{'='*80}")
    print("TRAINING PROGRESS")
    print(f"{'='*80}\n")

    for epoch in range(1, num_epochs + 1):
        # Train
        loss = train_epoch(model, data, optimizer, device)

        # Evaluate
        if epoch % 5 == 0:
            val_auc, val_ap = evaluate(model, data, val_edge, device)

            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_matchmaking_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch}")
                    break

    # Load best model
    model.load_state_dict(torch.load('best_matchmaking_model.pt'))
    test_auc, test_ap = evaluate(model, data, test_edge, device)

    print(f"\n{'='*80}")
    print("FINAL TEST RESULTS")
    print(f"{'='*80}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP:  {test_ap:.4f}")
    print(f"{'='*80}\n")

    return model, device

# Train the model
model, device = train_matchmaking_model(
    graph_data, train_edge, val_edge, test_edge,
    hidden_channels=128,
    out_channels=64,
    num_epochs=100,
    lr=0.001
)






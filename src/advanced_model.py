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

class FixedAdvancedModel(nn.Module):
    """
    Fixed advanced model without LazyLinear issues.
    """
    def __init__(self, hidden_channels=256, out_channels=128, metadata=None):
        super().__init__()

        # Base GNN
        self.gnn = FixedAdvancedGNN(hidden_channels, out_channels)
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr='mean')

        # Brand-specific encoder
        self.brand_encoder = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )

        # Influencer-specific encoder
        self.influencer_encoder = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )

        # Multi-head attention
        self.attention_heads = 4
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
            nn.Dropout(0.3),

            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(0.2),

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
    def __init__(self, alpha=0.25, gamma=2.5, label_smoothing=0.1):
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

def mixup_data(z_brand, z_inf, labels, alpha=0.2):
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
    num_candidates = num_negatives * 5
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



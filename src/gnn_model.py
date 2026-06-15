# ==================================================
# GRAPHSAGE BASED GNN ARCHITECTURE
# ==================================================

class HomogeneousGNN(nn.Module):
    """Base GNN model using GraphSAGE convolutions."""
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv3(x, edge_index)
        return x


# ==================================================
# HETEROGENEOUS MATCHMAKING GNN MODEL
# ==================================================

class MatchmakingGNN(nn.Module):
    """
    Complete GNN model for influencer-brand matchmaking.
    Learns embeddings that capture:
    - Language compatibility
    - Content type alignment
    - Engagement matching
    - Budget fit
    """
    def __init__(self, hidden_channels=128, out_channels=64, metadata=None):
        super().__init__()

        # Convert base model to heterogeneous
        self.gnn = to_hetero(HomogeneousGNN(hidden_channels, out_channels), metadata=metadata, aggr='mean')

        # Matching predictor head
        self.predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        """Get node embeddings."""
        z_dict = self.gnn(x_dict, edge_index_dict)
        return z_dict

    def predict_match(self, z_brand, z_influencer):
        """Predict match probability between brand and influencer."""
        # Concatenate embeddings
        edge_feat = torch.cat([z_brand, z_influencer], dim=-1)
        # Predict match score
        return self.predictor(edge_feat)

print("\n GNN Model architecture defined!")

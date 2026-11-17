"""GNN player-graph encoder for StatsBomb 360 freeze-frame data.

Complements src.models.set_transformer (attention-based, implicit relations)
with an explicit-graph alternative: nodes are visible players, edges are
inverse physical distance between them, following Bekkers & Sahasrabudhe
(2024)'s use of graph convolutions over player positions (NEXT_STEPS.md,
Tier 2 direction #6, "GNN Player Graph").
"""

import numpy as np
import torch
import torch.nn as nn

from src.models.set_transformer import PLAYER_FEATURE_DIM

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def build_distance_adjacency(players: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """players: (B, N, 5) with cols [x_norm, y_norm, teammate, actor, keeper].

    Returns (B, N, N) inverse-distance adjacency (closer players = larger
    weight), masked to real players only, with a zero diagonal (self
    contribution is handled separately by the GraphConv layer's W_self).
    """
    xy = players[:, :, :2].clone()
    xy[:, :, 0] = xy[:, :, 0] * PITCH_LENGTH
    xy[:, :, 1] = xy[:, :, 1] * PITCH_WIDTH

    diff = xy.unsqueeze(2) - xy.unsqueeze(1)  # (B, N, N, 2)
    dist = torch.norm(diff, dim=-1)  # (B, N, N), in yards
    adj = 1.0 / (1.0 + dist)

    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, N, N)
    eye = torch.eye(players.shape[1], device=players.device, dtype=torch.bool).unsqueeze(0)
    return adj * pair_mask.float() * (~eye).float()


class GraphConvLayer(nn.Module):
    """Simplest message-passing layer: each node aggregates distance-weighted
    neighbor features and combines them with its own via separate linear maps."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_self = nn.Linear(d_model, d_model)
        self.W_neigh = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neighbor_msg = torch.matmul(adj, x)
        return torch.relu(self.W_self(x) + self.W_neigh(neighbor_msg))


class GNNPlayerGraphEncoder(nn.Module):
    """Permutation-invariant encoder: distance-graph message passing + masked mean pooling."""

    def __init__(self, in_dim: int = PLAYER_FEATURE_DIM, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([GraphConvLayer(d_model) for _ in range(n_layers)])

    def forward(self, players: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        adj = build_distance_adjacency(players, mask)
        x = self.embed(players)
        for layer in self.layers:
            x = layer(x, adj)

        mask_f = mask.unsqueeze(-1).float()
        summed = (x * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / counts


class PassSuccessWithGNN(nn.Module):
    """Event features + GNN player-graph embedding -> pass-success logit."""

    def __init__(self, n_event_features: int, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.gnn = GNNPlayerGraphEncoder(PLAYER_FEATURE_DIM, d_model, n_layers)
        self.head = nn.Sequential(
            nn.Linear(n_event_features + d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, event_feats: torch.Tensor, players: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedding = self.gnn(players, mask)
        x = torch.cat([event_feats, embedding], dim=-1)
        return self.head(x).squeeze(-1)


def train_gnn_model(
    event_train: torch.Tensor, players_train: torch.Tensor, mask_train: torch.Tensor, y_train: torch.Tensor,
    event_valid: torch.Tensor, players_valid: torch.Tensor, mask_valid: torch.Tensor, y_valid: torch.Tensor,
    device: torch.device,
    n_epochs: int = 100,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
) -> PassSuccessWithGNN:
    from sklearn.metrics import roc_auc_score

    model = PassSuccessWithGNN(event_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(event_train, players_train, mask_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    event_valid_d = event_valid.to(device)
    players_valid_d = players_valid.to(device)
    mask_valid_d = mask_valid.to(device)
    y_valid_np = y_valid.numpy()

    best_val_auroc = -1.0
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        for eb, pb, mb, yb in loader:
            eb, pb, mb, yb = eb.to(device), pb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(eb, pb, mb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(event_valid_d, players_valid_d, mask_valid_d)).cpu().numpy()
        val_auroc = roc_auc_score(y_valid_np, val_prob)
        scheduler.step(val_auroc)
        print(f"  epoch {epoch + 1}/{n_epochs}  val_auroc={val_auroc:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"  early stopping (no improvement in {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model


def predict_gnn_model(
    model: PassSuccessWithGNN,
    event_feats: torch.Tensor,
    players: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(event_feats), batch_size):
            eb = event_feats[i:i + batch_size].to(device)
            pb = players[i:i + batch_size].to(device)
            mb = mask[i:i + batch_size].to(device)
            probs.append(torch.sigmoid(model(eb, pb, mb)).cpu().numpy())
    return np.concatenate(probs)


def extract_gnn_embeddings(
    model: PassSuccessWithGNN,
    players: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(players), batch_size):
            pb = players[i:i + batch_size].to(device)
            mb = mask[i:i + batch_size].to(device)
            embeddings.append(model.gnn(pb, mb).cpu().numpy())
    return np.concatenate(embeddings, axis=0)

"""Set Transformer encoder for StatsBomb 360 freeze-frame data.

A permutation-invariant, masked multi-head self-attention encoder over the set of
visible players at the moment of a pass, producing a fixed-size embedding that is
combined with hand-crafted event features for pass-success prediction. This is the
real-scale counterpart to the concept check in Desktop/set_playground.py.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.features.context_360_features import _parse_freeze_frame

MAX_PLAYERS = 22
PLAYER_FEATURE_DIM = 5  # x_norm, y_norm, teammate, actor, keeper


def freeze_frames_to_tensor(
    df: pd.DataFrame,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
    max_players: int = MAX_PLAYERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the freeze_frame column into a padded (n, max_players, 5) array
    and a (n, max_players) boolean mask (True = real player, False = padding)."""
    n = len(df)
    players = np.zeros((n, max_players, PLAYER_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((n, max_players), dtype=bool)

    for i, ff in enumerate(df["freeze_frame"].values):
        parsed = _parse_freeze_frame(ff)
        if not parsed:
            continue
        for j, p in enumerate(parsed[:max_players]):
            loc = p.get("location")
            if loc is None or len(loc) < 2:
                continue
            players[i, j, 0] = loc[0] / pitch_length
            players[i, j, 1] = loc[1] / pitch_width
            players[i, j, 2] = 1.0 if p.get("teammate") else 0.0
            players[i, j, 3] = 1.0 if p.get("actor") else 0.0
            players[i, j, 4] = 1.0 if p.get("keeper") else 0.0
            mask[i, j] = True

    return players, mask


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), mask: (B, N) bool, True = real player
        B, N, D = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        key_mask = mask[:, None, None, :]  # (B, 1, 1, N), broadcasts over query dim and heads
        scores = scores.masked_fill(~key_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.W_o(out)


class SetTransformerEncoder(nn.Module):
    """Permutation-invariant encoder: masked self-attention + masked mean pooling."""

    def __init__(self, in_dim: int = PLAYER_FEATURE_DIM, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([MaskedSelfAttention(d_model, n_heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, players: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(players)
        for attn, norm in zip(self.layers, self.norms):
            x = norm(x + attn(x, mask))

        mask_f = mask.unsqueeze(-1).float()
        summed = (x * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / counts  # masked mean pooling -> fixed-size, permutation-invariant


class PassSuccessWithSetTransformer(nn.Module):
    """Event features + Set Transformer freeze-frame embedding -> pass-success logit."""

    def __init__(self, n_event_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.set_encoder = SetTransformerEncoder(PLAYER_FEATURE_DIM, d_model, n_heads, n_layers)
        self.head = nn.Sequential(
            nn.Linear(n_event_features + d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, event_feats: torch.Tensor, players: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedding = self.set_encoder(players, mask)
        x = torch.cat([event_feats, embedding], dim=-1)
        return self.head(x).squeeze(-1)  # logits


def train_set_transformer(
    event_train: torch.Tensor, players_train: torch.Tensor, mask_train: torch.Tensor, y_train: torch.Tensor,
    event_valid: torch.Tensor, players_valid: torch.Tensor, mask_valid: torch.Tensor, y_valid: torch.Tensor,
    device: torch.device,
    n_epochs: int = 100,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
) -> PassSuccessWithSetTransformer:
    """Train with LR-on-plateau scheduling and early-stopping-by-best-validation-AUROC
    (mirrors this project's early_stopping_rounds convention for the boosted-tree models)."""
    from sklearn.metrics import roc_auc_score

    model = PassSuccessWithSetTransformer(event_train.shape[1]).to(device)
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
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch + 1}/{n_epochs}  val_auroc={val_auroc:.4f}  lr={current_lr:.2e}")

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


def extract_embeddings(
    model: PassSuccessWithSetTransformer,
    players: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Extract the pooled Set Transformer embedding (pre-head) for downstream use
    (e.g. concatenating with hand-crafted features for an LGBM model)."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(players), batch_size):
            pb = players[i:i + batch_size].to(device)
            mb = mask[i:i + batch_size].to(device)
            embeddings.append(model.set_encoder(pb, mb).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def predict_set_transformer(
    model: PassSuccessWithSetTransformer,
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

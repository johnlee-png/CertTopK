"""Sequence Transformer: encodes the K actions immediately preceding a pass
using order-aware (positionally-encoded) self-attention, testing whether
recent possession context improves pass-success prediction (NEXT_STEPS.md,
Tier 2 direction #8, "Sequence Transformer").

Unlike src.models.set_transformer (permutation-invariant over an unordered
player set), action *order* matters here, so a learned positional embedding
is added before the attention layers.
"""

import numpy as np
import torch
import torch.nn as nn

from src.features.sequence_context import SEQ_FEATURE_DIM, MAX_SEQ_LEN


class PositionalSelfAttention(nn.Module):
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
        B, N, D = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        key_mask = mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)  # rows with no history at all -> zero contribution
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.W_o(out)


class SequenceTransformerEncoder(nn.Module):
    def __init__(self, in_dim: int = SEQ_FEATURE_DIM, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.layers = nn.ModuleList([PositionalSelfAttention(d_model, n_heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(seq) + self.pos_embed
        for attn, norm in zip(self.layers, self.norms):
            x = norm(x + attn(x, mask))

        mask_f = mask.unsqueeze(-1).float()
        summed = (x * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / counts


class PassSuccessWithSequence(nn.Module):
    """Event features + pre-pass action-sequence embedding -> pass-success logit."""

    def __init__(self, n_event_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.seq_encoder = SequenceTransformerEncoder(SEQ_FEATURE_DIM, d_model, n_heads, n_layers, max_len)
        self.head = nn.Sequential(
            nn.Linear(n_event_features + d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, event_feats: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedding = self.seq_encoder(seq, mask)
        x = torch.cat([event_feats, embedding], dim=-1)
        return self.head(x).squeeze(-1)


def train_sequence_model(
    event_train, seq_train, mask_train, y_train,
    event_valid, seq_valid, mask_valid, y_valid,
    device: torch.device,
    n_epochs: int = 100,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
) -> PassSuccessWithSequence:
    from sklearn.metrics import roc_auc_score

    model = PassSuccessWithSequence(event_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(event_train, seq_train, mask_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    event_valid_d = event_valid.to(device)
    seq_valid_d = seq_valid.to(device)
    mask_valid_d = mask_valid.to(device)
    y_valid_np = y_valid.numpy()

    best_val_auroc = -1.0
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        for eb, sb, mb, yb in loader:
            eb, sb, mb, yb = eb.to(device), sb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(eb, sb, mb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(event_valid_d, seq_valid_d, mask_valid_d)).cpu().numpy()
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


def predict_sequence_model(model, event_feats, seq, mask, device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(event_feats), batch_size):
            eb = event_feats[i:i + batch_size].to(device)
            sb = seq[i:i + batch_size].to(device)
            mb = mask[i:i + batch_size].to(device)
            probs.append(torch.sigmoid(model(eb, sb, mb)).cpu().numpy())
    return np.concatenate(probs)


def extract_sequence_embeddings(model, seq, mask, device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(seq), batch_size):
            sb = seq[i:i + batch_size].to(device)
            mb = mask[i:i + batch_size].to(device)
            embeddings.append(model.seq_encoder(sb, mb).cpu().numpy())
    return np.concatenate(embeddings, axis=0)

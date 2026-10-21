
"""A compact PyTorch implementation of SASRec for implicit-feedback data."""

from __future__ import annotations

import copy
import random
from collections.abc import Hashable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

    Item index 0 is reserved for left padding. The causal mask ensures that a
    position can only attend to the prefix preceding it.
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_items = num_items
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.final_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        length = sequences.shape[1]
        positions = torch.arange(length, device=sequences.device).unsqueeze(0)
        x = self.item_embedding(sequences) + self.position_embedding(positions)
        x = self.dropout(x)
        padding_mask = sequences.eq(0)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=sequences.device),
            diagonal=1,
        )
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def forward(
        self, sequences: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(sequences)
        positive_logits = (features * self.item_embedding(positives)).sum(dim=-1)
        negative_logits = (features * self.item_embedding(negatives)).sum(dim=-1)
        return positive_logits, negative_logits

    def score_all_items(self, sequences: torch.Tensor) -> torch.Tensor:
        features = self.encode(sequences)[:, -1, :]
        return features @ self.item_embedding.weight[1:].T


class _SequenceDataset(Dataset):
    def __init__(
        self, sequences: list[list[int]], num_items: int, max_len: int, seed: int
    ) -> None:
        self.sequences = [sequence for sequence in sequences if len(sequence) >= 2]
        self.num_items = num_items
        self.max_len = max_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        history = self.sequences[index]
        sequence = np.zeros(self.max_len, dtype=np.int64)
        positive = np.zeros(self.max_len, dtype=np.int64)
        negative = np.zeros(self.max_len, dtype=np.int64)
        seen = set(history)
        next_item = history[-1]
        cursor = self.max_len - 1
        for item in reversed(history[:-1]):
            sequence[cursor] = item
            positive[cursor] = next_item
            if len(seen) < self.num_items:
                sampled = self.rng.randint(1, self.num_items)
                while sampled in seen:
                    sampled = self.rng.randint(1, self.num_items)
                negative[cursor] = sampled
            next_item = item
            cursor -= 1
            if cursor < 0:
                break
        return tuple(torch.from_numpy(x) for x in (sequence, positive, negative))


@dataclass
class SASRecRecommender:
    model: SASRec
    item_to_index: dict[Hashable, int]
    index_to_item: dict[int, Hashable]
    user_sequences: dict[Hashable, list[int]]
    device: torch.device

    def recommend(self, user: Hashable, k: int = 10, filter_seen: bool = True) -> list:
        sequence = self.user_sequences.get(user, [])
        if not sequence or k <= 0:
            return []
        encoded = np.zeros(self.model.max_len, dtype=np.int64)
        tail = sequence[-self.model.max_len :]
        encoded[-len(tail) :] = tail
        inputs = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            scores = self.model.score_all_items(inputs).squeeze(0)
            if filter_seen:
                seen = torch.tensor(sorted(set(sequence)), device=self.device) - 1
                scores[seen] = -torch.inf
            available = torch.isfinite(scores).sum().item()
            top_indices = torch.topk(scores, min(k, available)).indices.add(1).tolist()
        return [self.index_to_item[index] for index in top_indices]

    def recommend_many(
        self,
        users: Iterable[Hashable],
        k: int = 10,
        filter_seen: bool = True,
        batch_size: int = 512,
    ) -> dict[Hashable, list]:
        """Generate recommendations in batches (substantially faster for evaluation)."""
        if k <= 0:
            raise ValueError("k must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        valid_users = [user for user in users if self.user_sequences.get(user)]
        recommendations: dict[Hashable, list] = {}
        self.model.eval()
        for start in range(0, len(valid_users), batch_size):
            batch_users = valid_users[start : start + batch_size]
            encoded = np.zeros((len(batch_users), self.model.max_len), dtype=np.int64)
            for row, user in enumerate(batch_users):
                tail = self.user_sequences[user][-self.model.max_len :]
                encoded[row, -len(tail) :] = tail
            inputs = torch.from_numpy(encoded).to(self.device)
            with torch.no_grad():
                scores = self.model.score_all_items(inputs)
                if filter_seen:
                    for row, user in enumerate(batch_users):
                        seen = torch.tensor(
                            sorted(set(self.user_sequences[user])), device=self.device
                        ) - 1
                        scores[row, seen] = -torch.inf
                count = min(k, scores.shape[1])
                top_indices = torch.topk(scores, count, dim=1).indices.add(1).cpu().tolist()
            for user, indices in zip(batch_users, top_indices, strict=True):
                seen = set(self.user_sequences[user]) if filter_seen else set()
                recommendations[user] = [
                    self.index_to_item[index] for index in indices if index not in seen
                ]
        return recommendations


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_sasrec(
    train: pd.DataFrame,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    max_len: int = 50,
    hidden_dim: int = 64,
    num_blocks: int = 2,
    num_heads: int = 1,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> tuple[SASRecRecommender, list[float]]:
    """Train SASRec from columns ``user``, ``item`` and ``timestamp``."""
    required = {"user", "item", "timestamp"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if train.empty:
        raise ValueError("Training data is empty")
    _set_seed(seed)
    items = train["item"].drop_duplicates().tolist()
    item_to_index = {item: index + 1 for index, item in enumerate(items)}
    index_to_item = {index: item for item, index in item_to_index.items()}
    ordered = train.sort_values(["user", "timestamp"], kind="stable")
    raw_sequences = ordered.groupby("user", sort=False)["item"].apply(list).to_dict()
    user_sequences = {
        user: [item_to_index[item] for item in sequence]
        for user, sequence in raw_sequences.items()
    }
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(selected_device)
    model = SASRec(
        len(items), max_len, hidden_dim, num_blocks, num_heads, dropout
    ).to(torch_device)
    dataset = _SequenceDataset(list(user_sequences.values()), len(items), max_len, seed)
    if not dataset:
        raise ValueError("SASRec needs at least one user with two training interactions")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for sequences, positives, negatives in loader:
            sequences = sequences.to(torch_device)
            positives = positives.to(torch_device)
            negatives = negatives.to(torch_device)
            positive_mask = positives.ne(0)
            negative_mask = negatives.ne(0)
            positive_logits, negative_logits = model(sequences, positives, negatives)
            loss = criterion(
                positive_logits[positive_mask], torch.ones_like(positive_logits[positive_mask])
            )
            if negative_mask.any():
                loss += criterion(
                    negative_logits[negative_mask],
                    torch.zeros_like(negative_logits[negative_mask]),
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        losses.append(total_loss / batches)
        print(f"Epoch {epoch + 1:02d}/{epochs}: loss={losses[-1]:.4f}")
    return SASRecRecommender(
        model, item_to_index, index_to_item, user_sequences, torch_device
    ), losses


def precision_recall_at_k(
    recommender: SASRecRecommender,
    test: pd.DataFrame,
    users: Iterable[Hashable] | None = None,
    k: int = 10,
    filter_seen: bool = True,
) -> tuple[float, float, int]:
    """Macro Precision@K and Recall@K, matching the existing notebook metrics."""
    if k <= 0:
        raise ValueError("k must be positive")
    actual_by_user = test.groupby("user")["item"].apply(set).to_dict()
    evaluation_users = (
        list(users)
        if users is not None
        else list(set(recommender.user_sequences).intersection(actual_by_user))
    )
    precisions: list[float] = []
    recalls: list[float] = []
    predictions = recommender.recommend_many(evaluation_users, k=k, filter_seen=filter_seen)
    for user in evaluation_users:
        actual = actual_by_user.get(user, set())
        if not actual:
            continue
        predicted = predictions.get(user, [])
        hits = len(set(predicted).intersection(actual))
        precisions.append(hits / k)
        recalls.append(hits / len(actual))
    if not precisions:
        raise ValueError("No users occur in both training and test data")
    return float(np.mean(precisions)), float(np.mean(recalls)), len(precisions)

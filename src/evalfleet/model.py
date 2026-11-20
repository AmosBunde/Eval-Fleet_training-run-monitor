"""A small decoder-only transformer. Nothing unusual: pre-norm blocks, learned positions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_len: int = 24
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff1 = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.ff2 = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.n_heads, d // self.n_heads).unbind(2)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x + self.proj(att.transpose(1, 2).reshape(b, t, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(t, device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))

    @torch.no_grad()
    def greedy(self, ids: torch.Tensor, n_new: int, eos_id: int) -> torch.Tensor:
        """Greedy continuation of a batch of equal-length prompts. Returns the new tokens."""
        out = []
        done = torch.zeros(ids.shape[0], dtype=torch.bool)
        for _ in range(n_new):
            nxt = self(ids)[:, -1].argmax(-1)
            out.append(nxt)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            done |= nxt == eos_id
            if bool(done.all()):
                break
        return torch.stack(out, dim=1)

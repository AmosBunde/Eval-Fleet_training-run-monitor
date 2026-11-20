"""Train one small model on the curriculum, saving checkpoints at fixed steps."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .faults import FaultSpec
from .model import ModelConfig, TinyTransformer
from .skills import EOS, MAX_LEN, PAD, VOCAB_SIZE, build_item_sets, encode


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 2400
    ckpt_every: int = 80
    batch_size: int = 64
    lr: float = 2e-3
    warmup: int = 100
    final_lr_frac: float = 0.2
    weight_decay: float = 0.01
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4

    @property
    def ckpt_steps(self) -> list[int]:
        return list(range(self.ckpt_every, self.steps + 1, self.ckpt_every))

    def model_config(self) -> ModelConfig:
        return ModelConfig(VOCAB_SIZE, MAX_LEN, self.d_model, self.n_layers, self.n_heads)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    seed: int
    fault: FaultSpec = field(default_factory=FaultSpec)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RunSpec":
        return RunSpec(d["run_id"], d["seed"], FaultSpec(**d["fault"]), TrainConfig(**d["train"]))


def base_lr(cfg: TrainConfig, step: int) -> float:
    """Linear warm-up, then cosine decay to ``final_lr_frac`` of the peak."""
    if step <= cfg.warmup:
        return cfg.lr * step / cfg.warmup
    frac = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (cfg.final_lr_frac + (1 - cfg.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def make_batch(spec: RunSpec, step: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids and a mask over the positions whose next token is part of the answer."""
    items = build_item_sets()
    pool = spec.fault.skill_pool(step)
    rows, starts = [], []
    for _ in range(spec.train.batch_size):
        skill = pool[rng.randrange(len(pool))]
        prompt, answer = spec.fault.training_example(
            skill, step, rng, items.held_out, items.main[skill.name]
        )
        rows.append(encode(prompt + answer + EOS))
        starts.append(len(prompt))
    width = max(len(r) for r in rows)
    ids = np.full((len(rows), width), PAD, dtype=np.int64)
    mask = np.zeros((len(rows), width - 1), dtype=bool)
    for i, (r, s) in enumerate(zip(rows, starts)):
        ids[i, : len(r)] = r
        mask[i, s - 1 : len(r) - 1] = True
    return torch.from_numpy(ids), torch.from_numpy(mask)


def train_run(spec: RunSpec, ckpt_dir: Path, log=print) -> dict:
    """Train ``spec`` and write ``step_<n>.pt`` files plus ``run.json`` into ``ckpt_dir``."""
    cfg = spec.train
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(spec.seed)
    rng = random.Random(spec.seed)
    model = TinyTransformer(cfg.model_config())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    losses: list[float] = []
    t0 = time.time()
    running = 0.0
    for step in range(1, cfg.steps + 1):
        lr = base_lr(cfg, step) * spec.fault.lr_multiplier(step)
        for group in opt.param_groups:
            group["lr"] = lr
        ids, mask = make_batch(spec, step, rng)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits[mask], ids[:, 1:][mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item()
        if step % cfg.ckpt_every == 0:
            losses.append(running / cfg.ckpt_every)
            running = 0.0
            torch.save(model.state_dict(), ckpt_dir / f"step_{step:06d}.pt")
    meta = {
        "spec": spec.to_dict(),
        "ckpt_steps": cfg.ckpt_steps,
        "train_loss": losses,
        "train_seconds": round(time.time() - t0, 1),
    }
    (ckpt_dir / "run.json").write_text(json.dumps(meta, indent=1))
    log(f"[train] {spec.run_id}: {cfg.steps} steps in {meta['train_seconds']}s, final loss {losses[-1]:.3f}")
    return meta

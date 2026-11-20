"""Evaluate every checkpoint of a run on the eval fleet and keep item-level results.

The result file is a compressed ``.npz`` with bit-packed arrays of shape
``[checkpoint, skill, set, item]`` (set 0 is main, set 1 is twin):

* ``score``   - what the production grader reported (this is what the monitor watches);
* ``regrade`` - what the reference grader says about the same raw outputs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .faults import FaultSpec
from .grader import production_grade, reference_grade
from .model import TinyTransformer
from .skills import (
    EOS,
    MAX_NEW_TOKENS,
    PAD,
    SKILL_NAMES,
    SKILLS,
    STOI,
    build_item_sets,
    decode,
    encode,
)
from .train import RunSpec


@lru_cache(maxsize=1)
def _flat_items():
    sets = build_item_sets()
    flat = []
    for k, skill in enumerate(SKILLS):
        for s, group in enumerate((sets.main, sets.twin)):
            for i, item in enumerate(group[skill.name]):
                flat.append((k, s, i, item))
    return flat


@lru_cache(maxsize=1)
def _eval_batch() -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Every eval item as ``prompt + answer + EOS``, padded, with each answer's span."""
    flat = _flat_items()
    rows = [encode(item.prompt + item.answer + EOS) for _, _, _, item in flat]
    ids = np.full((len(rows), max(len(r) for r in rows)), PAD, dtype=np.int64)
    spans = []
    for j, (r, (_, _, _, item)) in enumerate(zip(rows, flat)):
        ids[j, : len(r)] = r
        spans.append((len(item.prompt), len(r)))
    return torch.from_numpy(ids), spans


@torch.no_grad()
def raw_outputs(model: TinyTransformer) -> list[str]:
    """The model's greedy answer for every eval item, up to its first wrong character.

    One teacher-forced pass replaces step-by-step decoding. Greedy decoding reproduces the
    answer exactly when, and only when, the most likely next character is right at every
    position given the correct prefix. So the greedy output and the teacher-forced
    predictions agree up to and including the first wrong character, and both graders
    depend only on that part. ``tests/test_testbed.py`` checks this against real decoding.
    """
    ids, spans = _eval_batch()
    pred = model(ids[:, :-1]).argmax(-1).numpy()
    target = ids.numpy()[:, 1:]
    outs = []
    for j, (start, end) in enumerate(spans):
        p, t = pred[j, start - 1 : end - 1], target[j, start - 1 : end - 1]
        wrong = np.nonzero(p != t)[0]
        cut = int(wrong[0]) + 1 if wrong.size else len(p)
        outs.append(decode(p[:cut]))
    return outs


@torch.no_grad()
def greedy_outputs(model: TinyTransformer, prompts: list[str]) -> list[str]:
    """Plain greedy decoding, one prompt at a time. Slow; used for spot checks and tests."""
    outs = []
    for prompt in prompts:
        new = model.greedy(torch.tensor([encode(prompt)]), MAX_NEW_TOKENS, STOI[EOS])[0]
        outs.append(decode(new))
    return outs


def grade_checkpoint(outs: list[str], fault: FaultSpec, step: int) -> tuple[np.ndarray, np.ndarray]:
    flat = _flat_items()
    n = len(build_item_sets().main[SKILL_NAMES[0]])
    score = np.zeros((len(SKILLS), 2, n), dtype=bool)
    regrade = np.zeros_like(score)
    for (k, s, i, item), raw in zip(flat, outs):
        budget = fault.answer_budget(SKILLS[k], step)
        score[k, s, i] = production_grade(raw, item.answer, budget)
        regrade[k, s, i] = reference_grade(raw, item.answer)
    return score, regrade


def evaluate_run(ckpt_dir: Path, out_path: Path, log=print) -> Path:
    """Evaluate all checkpoints in ``ckpt_dir`` and write the item-level result file."""
    meta = json.loads((ckpt_dir / "run.json").read_text())
    spec = RunSpec.from_dict(meta["spec"])
    torch.set_num_threads(1)
    model = TinyTransformer(spec.train.model_config()).eval()
    t0 = time.time()
    scores, regrades = [], []
    for step in meta["ckpt_steps"]:
        model.load_state_dict(torch.load(ckpt_dir / f"step_{step:06d}.pt"))
        s, r = grade_checkpoint(raw_outputs(model), spec.fault, step)
        scores.append(s)
        regrades.append(r)
    meta["eval_seconds"] = round(time.time() - t0, 1)
    save_results(out_path, np.stack(scores), np.stack(regrades), meta)
    log(f"[eval]  {spec.run_id}: {len(scores)} checkpoints in {meta['eval_seconds']}s")
    return out_path


def save_results(path: Path, score: np.ndarray, regrade: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        score=np.packbits(score, axis=-1),
        regrade=np.packbits(regrade, axis=-1),
        n_items=np.int64(score.shape[-1]),
        steps=np.asarray(meta["ckpt_steps"], dtype=np.int64),
        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    )
    tmp.rename(path)  # a result file either exists whole or not at all


@dataclass
class RunResult:
    """Item-level results of one run. ``score[t, k, s, i]`` with s=0 main, s=1 twin."""

    run_id: str
    steps: np.ndarray
    score: np.ndarray
    regrade: np.ndarray
    fault: FaultSpec
    meta: dict
    skills: tuple[str, ...] = SKILL_NAMES

    @property
    def onset_index(self) -> int | None:
        """Index of the last clean checkpoint, or None for a clean run."""
        if self.fault.kind == "clean":
            return None
        return int(np.searchsorted(self.steps, self.fault.onset_step))


def load_results(path: Path) -> RunResult:
    with np.load(path) as z:
        n = int(z["n_items"])
        meta = json.loads(bytes(z["meta"]).decode())
        score = np.unpackbits(z["score"], axis=-1, count=n).astype(bool)
        regrade = np.unpackbits(z["regrade"], axis=-1, count=n).astype(bool)
        steps = z["steps"].copy()
    spec = RunSpec.from_dict(meta["spec"])
    return RunResult(spec.run_id, steps, score, regrade, spec.fault, meta)

"""The experiment matrix and a resumable runner.

A run is finished when ``results/runs/<run_id>.npz`` exists. The runner skips finished
runs, so it can be stopped and restarted. Model checkpoints go to ``results/checkpoints``
(not in git) and are deleted once a run has been evaluated unless asked otherwise.
"""

from __future__ import annotations

import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .evaluate import evaluate_run
from .faults import GRADER_BUG_SKILLS, FaultSpec
from .train import RunSpec, TrainConfig, train_run, validate_run_id

# Faults start after one of these checkpoints (index into the 30 checkpoints of a run).
ONSET_INDICES = (14, 17, 20, 15, 18, 16, 19)
REGRESSION_SKILLS = ("copy", "reverse", "sort", "depth", "max", "index")
CONTAMINATION_SKILLS = ("add", "modadd")
CORRUPT_RATES = (1.0, 0.5, 0.25)  # share of the skill's training targets made wrong
CONTAMINATION_RATES = (1.0, 0.5)  # share of the skill's training examples that are eval items

# How many runs of each kind. Seeds are 1000 + position in the matrix; pilot runs used 900-949.
MATRIX_COUNTS = (
    ("clean", 18),
    ("corrupt", 6),
    ("remove", 6),
    ("lr_spike", 6),
    ("contaminate", 6),
    ("grader_bug", 6),
)


def build_matrix(train: TrainConfig = TrainConfig()) -> list[RunSpec]:
    specs: list[RunSpec] = []
    steps = train.ckpt_steps
    for kind, count in MATRIX_COUNTS:
        for i in range(count):
            seed = 1000 + len(specs)
            onset = steps[ONSET_INDICES[i % len(ONSET_INDICES)]]
            if kind == "clean":
                fault = FaultSpec()
            elif kind == "corrupt":
                rate = CORRUPT_RATES[i % len(CORRUPT_RATES)]
                fault = FaultSpec(kind, REGRESSION_SKILLS[i % len(REGRESSION_SKILLS)], onset, rate=rate)
            elif kind == "remove":
                fault = FaultSpec(kind, REGRESSION_SKILLS[i % len(REGRESSION_SKILLS)], onset)
            elif kind == "lr_spike":
                fault = FaultSpec(kind, None, onset, lr_factor=10.0, lr_steps=40)
            elif kind == "contaminate":
                rate = CONTAMINATION_RATES[(i // 2) % len(CONTAMINATION_RATES)]
                fault = FaultSpec(kind, CONTAMINATION_SKILLS[i % len(CONTAMINATION_SKILLS)], onset, rate=rate)
            else:
                fault = FaultSpec(kind, GRADER_BUG_SKILLS[i % len(GRADER_BUG_SKILLS)], onset)
            specs.append(RunSpec(f"{kind}-{i:02d}", seed, fault, train))
    return specs


def result_path(results_dir: Path, run_id: str) -> Path:
    return results_dir / "runs" / f"{validate_run_id(run_id)}.npz"


def run_one(spec: RunSpec, results_dir: Path, keep_checkpoints: bool = False) -> str:
    """Train and evaluate one run unless its result file already exists."""
    out = result_path(results_dir, spec.run_id)
    if out.exists():
        return f"[skip]  {spec.run_id}: result exists"
    ckpt_dir = results_dir / "checkpoints" / validate_run_id(spec.run_id)
    t0 = time.time()
    if not (ckpt_dir / "run.json").exists():
        train_run(spec, ckpt_dir, log=lambda *_: None)
    evaluate_run(ckpt_dir, out, log=lambda *_: None)
    if not keep_checkpoints:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return f"[done]  {spec.run_id}: {time.time() - t0:.0f}s"


def run_matrix(
    results_dir: Path,
    workers: int = 3,
    only: str | None = None,
    keep_checkpoints: bool = False,
    log=print,
) -> None:
    specs = [s for s in build_matrix() if only is None or s.run_id.startswith(only)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, s, results_dir, keep_checkpoints) for s in specs]
        for i, fut in enumerate(as_completed(futures), 1):
            log(f"{fut.result()}  ({i}/{len(specs)}, {time.time() - t0:.0f}s elapsed)")

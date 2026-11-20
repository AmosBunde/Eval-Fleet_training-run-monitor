"""Fault injection. Each fault has a known onset step, so detection can be scored.

Two families:

* real regressions change the model: ``corrupt`` (a skill's training targets become wrong),
  ``remove`` (a skill's training data stops) and ``lr_spike`` (the learning rate jumps for a
  short window);
* measurement faults leave the model's real skill alone and break the eval:
  ``contaminate`` (main eval items leak into the training data, so the main score rises
  without the skill generalising) and ``grader_bug`` (the harness starts cutting answers
  short, so correct long answers are marked wrong).

The onset step is always a checkpoint step. The checkpoint saved at the onset step is the
last clean one; every later checkpoint is affected.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .skills import SKILL_BY_NAME, SKILLS, Item, Skill, sample_training_example

REAL_REGRESSIONS = ("corrupt", "remove", "lr_spike")
MEASUREMENT_FAULTS = ("contaminate", "grader_bug")
FAULT_KINDS = ("clean",) + REAL_REGRESSIONS + MEASUREMENT_FAULTS

# Skills whose answers vary in length, so a shortened generation budget hits only some items.
GRADER_BUG_SKILLS = ("copy", "reverse", "sort")


@dataclass(frozen=True)
class FaultSpec:
    kind: str = "clean"
    skill: str | None = None  # the affected skill; None for clean and lr_spike
    onset_step: int | None = None
    rate: float = 1.0  # corrupt: share of targets replaced; contaminate: share of leaked examples
    lr_factor: float = 1.0  # lr_spike: multiplier during the spike
    lr_steps: int = 0  # lr_spike: length of the spike in steps

    def __post_init__(self):
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}")
        if self.kind != "clean" and self.onset_step is None:
            raise ValueError("a fault needs an onset step")
        if self.kind in ("corrupt", "remove", "contaminate", "grader_bug"):
            if self.skill not in SKILL_BY_NAME:
                raise ValueError(f"fault {self.kind} needs a valid skill, got {self.skill!r}")
        if self.kind == "grader_bug" and self.skill not in GRADER_BUG_SKILLS:
            raise ValueError("grader_bug needs a skill with variable-length answers")

    def to_dict(self) -> dict:
        return asdict(self)

    def active(self, step: int) -> bool:
        """True for training steps (1-based) and checkpoints after the onset."""
        return self.kind != "clean" and step > self.onset_step

    # --- training-time hooks -------------------------------------------------------------

    def skill_pool(self, step: int) -> tuple[Skill, ...]:
        if self.kind == "remove" and self.active(step):
            return tuple(s for s in SKILLS if s.name != self.skill)
        return SKILLS

    def lr_multiplier(self, step: int) -> float:
        if self.kind == "lr_spike" and self.onset_step < step <= self.onset_step + self.lr_steps:
            return self.lr_factor
        return 1.0

    def training_example(
        self,
        skill: Skill,
        step: int,
        rng: random.Random,
        held_out: frozenset[str],
        main_items: tuple[Item, ...],
    ) -> tuple[str, str]:
        """One training (prompt, answer) pair for ``skill`` at ``step``, with the fault applied."""
        hit = self.active(step) and skill.name == self.skill
        if hit and self.kind == "contaminate" and rng.random() < self.rate:
            item = main_items[rng.randrange(len(main_items))]
            return item.prompt, item.answer
        prompt, answer = sample_training_example(skill, rng, held_out)
        if hit and self.kind == "corrupt" and rng.random() < self.rate:
            # Pair the prompt with the answer to a different input: well-formed but wrong.
            for _ in range(20):
                _, other = sample_training_example(skill, rng, held_out)
                if other != answer:
                    return prompt, other
        return prompt, answer

    # --- eval-time hook ------------------------------------------------------------------

    def answer_budget(self, skill: Skill, step: int) -> int:
        """How many answer characters the harness keeps for ``skill`` at checkpoint ``step``."""
        if self.kind == "grader_bug" and self.active(step) and skill.name == self.skill:
            return skill.max_answer_len - 1
        return skill.max_answer_len

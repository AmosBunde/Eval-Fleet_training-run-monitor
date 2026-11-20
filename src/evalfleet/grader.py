"""Grading. Two independent paths over the same raw model output.

``production_grade`` is what the eval harness reports and what the monitor watches. It keeps
only ``budget`` characters of the answer, as a harness with a generation limit would. When
the budget is set one character too low (the injected grader bug), the longest correct
answers are cut and marked wrong.

``reference_grade`` is a second, simpler grader that never truncates. Comparing the two on
the same outputs is the grader self-consistency check.
"""

from __future__ import annotations

from .skills import EOS


def production_grade(raw: str, answer: str, budget: int) -> bool:
    kept = raw[:budget]  # the harness's generation limit
    cut = kept.find(EOS)
    if cut >= 0:
        kept = kept[:cut]
    return kept.strip() == answer.strip()


def reference_grade(raw: str, answer: str) -> bool:
    return raw.split(EOS, 1)[0] == answer

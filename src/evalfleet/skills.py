"""The multi-skill curriculum: eight small algorithmic tasks written as text.

Every example is a line ``<tag>:<input>=<answer>``. The model is trained to continue the
line after ``=``. Each skill has a generator (used for training data) and two fixed item
sets, ``main`` and ``twin``, drawn from the same distribution. The twin set is never used
for anything except checking the main set, which is what makes contamination detectable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

ALPHABET = "0123456789()[]+|:=$crsdxiam"
PAD = 0
STOI = {ch: i + 1 for i, ch in enumerate(ALPHABET)}
ITOS = {i: ch for ch, i in STOI.items()}
VOCAB_SIZE = len(ALPHABET) + 1
EOS = "$"
MAX_LEN = 24  # longest encoded line, with room to spare

ITEM_SEED = 20261120  # the eval items are the same for every run
N_ITEMS = 200  # items per set (main and twin) per skill


def _digits(rng: random.Random, lo: int, hi: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(rng.randint(lo, hi)))


def _copy(rng: random.Random) -> tuple[str, str]:
    s = _digits(rng, 3, 6)
    return s, s


def _reverse(rng: random.Random) -> tuple[str, str]:
    s = _digits(rng, 3, 6)
    return s, s[::-1]


def _sort(rng: random.Random) -> tuple[str, str]:
    s = _digits(rng, 3, 6)
    return s, "".join(sorted(s))


def _add(rng: random.Random) -> tuple[str, str]:
    a = int(_digits(rng, 1, 2))
    b = int(_digits(rng, 1, 2))
    return f"{a}+{b}", str(a + b)


def _depth(rng: random.Random) -> tuple[str, str]:
    """A balanced string of two bracket types; the answer is its deepest nesting."""
    n_pairs = rng.randint(3, 6)
    out: list[str] = []
    stack: list[str] = []
    opened = 0
    deepest = 0
    while opened < n_pairs or stack:
        can_open = opened < n_pairs
        if can_open and (not stack or rng.random() < 0.55):
            kind = rng.choice("([")
            out.append(kind)
            stack.append(")" if kind == "(" else "]")
            opened += 1
            deepest = max(deepest, len(stack))
        else:
            out.append(stack.pop())
    return "".join(out), str(deepest)


def _max(rng: random.Random) -> tuple[str, str]:
    s = _digits(rng, 4, 7)
    return s, max(s)


def _index(rng: random.Random) -> tuple[str, str]:
    """The character at a given (1-based) position of a string."""
    s = _digits(rng, 4, 7)
    k = rng.randint(1, len(s))
    return f"{k}|{s}", s[k - 1]


def _modadd(rng: random.Random) -> tuple[str, str]:
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    return f"{a}+{b}", str((a + b) % 17)


@dataclass(frozen=True)
class Skill:
    name: str
    tag: str
    generate: Callable[[random.Random], tuple[str, str]]
    max_answer_len: int

    def prompt(self, x: str) -> str:
        return f"{self.tag}:{x}="


SKILLS: tuple[Skill, ...] = (
    Skill("copy", "c", _copy, 6),
    Skill("reverse", "r", _reverse, 6),
    Skill("sort", "s", _sort, 6),
    Skill("depth", "d", _depth, 1),
    Skill("max", "x", _max, 1),
    Skill("index", "i", _index, 1),
    Skill("add", "a", _add, 3),
    Skill("modadd", "m", _modadd, 2),
)
SKILL_NAMES = tuple(s.name for s in SKILLS)
SKILL_BY_NAME = {s.name: s for s in SKILLS}
MAX_NEW_TOKENS = max(s.max_answer_len for s in SKILLS) + 1


def encode(text: str) -> list[int]:
    return [STOI[ch] for ch in text]


def decode(ids) -> str:
    return "".join(ITOS[int(i)] for i in ids if int(i) != PAD)


@dataclass(frozen=True)
class Item:
    skill: str
    prompt: str  # the full prompt, ending in "="
    answer: str


@dataclass(frozen=True)
class ItemSets:
    main: dict[str, tuple[Item, ...]]
    twin: dict[str, tuple[Item, ...]]
    held_out: frozenset[str]  # every eval prompt; the training sampler must avoid these


@lru_cache(maxsize=4)
def build_item_sets(n_items: int = N_ITEMS, seed: int = ITEM_SEED) -> ItemSets:
    """Draw the fixed main and twin item sets. Prompts are unique across both sets."""
    main: dict[str, tuple[Item, ...]] = {}
    twin: dict[str, tuple[Item, ...]] = {}
    held: set[str] = set()
    for k, skill in enumerate(SKILLS):
        rng = random.Random(seed * 100 + k)
        items: list[Item] = []
        seen: set[str] = set()
        while len(items) < 2 * n_items:
            x, y = skill.generate(rng)
            p = skill.prompt(x)
            if p in seen:
                continue
            seen.add(p)
            items.append(Item(skill.name, p, y))
        # Interleave so main and twin are exchangeable draws from one stream.
        main[skill.name] = tuple(items[0::2])
        twin[skill.name] = tuple(items[1::2])
        held |= seen
    return ItemSets(main, twin, frozenset(held))


def sample_training_example(
    skill: Skill, rng: random.Random, held_out: frozenset[str]
) -> tuple[str, str]:
    """One fresh (prompt, answer) pair that is not an eval item."""
    while True:
        x, y = skill.generate(rng)
        p = skill.prompt(x)
        if p not in held_out:
            return p, y

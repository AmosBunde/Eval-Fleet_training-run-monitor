"""The statistics, written out with numpy and scipy distribution functions only.

Everything here works on arrays so that a whole run (every checkpoint and eval) is tested
in one call. Each function is unit-tested against hand-computed or textbook values.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import binom, hypergeom

# --- intervals ------------------------------------------------------------------------------


def wilson_interval(k, n, conf: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion ``k/n``."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = ndtri(0.5 + conf / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def bootstrap_ci(values, n_boot: int = 10000, conf: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values`` (resampling whole units)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(lo), float(hi)


# --- unpaired tests -------------------------------------------------------------------------


def two_proportion_z_pvalue(k_ref, n_ref, k_new, n_new) -> np.ndarray:
    """One-sided pooled two-proportion z-test. Small p means the new proportion is lower.

    This treats the two samples as independent. When both checkpoints are scored on the same
    items it ignores the pairing and is usually conservative.
    """
    k_ref, n_ref, k_new, n_new = (np.asarray(a, dtype=float) for a in (k_ref, n_ref, k_new, n_new))
    pooled = (k_ref + k_new) / (n_ref + n_new)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_ref + 1 / n_new))
    diff = k_ref / n_ref - k_new / n_new
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, diff / se, 0.0)
    return ndtr(-z)


def fisher_exact_greater(k_a, n_a, k_b, n_b) -> np.ndarray:
    """One-sided Fisher exact test. Small p means proportion A is higher than proportion B."""
    k_a, n_a, k_b, n_b = (np.asarray(a, dtype=np.int64) for a in (k_a, n_a, k_b, n_b))
    # Given the margins, k_a is hypergeometric: population n_a+n_b, k_a+k_b successes, n_a draws.
    return hypergeom.sf(k_a - 1, n_a + n_b, k_a + k_b, n_a)


def binomial_tail_pvalue(k, n, p0) -> np.ndarray:
    """Exact one-sided binomial test: P(X >= k) for X ~ Binomial(n, p0)."""
    return binom.sf(np.asarray(k) - 1, n, p0)


# --- paired tests ---------------------------------------------------------------------------


def discordant_counts(ref, new) -> tuple[np.ndarray, np.ndarray]:
    """Items that went right->wrong (``lost``) and wrong->right (``gained``), along the last axis."""
    ref = np.asarray(ref, dtype=bool)
    new = np.asarray(new, dtype=bool)
    return (ref & ~new).sum(-1), (~ref & new).sum(-1)


def mcnemar_exact_pvalue(lost, gained) -> np.ndarray:
    """One-sided exact McNemar (sign) test. Small p means more items were lost than gained.

    Under no change, each discordant item is equally likely to be a loss or a gain, so the
    number lost is Binomial(lost + gained, 1/2). Concordant items carry no information.
    """
    lost = np.asarray(lost)
    gained = np.asarray(gained)
    return binom.sf(lost - 1, lost + gained, 0.5)


LAMBDA_GRID = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])


def betting_evalue(lost, gained, n, margin: float = 0.0, lambdas=LAMBDA_GRID) -> np.ndarray:
    """E-value against "the score fell by at most ``margin``", from item-level differences.

    For each item let d = ref - new, in {-1, 0, +1}. The null says E[d] <= margin. For a fixed
    bet size lam in [0, 1/(1+margin)], the product over items of (1 + lam*(d - margin)) is
    non-negative and has expectation at most 1 under the null, so it is an e-value; so is the
    average over a grid of bet sizes fixed in advance. By Markov's inequality,
    P(e >= 1/alpha) <= alpha at any sample size, with no normal approximation.
    """
    lost = np.asarray(lost, dtype=float)[..., None]
    gained = np.asarray(gained, dtype=float)[..., None]
    n = np.asarray(n, dtype=float)[..., None]
    lam = np.asarray(lambdas, dtype=float) / (1 + margin)
    log_e = (
        lost * np.log1p(lam * (1 - margin))
        + gained * np.log1p(-lam * (1 + margin))
        + (n - lost - gained) * np.log1p(-lam * margin)
    )
    peak = log_e.max(-1, keepdims=True)
    return np.exp(peak[..., 0]) * np.exp(log_e - peak).mean(-1)


# --- multiplicity ---------------------------------------------------------------------------


def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    """Per-test level that keeps the chance of any false rejection among ``n_tests`` below alpha."""
    return alpha / max(1, n_tests)


def spending_schedule(alpha: float, n_looks: int, kind: str = "uniform") -> np.ndarray:
    """Split ``alpha`` across ``n_looks`` sequential looks. The parts sum to at most alpha.

    ``uniform`` needs the number of looks in advance. ``series`` uses alpha/(t(t+1)), which
    sums to less than alpha for any number of looks, so the horizon need not be known.
    Either way the union bound gives: P(any false alarm over all looks) <= alpha, whatever
    the dependence between looks.
    """
    t = np.arange(1, n_looks + 1, dtype=float)
    if kind == "uniform":
        return np.full(n_looks, alpha / n_looks)
    if kind == "series":
        return alpha / (t * (t + 1))
    raise ValueError(f"unknown spending schedule {kind!r}")


def benjamini_hochberg(pvalues, q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure. Returns a boolean mask of rejections."""
    p = np.asarray(pvalues, dtype=float).ravel()
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        reject[order[: np.nonzero(passed)[0].max() + 1]] = True
    return reject.reshape(np.shape(pvalues))


# --- eval health ----------------------------------------------------------------------------


def split_half_reliability(scores, n_splits: int = 200, seed: int = 0) -> float:
    """Split-half reliability of an eval across checkpoints, Spearman-Brown corrected.

    ``scores`` is ``[checkpoint, item]``. Items are split at random into two halves; the two
    half-scores are correlated across checkpoints and the correlation is stepped up to full
    length. A value near 1 means the eval ranks checkpoints the same way whichever half of
    its items is used. It is undefined (nan) when a half-score does not vary at all.
    """
    x = np.asarray(scores, dtype=float)
    n = x.shape[1]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a = x[:, perm[: n // 2]].mean(1)
        b = x[:, perm[n // 2 :]].mean(1)
        if a.std() == 0 or b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        # The step-up formula blows up as r nears -1; floor strongly negative splits at -1.
        out.append(2 * r / (1 + r) if r > -1 / 3 else -1.0)
    return float(np.mean(out)) if out else float("nan")


def minimum_detectable_drop(n: int, alpha: float, base_rate: float = 0.9) -> float:
    """Smallest one-step score drop the exact paired test can flag at level ``alpha``.

    Best case for the monitor: no item flips wrong->right, so the test needs ``m`` items
    lost with 0.5**m <= alpha. Returned as a fraction of the ``n`` items, capped by how many
    items were right to begin with.
    """
    m = int(np.ceil(np.log(alpha) / np.log(0.5)))
    return min(m, int(round(base_rate * n))) / n if m <= n else float("nan")

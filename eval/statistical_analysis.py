"""Statistical analysis utilities for RL evaluation."""

import numpy as np
from scipy import stats


def confidence_interval(data: list | np.ndarray, confidence: float = 0.95) -> tuple:
    """Compute confidence interval using t-distribution.

    Returns: (mean, lower, upper, margin_of_error)
    """
    data = np.array(data)
    n = len(data)
    if n < 2:
        mean = float(data[0]) if n == 1 else 0.0
        return mean, mean, mean, 0.0

    mean = np.mean(data)
    se = stats.sem(data)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se

    return float(mean), float(mean - margin), float(mean + margin), float(margin)


def paired_t_test(baseline: list | np.ndarray, rl: list | np.ndarray) -> tuple:
    """Paired t-test between baseline and RL results.

    Returns: (t_statistic, p_value)

    When paired differences have zero variance but a non-zero mean, uses a
    one-sample t-test of the differences against zero — this correctly identifies
    a deterministic, consistent effect (e.g., silent failure eliminated across
    all seeds). Returns (0.0, 1.0) only when both arrays are truly identical.
    """
    baseline = np.array(baseline)
    rl = np.array(rl)
    if len(baseline) != len(rl) or len(baseline) < 2:
        return 0.0, 1.0

    diff = rl - baseline
    if np.std(diff) == 0:
        if np.mean(diff) == 0.0:
            # Truly identical across seeds — no detectable difference
            return 0.0, 1.0
        else:
            # Constant non-zero difference across all seeds: one-sample t-test
            # against zero gives t = -inf / p -> 0 (deterministic effect)
            t_stat, p_value = stats.ttest_1samp(diff, 0)
            return float(t_stat), float(p_value)

    t_stat, p_value = stats.ttest_rel(rl, baseline)
    return float(t_stat), float(p_value)


def cohens_d(baseline: list | np.ndarray, rl: list | np.ndarray) -> float:
    """Compute Cohen's d effect size (paired).

    Returns float('inf') when the difference is perfectly consistent (zero
    variance) but non-zero — this represents an unbounded effect size, not
    zero. Returns 0.0 only when there is genuinely no difference.
    """
    baseline = np.array(baseline)
    rl = np.array(rl)
    diff = rl - baseline
    if np.std(diff) == 0:
        # Constant difference across all seeds: effect is either 0 or infinite
        return float('inf') if np.mean(diff) != 0.0 else 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def format_ci(mean: float, lower: float, upper: float, as_pct: bool = True,
              clamp: bool = True) -> str:
    """Format a confidence interval for display.

    clamp: if True, clamps percentage bounds to [0, 1] to avoid nonsensical
    values like 103.2% when small-sample t-CIs exceed the feasible range.
    """
    if clamp:
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    if as_pct:
        return f"{mean:.1%} [{lower:.1%}, {upper:.1%}]"
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"

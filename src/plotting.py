"""
Utility functions for plotting the frequency of cast vote records (CVRs).

Usage:
    python -m src.plotting           # draws a demo figure using sample data

The main entrypoint is `plot_ballot_frequencies`, which accepts a CVR array
and the number of seats `m`. Ballots that contain only the sentinel value -1
are discarded before tallying and quota calculation.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# --- Controls for compact layout ---
FIGSIZE = (4, 3)  # (width, height) in inches
BAR_WIDTH = 0.62
FONTSIZE = 8
SENTINEL = -1
# -----------------------------------


def _clean_ballot(ballot: np.ndarray, sentinel: int = SENTINEL) -> Tuple[int, ...] | None:
    """Return the ballot without sentinel values or None if it contains no votes."""
    trimmed = ballot[ballot != sentinel]
    if trimmed.size == 0:
        return None
    return tuple(int(x) for x in trimmed.tolist())


def tally_ballots(cvr: Sequence[np.ndarray], sentinel: int = SENTINEL) -> Counter:
    """
    Count identical ballots after removing sentinel values.

    Ballots made entirely of sentinels are excluded from the count.
    """
    counter: Counter = Counter()
    for ballot in cvr:
        cleaned = _clean_ballot(np.asarray(ballot), sentinel=sentinel)
        if cleaned is not None:
            counter[cleaned] += 1
    return counter


def plot_ballot_frequencies(
    cvr: Sequence[np.ndarray],
    m: int,
    *,
    quota: float | None = None,
    highlight_ballot: Iterable[int] | None = None,
    title: str | None = None,
    sentinel: int = SENTINEL,
) -> None:
    """
    Plot a bar chart showing how often each (cleaned) ballot occurs.

    Parameters
    ----------
    cvr:
        Iterable of rank-order ballots encoded as integer numpy arrays.
    m:
        Number of seats. Used to compute the quota q = (# valid ballots)/(m + 1).
    highlight_ballot:
        Optional ballot to highlight, e.g. highlight_ballot=(4,) sets that bar red.
    sentinel:
        Value that indicates an empty ranking slot.
    """
    ballot_counts = tally_ballots(cvr, sentinel=sentinel)
    if not ballot_counts:
        raise ValueError("No valid ballots remain after removing sentinel-only records.")

    valid_ballots_cast = sum(ballot_counts.values())
    if quota is None:
        quota = valid_ballots_cast / (m + 1)

    # Order bars by frequency (descending) then lexicographically for stability.
    ordered = sorted(ballot_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [_format_ballot_label(ballot) for ballot, _ in ordered]
    heights = [count for _, count in ordered]

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    x = list(range(len(labels)))
    bars = ax.bar(x, heights, width=BAR_WIDTH)
    ax.bar_label(bars, fmt="%d", padding=2, fontsize=FONTSIZE)

    if highlight_ballot is not None:
        highlight_label = _format_ballot_label(tuple(highlight_ballot))
        if highlight_label in labels:
            bars[labels.index(highlight_label)].set_facecolor("red")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)

    top = max(max(heights), quota) * 1.15
    ax.set_ylim(0, top)
    ax.margins(x=0.02)

    ax.axhline(quota, linestyle="--", color="tab:red", linewidth=1)
    ax.annotate(
        f"m={m} quota is {quota:.2f}",
        xy=(1.0, quota),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=FONTSIZE,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Ballots", fontsize=FONTSIZE)
    ax.set_ylabel("# of times cast", fontsize=FONTSIZE)
    if title is not None:
        ax.set_title(title, fontsize=FONTSIZE + 2)

    plt.show()


def _format_ballot_label(ballot: Tuple[int, ...]) -> str:
    """Return a string such as '(4, 1, 3)'."""
    return f"({', '.join(map(str, ballot))})"


def _demo_cvr() -> np.ndarray:
    """Return the sample CVR array from the request."""
    ballots = (
        [np.array([1, 2, -1], dtype=np.int8)] * 700
        + [np.array([2, -1, -1], dtype=np.int8)] * 770
        + [np.array([3, -1, -1], dtype=np.int8)] * 880
        + [np.array([4, 1, -1], dtype=np.int8)] * 350
        + [np.array([5, 1, -1], dtype=np.int8)] * 300
        + [np.array([-1, -1, -1], dtype=np.int8)] * 100
    )
    return np.array(ballots)


if __name__ == "__main__":
    demo_cvr = _demo_cvr()
    plot_ballot_frequencies(demo_cvr, m=2, highlight_ballot=(4,))

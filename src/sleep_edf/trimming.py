"""Label-based offline Wake protocols; indices always remain on the original grid."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WakeSelection:
    full: np.ndarray
    trimmed: np.ndarray
    first_sleep: int | None
    last_sleep: int | None
    start_epoch: int | None
    stop_epoch_exclusive: int | None
    no_valid_sleep: bool


def select_wake_protocols(labels, before=60, after=60):
    """None/negative excluded labels never become supervised epochs.

    The inclusion window is closed at the last sleep epoch + after; use an
    exclusive stop of last+after+1. Recording limits cap the context naturally.
    """
    labels = list(labels)
    if before < 0 or after < 0:
        raise ValueError("Wake context cannot be negative")
    valid = np.array([label is not None and label in (0, 1, 2, 3, 4) for label in labels], dtype=bool)
    sleep = np.flatnonzero([label in (1, 2, 3, 4) for label in labels])
    trimmed = np.zeros(len(labels), dtype=bool)
    if not len(sleep):
        return WakeSelection(valid, trimmed, None, None, None, None, True)
    first, last = int(sleep[0]), int(sleep[-1])
    start, stop = max(0, first - before), min(len(labels), last + after + 1)
    trimmed[start:stop] = valid[start:stop]
    return WakeSelection(valid, trimmed, first, last, start, stop, False)

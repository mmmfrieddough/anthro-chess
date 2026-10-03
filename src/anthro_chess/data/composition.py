"""Reweighting how often a training selection draws each rating.

A composition thins: each game is kept with probability proportional to a
weight taken against a smoothed density of its players' ratings, so a rare
rating is drawn at its full rate and a common one at a share of its own. No
example is repeated and none carries a loss weight.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from anthro_chess.data.artifacts import DataLoadingError
from anthro_chess.data.loading import _RANK_SPACE, _rank_key
from anthro_chess.data.prepare import _rating_bucket

#: Bumped when the fit or the thinning changes, so a run resumed under code that
#: would compose differently is refused rather than drawing other games.
RATING_COMPOSITION_VERSION = 2

#: Kernel width, in rating points, of the density the weight is taken against.
#: Wide enough to absorb the spike a source puts at its provisional starting
#: rating, which a narrower kernel turns into a dip in the weight there.
DENSITY_BANDWIDTH = 50

_KERNEL_REACH = 4 * DENSITY_BANDWIDTH
#: Acceptance for a rating past either end of the fit. Its density is taken as
#: nil, so it is kept as often as the heaviest rating.
_OUTSIDE_FIT = 1.0
#: Precision the fitted acceptance is held at, so its digest does not move with
#: the summation order of whichever array library computed it.
_ACCEPTANCE_DIGITS = 9


@dataclass(frozen=True)
class RatingComposition:
    """A fitted composition: the share of decisions kept at each rating.

    ``acceptance`` is indexed by normalized rating.
    """

    balance: float
    acceptance: tuple[float, ...]
    #: What the fit sample says the composition does, for the run record.
    report: dict[str, Any]

    @property
    def sha256(self) -> str:
        """Return what a resumed run has to have fitted to draw the same games."""

        return sha256(
            json.dumps(
                {
                    "version": RATING_COMPOSITION_VERSION,
                    "balance": self.balance,
                    "acceptance": self.acceptance,
                },
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def game_acceptance(self, white: int, black: int, length: int) -> float:
        """Return the probability a game is kept, from its decisions' weights."""

        white_decisions, black_decisions = _decisions_by_side(length)
        kept = white_decisions * self._at(white) + black_decisions * self._at(black)
        return float(kept / max(length, 1))

    def keeps(
        self,
        seed: str,
        game_id: int,
        white: int,
        black: int,
        length: int,
    ) -> bool:
        """Say whether the thinning keeps one game.

        Drawn against a digest of its own, so a game's place in the subsample's
        rank space says nothing about whether it survives this.
        """

        key = _rank_key(f"{seed}\0rating-composition", game_id)
        threshold = self.game_acceptance(white, black, length) * _RANK_SPACE
        return int.from_bytes(key[:8], "big") < threshold

    def _at(self, rating: int) -> float:
        if 0 <= rating < len(self.acceptance):
            return self.acceptance[rating]
        return _OUTSIDE_FIT


def fit_rating_composition(
    white: Sequence[int],
    black: Sequence[int],
    length: Sequence[int],
    speed: Sequence[str],
    balance: float,
    *,
    population_games: int,
) -> RatingComposition:
    """Fit a composition from a sample of games the selection would draw.

    Each game is its two ratings, its speed class, and its decision count, and
    the density is over decisions by the rating of the player making them,
    because that is the rating the model is conditioned on at each one. The
    speed class is only reported on: thinning by rating moves that mix too.
    ``population_games`` is how many games the sample stands for.
    """

    # Deferred so this module still imports on an install carrying no extras.
    import numpy as np

    if not len(length):
        raise DataLoadingError("a rating composition needs games to fit against")
    white_rating = np.asarray(white, dtype=np.int64)
    black_rating = np.asarray(black, dtype=np.int64)
    decisions = np.asarray(length, dtype=np.int64)
    white_decisions, black_decisions = _decisions_by_side(decisions)
    size = int(max(white_rating.max(), black_rating.max())) + 1
    histogram = np.bincount(
        white_rating, weights=white_decisions, minlength=size
    ) + np.bincount(black_rating, weights=black_decisions, minlength=size)

    offsets = np.arange(-_KERNEL_REACH, _KERNEL_REACH + 1)
    kernel = np.exp(-0.5 * (offsets / DENSITY_BANDWIDTH) ** 2)
    density = np.convolve(np.pad(histogram, _KERNEL_REACH), kernel, mode="valid")
    with np.errstate(divide="ignore"):
        ratio = density.max() / density
    weight = np.minimum(balance, ratio)
    # Normalized by the heaviest rating anything was drawn at rather than by
    # the balance, so a balance the data never reaches discards nothing for it.
    heaviest = float(weight[histogram > 0].max())
    # Capped where a rating between the fitted ones has a thinner density than
    # any of them, so no rating is kept more often than the heaviest one.
    acceptance = np.round(np.minimum(1.0, weight / heaviest), _ACCEPTANCE_DIGITS)

    kept = (
        white_decisions * acceptance[white_rating]
        + black_decisions * acceptance[black_rating]
    ) / np.maximum(decisions, 1)
    composed = decisions * kept
    drawn_total = decisions.sum()
    composed_total = composed.sum()
    clipped = weight >= balance
    # The ratings drawn equally often: the run about the peak short of the limit.
    peak = int(density.argmax())
    equalized = np.flatnonzero(~clipped)
    runs = np.split(equalized, np.flatnonzero(np.diff(equalized) != 1) + 1)
    flat = next((run for run in runs if run.size and run[0] <= peak <= run[-1]), None)
    clipped_decisions = (
        white_decisions * kept * clipped[white_rating]
        + black_decisions * kept * clipped[black_rating]
    ).sum()
    retained = float(composed_total / drawn_total)
    ratings, slot = np.unique(
        np.concatenate([white_rating, black_rating]), return_inverse=True
    )
    rating_buckets = np.asarray([_rating_bucket(int(rating)) for rating in ratings])
    report = {
        "density_bandwidth": DENSITY_BANDWIDTH,
        "equalized_ratings": (
            None if flat is None or flat.size < 2 else [int(flat[0]), int(flat[-1])]
        ),
        "fit_games": int(len(decisions)),
        "fit_decisions": int(drawn_total),
        "retained_game_share": float(kept.mean()),
        "retained_decision_share": retained,
        # Kish effective sample size as a share of what was drawn: the size a
        # loss weighted by the same acceptance would have had.
        "effective_sample_share": float(
            composed_total**2 / (drawn_total * (decisions * kept**2).sum())
        ),
        "clipped_decision_share": float(clipped_decisions / composed_total),
        "estimated_retained_decisions": int(
            population_games * decisions.mean() * retained
        ),
        "decision_share_by_rating": _shares_by(
            rating_buckets[slot],
            np.concatenate([white_decisions, black_decisions]),
            np.concatenate([kept, kept]),
        ),
        "decision_share_by_speed": _shares_by(speed, decisions, kept),
    }
    return RatingComposition(
        balance=balance,
        acceptance=tuple(float(value) for value in acceptance),
        report=report,
    )


def _decisions_by_side(length: Any) -> tuple[Any, Any]:
    """Split a game's decisions between the side moving first and the other."""

    return (length + 1) // 2, length // 2


def _shares_by(
    labels: Any,
    decisions: Any,
    kept: Any,
) -> dict[str, dict[str, float]]:
    """Return each label's share of decisions, as drawn and composed."""

    import numpy as np

    names, member = np.unique(np.asarray(labels), return_inverse=True)
    composed = decisions * kept
    drawn_at = np.bincount(member, weights=decisions) / decisions.sum()
    composed_at = np.bincount(member, weights=composed) / composed.sum()
    return {
        str(name): {"as_drawn": float(drawn), "composed": float(share)}
        for name, drawn, share in zip(names, drawn_at, composed_at, strict=True)
    }

"""Reweighting how often a training selection draws each rating.

The rating axis arrives with the population's shape, roughly normal about the
corpus median, so the ratings the dial's endpoints ask for are the ones training
sees least. A composition evens that out by thinning: each game is kept with
probability proportional to a weight over its players' ratings, so a rare rating
is drawn at its full rate and a common one at a share of its own. No example is
repeated and none carries a loss weight. Discarding is what that costs, and a
corpus many times the horizon can afford it.

The weight is taken pointwise against a smoothed density rather than per band,
so it has no edges the data does not have.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from anthro_chess.data.artifacts import DataLoadingError
from anthro_chess.data.config import RatingCompositionConfig
from anthro_chess.data.prepare import _rating_bucket

#: Bumped when the fit or the thinning changes, so a run resumed under code that
#: would compose differently is refused rather than drawing other games.
RATING_COMPOSITION_VERSION = 1

#: Kernel width, in rating points, of the density the weight is taken against.
#: Wide enough to absorb the spike a source puts at its provisional starting
#: rating, which a narrower kernel turns into a dip in the weight there.
DENSITY_BANDWIDTH = 50

_KERNEL_REACH = 4 * DENSITY_BANDWIDTH
#: What a rating the fit never saw is kept at. Its density is taken as nil, so
#: its weight is the heaviest any rating can have.
_OUTSIDE_FIT = 1.0
_RANK_SPACE = 1 << 64
#: Precision the fitted acceptance is held at, so its digest does not move with
#: the summation order of whichever array library computed it.
_ACCEPTANCE_DIGITS = 9


@dataclass(frozen=True)
class RatingComposition:
    """A fitted composition: the share of decisions kept at each rating.

    ``acceptance`` is indexed by normalized rating.
    """

    config: RatingCompositionConfig
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
                    "config": self.config.model_dump(mode="json"),
                    "acceptance": self.acceptance,
                    "beyond": _OUTSIDE_FIT,
                },
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def game_acceptance(self, white: int, black: int, length: int) -> float:
        """Return the probability a game is kept, from its decisions' weights."""

        white_decisions, black_decisions = _decisions_by_side(length)
        kept = white_decisions * self._at(white) + black_decisions * self._at(black)
        return min(1.0, kept / max(length, 1))

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

        key = sha256(f"{seed}\0rating-composition\0{game_id}".encode()).digest()
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
    config: RatingCompositionConfig,
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

    # Deferred for the reason `anthro_chess.data.loading` defers it.
    import numpy as np

    if not len(length):
        raise DataLoadingError("a rating composition needs games to fit against")
    white_rating = np.asarray(white, dtype=np.int64)
    black_rating = np.asarray(black, dtype=np.int64)
    decisions = np.asarray(length, dtype=np.int64)
    white_decisions = (decisions + 1) // 2
    black_decisions = decisions // 2
    size = int(max(white_rating.max(), black_rating.max())) + 1
    histogram = np.bincount(
        white_rating, weights=white_decisions, minlength=size
    ) + np.bincount(black_rating, weights=black_decisions, minlength=size)

    offsets = np.arange(-_KERNEL_REACH, _KERNEL_REACH + 1)
    kernel = np.exp(-0.5 * (offsets / DENSITY_BANDWIDTH) ** 2)
    density = np.convolve(np.pad(histogram, _KERNEL_REACH), kernel, mode="valid")
    with np.errstate(divide="ignore"):
        ratio = density.max() / density
    clip = config.maximum_weight
    weight = np.minimum(clip, ratio**config.strength)
    # Normalized by the heaviest rating anything was drawn at rather than by
    # the clip, so a strength of zero keeps every game instead of thinning all
    # of them alike.
    heaviest = float(weight[histogram > 0].max())
    # Capped where a rating between the fitted ones has a thinner density than
    # any of them, so no rating is kept more often than the heaviest one.
    acceptance = np.round(np.minimum(1.0, weight / heaviest), _ACCEPTANCE_DIGITS)

    kept = np.minimum(
        1.0,
        (
            white_decisions * acceptance[white_rating]
            + black_decisions * acceptance[black_rating]
        )
        / np.maximum(decisions, 1),
    )
    composed = decisions * kept
    clipped = weight >= clip
    clipped_decisions = (
        white_decisions * kept * clipped[white_rating]
        + black_decisions * kept * clipped[black_rating]
    ).sum()
    retained = float(composed.sum() / decisions.sum())
    report = {
        "density_bandwidth": DENSITY_BANDWIDTH,
        "fit_games": int(len(decisions)),
        "fit_decisions": int(decisions.sum()),
        "retained_game_share": float(kept.mean()),
        "retained_decision_share": retained,
        # The Kish size, as a share of what was drawn from. Thinning gives every
        # kept decision unit weight, so this is the effective size a loss
        # weighted the same way would have had, and how far the draw sits from
        # the population.
        "effective_sample_share": float(
            composed.sum() ** 2 / (decisions.sum() * (decisions * kept**2).sum())
        ),
        "clipped_decision_share": float(clipped_decisions / composed.sum()),
        "estimated_retained_decisions": int(
            population_games * decisions.mean() * retained
        ),
        "decision_share_by_rating": _shares_by_bucket(
            white_rating,
            black_rating,
            white_decisions,
            black_decisions,
            kept,
        ),
        "decision_share_by_speed": _shares_by_speed(speed, decisions, kept),
    }
    return RatingComposition(
        config=config,
        acceptance=tuple(float(value) for value in acceptance),
        report=report,
    )


def _decisions_by_side(length: int) -> tuple[int, int]:
    """Split a game's decisions between the side moving first and the other."""

    return (length + 1) // 2, length // 2


def _shares_by_bucket(
    white: Any,
    black: Any,
    white_decisions: Any,
    black_decisions: Any,
    kept: Any,
) -> dict[str, dict[str, float]]:
    """Return each rating bucket's share of decisions, as drawn and composed."""

    import numpy as np

    decisions = np.concatenate([white_decisions, black_decisions])
    composed = decisions * np.concatenate([kept, kept])
    ratings, slot = np.unique(np.concatenate([white, black]), return_inverse=True)
    drawn_at = np.bincount(slot, weights=decisions) / decisions.sum()
    composed_at = np.bincount(slot, weights=composed) / composed.sum()
    shares: dict[str, dict[str, float]] = {}
    for rating, drawn_share, composed_share in zip(
        ratings, drawn_at, composed_at, strict=True
    ):
        bucket = shares.setdefault(
            _rating_bucket(int(rating)), {"as_drawn": 0.0, "composed": 0.0}
        )
        bucket["as_drawn"] += float(drawn_share)
        bucket["composed"] += float(composed_share)
    return dict(sorted(shares.items()))


def _shares_by_speed(
    speed: Sequence[str],
    decisions: Any,
    kept: Any,
) -> dict[str, dict[str, float]]:
    """Return each speed class's share of decisions, as drawn and composed."""

    import numpy as np

    classes, member = np.unique(np.asarray(speed), return_inverse=True)
    drawn_at = np.bincount(member, weights=decisions) / decisions.sum()
    composed = decisions * kept
    composed_at = np.bincount(member, weights=composed) / composed.sum()
    return {
        str(name): {"as_drawn": float(drawn), "composed": float(share)}
        for name, drawn, share in zip(classes, drawn_at, composed_at, strict=True)
    }

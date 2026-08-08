"""How often each opening family appears in a training selection.

This is the axis the per-family held-out reading is plotted against. On its own
a per-family loss table says almost nothing: rare openings are genuinely harder
to predict, so "the rare families have higher loss" is the expected result
whether or not they are undertrained. `docs/evaluation.md` owns why that makes
the shape of loss against training frequency the reading, and
`docs/decisions/0016-sampling-axes-versus-measured-distributions.md` owns the
decision it exists to falsify.

Counting is derived here rather than stored beside the corpus, for the reason
`docs/decisions/0015-owned-opening-book.md` gives for classification generally:
a book or granularity change would otherwise mean regenerating data. The cost is
a replay of every training game's opening, which is why the reading that needs
it asks for it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import log10
from pathlib import Path

from anthro_chess.data.artifacts import DataLoadingError, read_normalized_rows
from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.evaluation.aggregation import (
    OPENING_FAMILY_DIMENSION,
    SliceTable,
)
from anthro_chess.evaluation.openings import (
    UNCLASSIFIED,
    OpeningBook,
    OpeningClassificationError,
    OpeningLabel,
    OpeningLevel,
    classify_action_ids,
    load_book,
    opening_distribution,
)

OPENING_FREQUENCY_VERSION = 1

#: Tier a family falls in, by its share of the training selection, from the most
#: common down. The boundaries are code-owned rather than configurable: they are
#: part of what the committed per-tier series mean, and a configured boundary
#: would move families between series without changing a metric identity. The
#: last entry's floor of zero is what makes the table total.
OPENING_FREQUENCY_TIERS: tuple[tuple[str, float], ...] = (
    ("common_opening", 0.05),
    ("uncommon_opening", 0.01),
    ("rare_opening", 0.001),
    ("very_rare_opening", 0.0),
)

#: A family the scored games realized and the training selection never did. Its
#: share is zero rather than small, which is a different statement.
UNSEEN_TIER = "unseen_opening"

#: Games the book named nothing in. Not a family, so it is reported on its own
#: rather than tiered by the share of the corpus that is unnamed, which would
#: file "no opening identified" beside the most popular defense there is.
UNCLASSIFIED_TIER = "unclassified_opening"

OPENING_TIER_NAMES: tuple[str, ...] = (
    *(name for name, _ in OPENING_FREQUENCY_TIERS),
    UNSEEN_TIER,
    UNCLASSIFIED_TIER,
)

#: Families below this share of training games form the tail the slope is fitted
#: over. Read off the tier table rather than restated, so the fit and the tiers
#: cut the distribution in the same place by construction.
TAIL_SHARE_CEILING = dict(OPENING_FREQUENCY_TIERS)["uncommon_opening"]


class OpeningFrequencyError(ValueError):
    """Raised when a training selection cannot be classified by opening."""


@dataclass(frozen=True)
class OpeningFrequency:
    """Opening-family counts over one split of one normalized corpus."""

    split: str
    games: int
    family_games: Mapping[str, int]
    paths: tuple[str, ...]

    @property
    def classified_games(self) -> int:
        """Return how many games the book named an opening in."""

        return sum(self.family_games.values())

    def share(self, family: str) -> float:
        """Return one family's share of the training selection.

        The denominator is every training game rather than every classified one,
        so a share is comparable across corpora whose unnamed fraction differs.
        """

        return self.family_games.get(family, 0) / self.games

    def tier(self, family: str) -> str:
        """Return the committed frequency tier one family falls in."""

        if family == UNCLASSIFIED:
            return UNCLASSIFIED_TIER
        share = self.share(family)
        if share <= 0.0:
            return UNSEEN_TIER
        return next(
            name for name, minimum in OPENING_FREQUENCY_TIERS if share >= minimum
        )

    def as_record(self) -> dict[str, object]:
        """Return the stable provenance record stored with a reading."""

        return {
            "version": OPENING_FREQUENCY_VERSION,
            "split": self.split,
            "games": self.games,
            "classified_games": self.classified_games,
            "families": len(self.family_games),
            "normalized_paths": list(self.paths),
        }


def count_opening_families(
    paths: Sequence[Path],
    split: str,
    *,
    book: OpeningBook | None = None,
) -> OpeningFrequency:
    """Count each opening family's games in one split of a normalized corpus.

    Unnamed games are counted in the total and then dropped from the families,
    which is the one place this distribution differs from the generated-play
    side's: a share is of every training game, so what the book could not name
    belongs in the denominator but is not a family.
    """

    resolved = load_book() if book is None else book
    counts = dict(
        opening_distribution(
            _training_labels(paths, split, resolved),
            OpeningLevel.FAMILY,
        )
    )
    games = sum(counts.values())
    if not games:
        raise OpeningFrequencyError(
            f"the training corpus holds no {split} split games to classify"
        )
    counts.pop(UNCLASSIFIED, None)
    return OpeningFrequency(
        split=split,
        games=games,
        family_games=counts,
        paths=tuple(str(path) for path in paths),
    )


def _training_labels(
    paths: Sequence[Path],
    split: str,
    book: OpeningBook,
) -> Iterator[OpeningLabel]:
    """Classify one split of a normalized corpus, a shard at a time."""

    for path in paths:
        try:
            rows = read_normalized_rows(path, _COUNTED_COLUMNS)
        except DataLoadingError as error:
            raise OpeningFrequencyError(
                f"cannot read training corpus {path}: {error}"
            ) from error
        for row in rows:
            if row[NormalizedColumn.SPLIT.value] != split:
                continue
            try:
                yield classify_action_ids(
                    row[NormalizedColumn.ACTION_IDS.value],
                    initial_position=row[NormalizedColumn.INITIAL_POSITION.value],
                    book=book,
                )
            except OpeningClassificationError as error:
                raise OpeningFrequencyError(str(error)) from error


@dataclass(frozen=True)
class FamilyTailRow:
    """Where one scored family sits on the training-frequency axis.

    Only the frequency side. What the checkpoint did on the family is the
    ``opening_family`` slice the same artifact already carries, and a second
    copy of it here would be two definitions of one number.
    """

    family: str
    tier: str
    training_games: int
    training_share: float

    def as_record(self) -> dict[str, object]:
        """Return the stable detail-tier record for one family."""

        return {
            "family": self.family,
            "tier": self.tier,
            "training_games": self.training_games,
            "training_share": self.training_share,
        }


@dataclass(frozen=True)
class OpeningTailReading:
    """Where the scored families sit, and the one number the decision turns on.

    A negative slope says held-out loss is still falling as training frequency
    rises among the rare families, which is the shape that says more data on
    them would help. A slope at or above zero says the tail's loss is explained
    by those openings being harder rather than by their being scarce.

    Both stay in the detail tier. What reaches the committed store is the
    per-tier slice series, which carry a sampling floor because they are
    ordinary slices of the same scoring pass; a slope fitted over families has
    no such floor and would be quoted as though it did.
    """

    families: tuple[FamilyTailRow, ...]
    tail_families: int
    tail_move_loss_slope: float | None

    def as_record(self) -> dict[str, object]:
        """Return the detail-tier record for the whole reading."""

        return {
            "version": OPENING_FREQUENCY_VERSION,
            "tail_share_ceiling": TAIL_SHARE_CEILING,
            "tail_families": self.tail_families,
            "tail_move_loss_slope": self.tail_move_loss_slope,
            "families": [row.as_record() for row in self.families],
        }


def read_opening_tail(
    slices: SliceTable,
    frequency: OpeningFrequency,
) -> OpeningTailReading:
    """Place every scored family on the training-frequency axis, and fit its tail.

    Unnamed games carry no frequency, so they are left out entirely rather than
    entered at a share of zero: they would sit at the far end of the axis the
    slope is fitted over while saying nothing about how often any opening was
    trained on. A family the training selection never held keeps its row and its
    own tier, but a share of zero has no place on a log axis either, so the fit
    skips it too.
    """

    rows: list[FamilyTailRow] = []
    tail: list[tuple[float, float, int]] = []  # log share, move loss, positions
    for family, summary in sorted(slices.dimensions[OPENING_FAMILY_DIMENSION].items()):
        if family == UNCLASSIFIED:
            continue
        share = frequency.share(family)
        rows.append(
            FamilyTailRow(
                family=family,
                tier=frequency.tier(family),
                training_games=frequency.family_games.get(family, 0),
                training_share=share,
            )
        )
        if 0.0 < share < TAIL_SHARE_CEILING:
            tail.append((log10(share), summary.move_loss, summary.position_count))
    return OpeningTailReading(
        families=tuple(rows),
        tail_families=len(tail),
        tail_move_loss_slope=_weighted_slope(tail),
    )


def _weighted_slope(points: Sequence[tuple[float, float, int]]) -> float | None:
    """Return the position-weighted slope of loss on log training share.

    Weighting by scored positions makes the fit read the same population the
    committed tier series do, rather than giving a family measured on twelve
    positions the same say as one measured on twelve thousand. A pool holding
    one tail family, or several at one share, supplies no slope at all, which
    the zero variance reports.
    """

    weight = float(sum(positions for _, _, positions in points))
    if weight <= 0.0:
        return None
    mean_log_share = (
        sum(log_share * positions for log_share, _, positions in points) / weight
    )
    mean_loss = sum(loss * positions for _, loss, positions in points) / weight
    variance = sum(
        positions * (log_share - mean_log_share) ** 2
        for log_share, _, positions in points
    )
    if variance <= 0.0:
        return None
    covariance = sum(
        positions * (log_share - mean_log_share) * (loss - mean_loss)
        for log_share, loss, positions in points
    )
    return covariance / variance


#: The columns a family count reads. Reading the schema's remaining columns
#: only to discard them is most of what a pass over the training corpus costs.
_COUNTED_COLUMNS = (
    NormalizedColumn.SPLIT.value,
    NormalizedColumn.INITIAL_POSITION.value,
    NormalizedColumn.ACTION_IDS.value,
)


__all__ = [
    "OPENING_FREQUENCY_TIERS",
    "OPENING_FREQUENCY_VERSION",
    "OPENING_TIER_NAMES",
    "TAIL_SHARE_CEILING",
    "UNCLASSIFIED_TIER",
    "UNSEEN_TIER",
    "FamilyTailRow",
    "OpeningFrequency",
    "OpeningFrequencyError",
    "OpeningTailReading",
    "count_opening_families",
    "read_opening_tail",
]

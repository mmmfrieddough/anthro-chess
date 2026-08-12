"""Deterministic per-benchmark selection over a frozen evaluation pool.

A view is a derivation, never new stored data. Each benchmark records the
resolved view spec in its own artifact so a later run can reproduce exactly
which games it measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from pydantic import Field, StrictBool

from anthro_chess.config import ConfigModel
from anthro_chess.evaluation.pool import PoolGame, game_ids_sha256, rank_key

VIEW_SPEC_VERSION = 1


class ViewConfig(ConfigModel):
    """Code-owned schema for one benchmark's selection over a pool."""

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    maximum_games: int | None = Field(default=None, ge=1)
    seed: str = Field(default="anthro-evaluation-view-v1", min_length=1)
    minimum_plies: int | None = Field(default=None, ge=1)
    maximum_plies: int | None = Field(default=None, ge=1)
    require_ratings: StrictBool = False
    prefix_plies: int | None = Field(default=None, ge=1)
    #: Inclusive bounds on the day the source dated a game, for a reading taken
    #: over one era of a corpus that spans several. A game the corpus records no
    #: date for is excluded by either bound rather than assumed to be in range.
    minimum_date: date | None = None
    maximum_date: date | None = None


@dataclass(frozen=True)
class ViewSelection:
    """Selected game ids plus the spec needed to reproduce the selection."""

    name: str
    game_ids: tuple[int, ...]
    prefix_plies: int | None
    eligible_games: int
    excluded_games: dict[str, int]

    @property
    def selected_games(self) -> int:
        """Return how many games survived filtering and subsampling."""

        return len(self.game_ids)

    def as_record(self) -> dict[str, object]:
        """Return the stable spec record stored in benchmark artifacts."""

        return {
            "version": VIEW_SPEC_VERSION,
            "name": self.name,
            "selected_games": self.selected_games,
            "eligible_games": self.eligible_games,
            "excluded_games": dict(sorted(self.excluded_games.items())),
            "prefix_plies": self.prefix_plies,
            "game_ids_sha256": game_ids_sha256(self.game_ids),
        }


def apply_view(games: Sequence[PoolGame], config: ViewConfig) -> ViewSelection:
    """Filter and deterministically subsample pool games for one benchmark."""

    if config.maximum_plies is not None and config.minimum_plies is not None:
        if config.maximum_plies < config.minimum_plies:
            raise ValueError("view maximum_plies must not be below minimum_plies")
    if config.maximum_date is not None and config.minimum_date is not None:
        if config.maximum_date < config.minimum_date:
            raise ValueError("view maximum_date must not be before minimum_date")

    excluded: dict[str, int] = {}
    eligible: list[PoolGame] = []
    for game in games:
        reason = _exclusion_reason(game, config)
        if reason is None:
            eligible.append(game)
        else:
            excluded[reason] = excluded.get(reason, 0) + 1

    ordered = sorted(eligible, key=lambda game: rank_key(config.seed, game.game_id))
    if config.maximum_games is not None:
        ordered = ordered[: config.maximum_games]

    # A declared name describes the selection its own config asks for, and a
    # sweep override that caps the games leaves it describing something larger
    # than what ran. Naming the cap where it bit is what keeps a stored name
    # from outliving the reading it labels.
    truncated = len(ordered) < len(eligible)
    return ViewSelection(
        name=f"{config.name}-{len(ordered)}" if truncated else config.name,
        game_ids=tuple(sorted(game.game_id for game in ordered)),
        prefix_plies=config.prefix_plies,
        eligible_games=len(eligible),
        excluded_games=excluded,
    )


def _exclusion_reason(game: PoolGame, config: ViewConfig) -> str | None:
    if config.minimum_plies is not None and game.ply_count < config.minimum_plies:
        return "below_minimum_plies"
    if config.maximum_plies is not None and game.ply_count > config.maximum_plies:
        return "above_maximum_plies"
    if config.prefix_plies is not None and game.ply_count < config.prefix_plies:
        return "shorter_than_prefix"
    if config.require_ratings and not game.has_ratings:
        return "missing_ratings"
    if config.minimum_date is not None or config.maximum_date is not None:
        if game.source_date is None:
            return "missing_date"
        if config.minimum_date is not None and game.source_date < config.minimum_date:
            return "before_minimum_date"
        if config.maximum_date is not None and game.source_date > config.maximum_date:
            return "after_maximum_date"
    return None

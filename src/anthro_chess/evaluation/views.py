"""Deterministic per-benchmark selection over a frozen evaluation pool.

A view is a derivation, never new stored data. Each benchmark records the
resolved view spec in its own artifact so a later run can reproduce exactly
which games it measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import Field, StrictBool

from anthro_chess.config import ConfigModel
from anthro_chess.data import Speed
from anthro_chess.evaluation.pool import PoolGame, game_ids_sha256, rank_key

VIEW_SPEC_VERSION = 2


class ViewConfig(ConfigModel):
    """Code-owned schema for one benchmark's selection over a pool."""

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    maximum_games: int | None = Field(default=None, ge=1)
    seed: str = Field(default="anthro-evaluation-view-v1", min_length=1)
    minimum_plies: int | None = Field(default=None, ge=1)
    maximum_plies: int | None = Field(default=None, ge=1)
    require_ratings: StrictBool = False
    prefix_plies: int | None = Field(default=None, ge=1)
    #: One speed class, the same one the slice tables report, absent to take
    #: every class. A filter here rather than downstream because it decides
    #: which games a cap then takes: filtering a capped selection afterwards
    #: would shrink it below the size it was declared at.
    speed: Speed | None = None


@dataclass(frozen=True)
class ViewSelection:
    """Selected game ids plus the spec needed to reproduce the selection."""

    name: str
    game_ids: tuple[int, ...]
    prefix_plies: int | None
    speed: Speed | None
    eligible_games: int
    excluded_games: dict[str, int]

    @property
    def selected_games(self) -> int:
        """Return how many games survived filtering and subsampling."""

        return len(self.game_ids)

    @property
    def excluded_summary(self) -> str:
        """Return why the view kept what it kept, for a caller that has to say.

        A view that selected nothing looks the same however it got there, and a
        speed class and a ply bound are not the same problem to fix.
        """

        return (
            ", ".join(
                f"{count} {reason}"
                for reason, count in sorted(self.excluded_games.items())
            )
            or "nothing excluded"
        )

    def as_record(self) -> dict[str, object]:
        """Return the stable spec record stored in benchmark artifacts."""

        return {
            "version": VIEW_SPEC_VERSION,
            "name": self.name,
            "selected_games": self.selected_games,
            "eligible_games": self.eligible_games,
            "excluded_games": dict(sorted(self.excluded_games.items())),
            "prefix_plies": self.prefix_plies,
            "speed": None if self.speed is None else str(self.speed),
            "game_ids_sha256": game_ids_sha256(self.game_ids),
        }


def apply_view(games: Sequence[PoolGame], config: ViewConfig) -> ViewSelection:
    """Filter and deterministically subsample pool games for one benchmark."""

    if config.maximum_plies is not None and config.minimum_plies is not None:
        if config.maximum_plies < config.minimum_plies:
            raise ValueError("view maximum_plies must not be below minimum_plies")

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
        speed=config.speed,
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
    if config.speed is not None and game.speed != config.speed:
        return "missing_time_control" if game.speed is None else "speed_mismatch"
    return None

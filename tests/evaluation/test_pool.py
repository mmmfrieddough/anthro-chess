"""Tests for freezing and loading the evaluation pool."""

import json
import tomllib
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import Speed
from anthro_chess.data.accounts import MarkedAccounts, account_digest
from anthro_chess.data.artifacts import file_sha256
from anthro_chess.data.schema import SCHEMA_VERSION
from anthro_chess.evaluation import (
    BENCHMARK_VERSION,
    EvaluationPoolError,
    PoolConfig,
    ViewConfig,
    apply_view,
    freeze_pool,
    load_pool,
)
from anthro_chess.evaluation import pool as pool_module


def _resolved(
    normalized: Path,
    manifest: Path,
    **overrides: object,
) -> ResolvedConfig[PoolConfig]:
    return ResolvedConfig(
        value=PoolConfig.model_validate(
            {
                "pool_id": "fixture-test",
                "normalized": str(normalized),
                "manifest": str(manifest),
                **overrides,
            }
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


@pytest.fixture
def corpus(
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> Callable[[Path], tuple[Path, Path]]:
    """Return a factory writing a mixed-split corpus beneath a directory."""

    def build(tmp_path: Path) -> tuple[Path, Path]:
        return write_corpus(
            tmp_path / "corpus",
            [
                normalized_row(1, split="train"),
                normalized_row(2, split="train"),
                normalized_row(3, split="validation"),
                normalized_row(4, split="test", plies=4, result="0-1"),
                normalized_row(5, split="test", plies=8, rating=900, clocks=False),
            ],
        )

    return build


def test_freeze_selects_only_the_test_split_and_records_provenance(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    record = json.loads(result.manifest_path.read_text())
    assert result.games == 2
    assert result.plies == 12
    assert record["benchmark_version"] == BENCHMARK_VERSION
    assert record["pool"] == {"id": "fixture-test", "version": 1, "split": "test"}
    assert record["source"]["manifest_sha256"]
    assert record["output"]["sha256"]
    assert set(record) >= {
        "action_vocabulary",
        "coverage",
        "encoding",
        "identity",
        "leakage",
        "marked_accounts",
        "preprocessing_version",
        "resolved_config",
        "sampling",
        "schema_version",
    }
    # Absent rather than zero, so a generation nothing filtered cannot read as
    # one a snapshot found nothing in.
    assert record["marked_accounts"] is None


def test_freeze_takes_the_admitted_rows_of_each_row_group(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    """Admission is decided from a projection and the rows are taken by position.

    A position is only meaningful against the row group it was found in, so a
    take resolved against the wrong one writes a real game that is a different
    game. The corpus is laid out so that the admitted rows sit at differing
    positions, one row group admits nothing, and each row is a different length.
    """

    admitted = (1, 2, 4, 9)
    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [
            normalized_row(
                index,
                split="test" if index in admitted else "train",
                plies=index + 1,
            )
            for index in range(10)
        ],
        games_per_shard=4,
        row_group_size=2,
    )

    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    pool = load_pool(tmp_path / "pool")
    assert {game.game_id: game.ply_count for game in pool.games} == {
        fixture_game_id(index): index + 1 for index in admitted
    }


def test_manifest_records_ids_and_content_hashes_for_later_leakage_checks(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    """#29 compares these against an evaluated checkpoint's training identity."""

    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    record = json.loads(result.manifest_path.read_text())
    games = record["identity"]["games"]
    assert [entry["game_id"] for entry in games] == sorted(
        (fixture_game_id(4), fixture_game_id(5))
    )
    assert all(len(entry["content_sha256"]) == 64 for entry in games)


def test_build_time_overlap_check_compares_against_the_train_split(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    leakage = json.loads(result.manifest_path.read_text())["leakage"]
    assert leakage["compared_split"] == "train"
    assert leakage["compared_games"] == 2
    assert leakage["overlapping_games"] == 0


def test_a_game_in_both_train_and_test_fails_the_build(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [normalized_row(7, split="train"), normalized_row(7, split="test")],
    )

    with pytest.raises(EvaluationPoolError, match="also appear in the train split"):
        freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")


def test_a_sample_fraction_admits_that_share_of_the_split(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The realized share is what the pool is sized by, so it is what is pinned.

    The bounds are three standard deviations either side of Binomial(400, 0.25),
    wide enough that this fixture's draw sits comfortably inside and narrow
    enough that a threshold off by a factor of two falls outside.
    """

    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [normalized_row(index, split="test") for index in range(400)],
    )

    result = freeze_pool(
        _resolved(normalized, manifest, sample_fraction=0.25),
        tmp_path / "pool",
    )

    assert 74 <= result.games <= 126
    sampling = json.loads(result.manifest_path.read_text())["sampling"]
    assert sampling["fraction"] == 0.25
    assert sampling["split_games"] == 400


def test_the_admission_seed_is_frozen(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """A changed seed redraws membership and drops games earlier pools held.

    Nothing else here can see that. The check that would is the containment
    verification a generation cut owes, which is still owed and has no earlier
    sampled pool to compare against anyway, so this is what stands between an
    edited seed and a break nobody notices. A failure here is repaired by
    restoring the seed rather than by accepting the new digest.
    """

    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [normalized_row(index, split="test") for index in range(40)],
    )

    result = freeze_pool(
        _resolved(normalized, manifest, sample_fraction=0.25),
        tmp_path / "pool",
    )

    assert pool_module.POOL_SAMPLE_SEED == "anthro-evaluation-pool-v1"
    assert (
        result.game_ids_sha256
        == "7a1604e77f9cb1a35b70ed8b2b8bb270ecd3e7d99c48b89b14d97c55bf9fe013"
    )


def test_a_sampled_pool_still_contains_the_one_a_smaller_corpus_produced(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """Containment is what a bounded pool has to keep, and why it is a fraction.

    Every generation must hold everything the last one did. Keeping the lowest
    ranked N of a larger split would not: the newly available games take ranks
    among the old ones and push some of them past N.
    """

    rows = [normalized_row(index, split="test") for index in range(240)]
    smaller, smaller_manifest = write_corpus(tmp_path / "smaller", rows[:120])
    grown, grown_manifest = write_corpus(tmp_path / "grown", rows)

    freeze_pool(
        _resolved(smaller, smaller_manifest, sample_fraction=0.25),
        tmp_path / "first",
    )
    freeze_pool(
        _resolved(grown, grown_manifest, sample_fraction=0.25),
        tmp_path / "second",
    )

    first = set(load_pool(tmp_path / "first").game_ids)
    second = set(load_pool(tmp_path / "second").game_ids)
    assert first <= second
    assert len(second) > len(first)


def test_a_generation_records_what_it_was_verified_to_contain(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    rows = [normalized_row(index, split="test") for index in range(240)]
    smaller, smaller_manifest = write_corpus(tmp_path / "smaller", rows[:120])
    grown, grown_manifest = write_corpus(tmp_path / "grown", rows)

    first = freeze_pool(
        _resolved(smaller, smaller_manifest, sample_fraction=0.25),
        tmp_path / "first",
    )
    second = freeze_pool(
        _resolved(
            grown,
            grown_manifest,
            sample_fraction=0.25,
            predecessor=str(tmp_path / "first"),
            predecessor_game_ids_sha256=first.game_ids_sha256,
        ),
        tmp_path / "second",
    )

    record = json.loads(second.manifest_path.read_text())["containment"]
    assert record["predecessor_game_ids_sha256"] == first.game_ids_sha256
    assert record["predecessor_games"] == first.games
    assert record["added_games"] == second.games - first.games


def test_the_first_generation_records_no_predecessor(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """Being first is a fact about the chain, not the absence of a check."""

    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    assert json.loads(result.manifest_path.read_text())["containment"] is None


def test_a_lowered_fraction_is_caught_as_a_missing_game(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The one direction the fraction may never move, seen where it does damage.

    Nothing compares the configured fractions. A generation that admits less
    fails because a game is gone, which is also what a moved sample seed and a
    newly rejecting filter look like from here.
    """

    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [normalized_row(index, split="test") for index in range(240)],
    )
    first = freeze_pool(
        _resolved(normalized, manifest, sample_fraction=0.5),
        tmp_path / "first",
    )

    with pytest.raises(EvaluationPoolError, match="drops .* game"):
        freeze_pool(
            _resolved(
                normalized,
                manifest,
                sample_fraction=0.25,
                predecessor=str(tmp_path / "first"),
                predecessor_game_ids_sha256=first.game_ids_sha256,
            ),
            tmp_path / "second",
        )


def test_containment_is_checked_against_the_named_generation_only(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """A check against the wrong predecessor passes while proving nothing."""

    rows = [normalized_row(index, split="test") for index in range(240)]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    freeze_pool(
        _resolved(normalized, manifest, sample_fraction=0.25),
        tmp_path / "first",
    )

    with pytest.raises(EvaluationPoolError, match="not the one this configuration"):
        freeze_pool(
            _resolved(
                normalized,
                manifest,
                sample_fraction=0.5,
                predecessor=str(tmp_path / "first"),
                predecessor_game_ids_sha256="0" * 64,
            ),
            tmp_path / "second",
        )


def test_a_pinned_predecessor_with_no_pool_to_read_is_refused(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    with pytest.raises(EvaluationPoolError, match="no predecessor pool is configured"):
        freeze_pool(
            _resolved(
                normalized,
                manifest,
                predecessor_game_ids_sha256="0" * 64,
            ),
            tmp_path / "pool",
        )


def test_an_admitted_game_in_both_splits_still_fails_the_build(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """Sampling narrows the overlap check to the games that can reach the pool.

    A duplicate the fraction excludes is no longer caught, which is the price
    of not holding the train split in memory to write a bounded pool from it.
    """

    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [normalized_row(index, split="train") for index in range(40)]
        + [normalized_row(index, split="test") for index in range(40)],
    )

    with pytest.raises(EvaluationPoolError, match="also appear in the train split"):
        freeze_pool(
            _resolved(normalized, manifest, sample_fraction=0.5),
            tmp_path / "pool",
        )


def test_an_empty_pool_blames_the_fraction_or_the_split_but_not_both(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """Either message sent to the other case costs a reader the wrong search."""

    populated, populated_manifest = corpus(tmp_path)
    with pytest.raises(EvaluationPoolError, match="sample fraction"):
        freeze_pool(
            _resolved(populated, populated_manifest, sample_fraction=1e-9),
            tmp_path / "sampled-out",
        )

    empty, empty_manifest = write_corpus(
        tmp_path / "no-test-games", [normalized_row(1, split="train")]
    )
    with pytest.raises(EvaluationPoolError, match="no normalized games"):
        freeze_pool(
            _resolved(empty, empty_manifest, sample_fraction=0.5),
            tmp_path / "empty-split",
        )


def _snapshot(
    path: Path,
    *usernames: str,
    covers: tuple[str, ...] = ("0" * 64,),
) -> Path:
    """Write a snapshot marking those accounts across the fixture's archive."""

    return MarkedAccounts(
        covers_archives=covers,
        queried_at="2026-08-13",
        accounts_total=40,
        accounts_queried=30,
        slots_total=200,
        slots_queried=180,
        digests=frozenset(account_digest(username) for username in usernames),
    ).write(path)


@pytest.fixture
def marked_corpus(
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> Callable[[Path], tuple[Path, Path]]:
    """Return a factory writing three held-out games by six named players."""

    def build(tmp_path: Path) -> tuple[Path, Path]:
        return write_corpus(
            tmp_path / "corpus",
            [
                normalized_row(1, split="train"),
                normalized_row(4, split="test"),
                normalized_row(5, split="test"),
                normalized_row(6, split="test"),
            ],
        )

    return build


def test_a_marked_account_of_either_colour_takes_its_games_out_of_the_pool(
    tmp_path: Path,
    marked_corpus: Callable[[Path], tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    """The rejection acts on the account, so the colour it played is immaterial."""

    normalized, manifest = marked_corpus(tmp_path)
    snapshot = _snapshot(tmp_path / "marked-accounts.txt", "white4", "black5")

    result = freeze_pool(
        _resolved(normalized, manifest, marked_accounts=str(snapshot)),
        tmp_path / "pool",
    )

    assert load_pool(tmp_path / "pool").game_ids == (fixture_game_id(6),)
    assert result.games == 1


def test_a_widened_snapshot_is_caught_as_a_missing_game(
    tmp_path: Path,
    marked_corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """Rejection is the door a comparison of settings would not have watched.

    The fraction and the sample seed are untouched here; a later census simply
    knows more. That is why the check is stated in games: nothing about the
    configuration changed, and a generation still lost one.
    """

    normalized, manifest = marked_corpus(tmp_path)
    later = _snapshot(tmp_path / "later.txt", "white4")

    first = freeze_pool(_resolved(normalized, manifest), tmp_path / "first")

    with pytest.raises(EvaluationPoolError, match="drops 1 game"):
        freeze_pool(
            _resolved(
                normalized,
                manifest,
                marked_accounts=str(later),
                predecessor=str(tmp_path / "first"),
                predecessor_game_ids_sha256=first.game_ids_sha256,
            ),
            tmp_path / "second",
        )


def test_the_recall_the_snapshot_claimed_is_recorded_with_the_generation(
    tmp_path: Path,
    marked_corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """`0047` puts this reading at designation, so the generation carries it.

    A snapshot rejects the marks the census had reached rather than every mark
    there is, and containment makes the cut permanent, so what the pool claims
    has to be readable off the pool rather than off whichever snapshot file is
    still on disk later. The count goes with it because a rejection nobody can
    audit is indistinguishable from one that never ran.
    """

    normalized, manifest = marked_corpus(tmp_path)
    snapshot = _snapshot(tmp_path / "marked-accounts.txt", "white4", "black5")

    result = freeze_pool(
        _resolved(normalized, manifest, marked_accounts=str(snapshot)),
        tmp_path / "pool",
    )

    assert json.loads(result.manifest_path.read_text())["marked_accounts"] == {
        "snapshot_sha256": file_sha256(snapshot),
        "covers_archives": ["0" * 64],
        "queried_at": "2026-08-13",
        "accounts_total": 40,
        "accounts_queried": 30,
        "accounts_marked": 2,
        "slots_total": 200,
        "slots_queried": 180,
        "slot_coverage": 0.9,
        "rejected_games": 2,
    }


def _spanning_archives(manifest: Path, *archive_sha256s: str) -> None:
    """Rewrite the fixture manifest as a corpus prepared from several archives."""

    record = json.loads(manifest.read_text())
    del record["input"]
    record["inputs"] = [
        {"file_name": f"fixture-{index}.pgn", "sha256": archive_sha256}
        for index, archive_sha256 in enumerate(archive_sha256s)
    ]
    manifest.write_text(json.dumps(record))


def test_a_snapshot_has_to_cover_every_archive_the_corpus_holds(
    tmp_path: Path,
    marked_corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """A corpus spans archives, and a snapshot short of the span is refused.

    An archive the census never counted is one whose accounts nobody was asked
    about, so filtering the months a snapshot knows and keeping the rest whole
    would cut a generation that reads as filtered and is not.
    """

    normalized, manifest = marked_corpus(tmp_path)
    _spanning_archives(manifest, "0" * 64, "1" * 64)
    partial = _snapshot(tmp_path / "partial.txt", "white4", covers=("0" * 64,))
    whole = _snapshot(tmp_path / "whole.txt", "white4", covers=("0" * 64, "1" * 64))

    with pytest.raises(EvaluationPoolError, match=f"does not cover archive {'1' * 64}"):
        freeze_pool(
            _resolved(normalized, manifest, marked_accounts=str(partial)),
            tmp_path / "refused",
        )

    assert (
        freeze_pool(
            _resolved(normalized, manifest, marked_accounts=str(whole)),
            tmp_path / "pool",
        ).games
        == 2
    )


def test_a_manifest_that_names_no_archive_refuses_the_snapshot_it_cannot_check(
    tmp_path: Path,
    marked_corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """A coverage check that silently did not happen is the failure to be loud about."""

    normalized, manifest = marked_corpus(tmp_path)
    record = json.loads(manifest.read_text())
    del record["input"]
    manifest.write_text(json.dumps(record))
    snapshot = _snapshot(tmp_path / "marked-accounts.txt", "white4")

    with pytest.raises(EvaluationPoolError, match="digest of every archive"):
        freeze_pool(
            _resolved(normalized, manifest, marked_accounts=str(snapshot)),
            tmp_path / "pool",
        )


def test_a_pool_the_snapshot_emptied_blames_the_snapshot(
    tmp_path: Path,
    marked_corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """The fraction's message would send a reader to the wrong setting."""

    normalized, manifest = marked_corpus(tmp_path)
    snapshot = _snapshot(tmp_path / "marked-accounts.txt", "white4", "black5", "white6")

    with pytest.raises(EvaluationPoolError, match="rejected as a marked account"):
        freeze_pool(
            _resolved(normalized, manifest, marked_accounts=str(snapshot)),
            tmp_path / "pool",
        )


def test_coverage_makes_thin_slices_visible(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    coverage = json.loads(result.manifest_path.read_text())["coverage"]
    assert coverage["games"] == 2
    assert coverage["plies"] == {
        "total": 12,
        "minimum_per_game": 4,
        "maximum_per_game": 8,
    }
    assert coverage["results"] == {"0-1": 1, "1-0": 1}
    assert coverage["clock_presence_games"] == {"absent": 1, "present": 1}
    # Thirteen scored positions over twelve plies: one pool game ends in a
    # resignation its loser made on their own turn, which is a decision the
    # model is scored at even though it moves nothing.
    assert coverage["color_positions"] == {"black": 6, "white": 7}
    assert sum(coverage["phase_positions"].values()) == 13
    assert sum(coverage["legal_move_count_positions"].values()) == 13
    assert coverage["rating_band_positions"] == {"1200_to_1599": 5, "under_1200": 8}
    assert coverage["positions_without_rating"] == 0


def test_expected_identity_rejects_a_pool_that_changed(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """A checked-in digest turns silent drift into a build failure."""

    normalized, manifest = corpus(tmp_path)
    baseline = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    matching = _resolved(
        normalized,
        manifest,
        expected_game_ids_sha256=baseline.game_ids_sha256,
    )
    assert freeze_pool(matching, tmp_path / "pool").games == 2

    with pytest.raises(EvaluationPoolError, match="expected identity"):
        freeze_pool(
            _resolved(normalized, manifest, expected_game_ids_sha256="0" * 64),
            tmp_path / "pool",
        )


def test_freezing_is_reproducible(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    first = freeze_pool(_resolved(normalized, manifest), tmp_path / "first")
    second = freeze_pool(_resolved(normalized, manifest), tmp_path / "second")

    assert first.game_ids_sha256 == second.game_ids_sha256
    assert first.games_path.read_bytes() == second.games_path.read_bytes()


def test_scanning_on_several_workers_freezes_the_same_pool(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """How the scan was divided must not reach the artifact.

    Shards complete in whatever order the workers finish them, so the pool's
    identity would follow the machine rather than the corpus if anything
    downstream of the merge depended on arrival order.
    """

    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [normalized_row(index, split="test") for index in range(60)],
        games_per_shard=7,
    )

    serial = freeze_pool(
        _resolved(normalized, manifest, sample_fraction=0.5),
        tmp_path / "serial",
    )
    parallel = freeze_pool(
        _resolved(normalized, manifest, sample_fraction=0.5),
        tmp_path / "parallel",
        workers=4,
    )

    assert parallel.games == serial.games
    assert parallel.game_ids_sha256 == serial.game_ids_sha256
    assert parallel.games_path.read_bytes() == serial.games_path.read_bytes()


def test_the_overlap_check_sees_a_duplicate_split_across_shards(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The leakage check is the one thing a shard cannot answer alone.

    Its two halves are found by different workers, so a merge that dropped
    either would turn a failing build into a silently leaking pool.
    """

    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [normalized_row(index, split="train") for index in range(40)]
        + [normalized_row(index, split="test") for index in range(40)],
        games_per_shard=40,
    )

    with pytest.raises(EvaluationPoolError, match="also appear in the train split"):
        freeze_pool(
            _resolved(normalized, manifest, sample_fraction=0.5),
            tmp_path / "pool",
            workers=4,
        )


def test_load_pool_round_trips_and_exposes_game_level_facts(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    normalized, manifest = corpus(tmp_path)
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    pool = load_pool(tmp_path / "pool")

    assert pool.game_ids == tuple(sorted((fixture_game_id(4), fixture_game_id(5))))
    by_id = {game.game_id: game for game in pool.games}
    assert by_id[fixture_game_id(4)].ply_count == 4
    assert by_id[fixture_game_id(4)].result == "0-1"
    assert by_id[fixture_game_id(5)].ply_count == 8
    assert by_id[fixture_game_id(5)].has_ratings is True


def test_a_loaded_game_carries_the_class_its_own_clock_derives(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    """A view slices on this, and the increment is why bounds cannot stand in.

    One minute plus three seconds is blitz where one minute alone is bullet, so
    a class runs diagonally across the two columns rather than along one.
    """

    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [
            normalized_row(1, split="test", time_initial_ms=60_000),
            normalized_row(
                2, split="test", time_initial_ms=60_000, time_increment_ms=3_000
            ),
            normalized_row(
                3, split="test", time_initial_ms=None, time_increment_ms=None
            ),
        ],
    )
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    by_id = {game.game_id: game for game in load_pool(tmp_path / "pool").games}

    assert by_id[fixture_game_id(1)].speed is Speed.BULLET
    assert by_id[fixture_game_id(2)].speed is Speed.BLITZ
    assert by_id[fixture_game_id(3)].speed is None


def test_a_dated_era_survives_the_pool_cut_and_slices_through_a_view(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    """An era reading is a view over the pool, not a second pass over the corpus."""

    normalized, manifest = write_corpus(
        tmp_path / "corpus",
        [
            normalized_row(1, split="test", source_date=date(2019, 12, 31)),
            normalized_row(2, split="test", source_date=date(2020, 6, 1)),
            normalized_row(3, split="test", source_date=None),
            normalized_row(4, split="train"),
        ],
    )
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    pool = load_pool(tmp_path / "pool")
    selection = apply_view(
        pool.games,
        ViewConfig(name="pandemic", minimum_date=date(2020, 1, 1)),
    )

    assert {game.game_id: game.source_date for game in pool.games} == {
        fixture_game_id(1): date(2019, 12, 31),
        fixture_game_id(2): date(2020, 6, 1),
        fixture_game_id(3): None,
    }
    assert selection.game_ids == (fixture_game_id(2),)
    assert selection.excluded_games == {
        "before_minimum_date": 1,
        "missing_date": 1,
    }


def test_a_second_load_reuses_the_parsed_games_without_reading_again(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep loads one pool once per benchmark, and the parse is the cost."""

    normalized, manifest = corpus(tmp_path)
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")
    first = load_pool(tmp_path / "pool")

    def unreadable(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        raise AssertionError("a repeated load re-read the pool")

    monkeypatch.setattr(pool_module, "read_normalized_rows", unreadable)
    second = load_pool(tmp_path / "pool")

    assert second.games == first.games


def test_a_reused_load_still_verifies_the_recorded_identity(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """Reuse saves the parse and nothing else the load exists to do."""

    normalized, manifest = corpus(tmp_path)
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")
    load_pool(tmp_path / "pool")

    manifest_path = tmp_path / "pool/manifest.json"
    record = json.loads(manifest_path.read_text())
    record["identity"]["game_ids_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(record))

    with pytest.raises(EvaluationPoolError, match="recorded identity"):
        load_pool(tmp_path / "pool")


def test_a_pool_rewritten_in_place_is_loaded_again_rather_than_remembered(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    """Reuse is keyed on the artifact's checksum, not on where it sits."""

    normalized, manifest = corpus(tmp_path)
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")
    assert load_pool(tmp_path / "pool").game_ids == tuple(
        sorted((fixture_game_id(4), fixture_game_id(5)))
    )

    replacement, replacement_manifest = write_corpus(
        tmp_path / "replacement",
        [normalized_row(9, split="test")],
    )
    freeze_pool(_resolved(replacement, replacement_manifest), tmp_path / "pool")

    assert load_pool(tmp_path / "pool").game_ids == (fixture_game_id(9),)


def test_load_pool_rejects_a_tampered_artifact(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")
    (tmp_path / "pool/games.parquet").write_bytes(b"corrupted")

    with pytest.raises(EvaluationPoolError, match="checksum mismatch"):
        load_pool(tmp_path / "pool")


def test_load_pool_rejects_an_incompatible_benchmark_version(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")
    manifest_path = tmp_path / "pool/manifest.json"
    record = json.loads(manifest_path.read_text())
    record["benchmark_version"] = BENCHMARK_VERSION + 1
    manifest_path.write_text(json.dumps(record))

    with pytest.raises(EvaluationPoolError, match="benchmark version"):
        load_pool(tmp_path / "pool")


def test_load_pool_names_the_stale_schema_rather_than_the_column_it_lacks(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """A pool holds whole rows, so an older generation is missing columns."""

    normalized, manifest = corpus(tmp_path)
    freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")
    manifest_path = tmp_path / "pool/manifest.json"
    record = json.loads(manifest_path.read_text())
    record["schema_version"] = SCHEMA_VERSION - 1
    manifest_path.write_text(json.dumps(record))

    with pytest.raises(EvaluationPoolError, match="normalized schema version"):
        load_pool(tmp_path / "pool")


def test_a_pool_from_another_generation_is_refused_by_what_pinned_one(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    fixture_game_id: Callable[[int], int],
) -> None:
    """Every other check asks only whether the pool is intact and readable.

    A superseded pool left on disk passes all of them, so the digest a reader
    was defined over is the one thing that can tell it from the pool the
    reading belongs to. A reader that pins nothing still loads it.
    """

    normalized, manifest = corpus(tmp_path)
    superseded = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")
    grown, grown_manifest = write_corpus(
        tmp_path / "grown",
        [
            normalized_row(4, split="test", plies=4, result="0-1"),
            normalized_row(5, split="test", plies=8, rating=900, clocks=False),
            normalized_row(6, split="test"),
        ],
    )
    later = freeze_pool(_resolved(grown, grown_manifest), tmp_path / "later")
    games = tuple(sorted((fixture_game_id(4), fixture_game_id(5))))

    assert load_pool(tmp_path / "pool").game_ids == games
    assert (
        load_pool(
            tmp_path / "pool",
            expected_game_ids_sha256=superseded.game_ids_sha256,
        ).game_ids
        == games
    )

    with pytest.raises(EvaluationPoolError) as refusal:
        load_pool(tmp_path / "pool", expected_game_ids_sha256=later.game_ids_sha256)

    message = str(refusal.value)
    assert later.game_ids_sha256 in message
    assert superseded.game_ids_sha256 in message


def test_every_shipped_pin_names_a_generation_this_repository_freezes() -> None:
    """A generation cut has to move every selection reading that pool.

    The digest is written once by the freeze selection and repeated by each
    reader, and a half-updated cut otherwise surfaces as a refusal on whichever
    machine next runs the one benchmark that was missed.
    """

    selections = {
        path: tomllib.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (Path(__file__).parents[2] / "configs/evaluation").glob("*.toml")
        )
    }
    frozen = {
        document["expected_game_ids_sha256"]
        for document in selections.values()
        if "expected_game_ids_sha256" in document
    }
    read = [
        (path.name, digest)
        for path, document in selections.items()
        for digest in _pinned_digests(document)
    ]

    assert frozen
    assert read
    assert [pin for pin in read if pin[1] not in frozen] == []


def _pinned_digests(document: Mapping[str, Any]) -> list[str]:
    """Return the generations one selection pins, at whatever depth it sits."""

    pinned = []
    for key, value in document.items():
        if key == "expected_pool_game_ids_sha256":
            pinned.append(value)
        elif isinstance(value, Mapping):
            pinned.extend(_pinned_digests(value))
    return pinned


def test_an_empty_selection_fails_rather_than_writing_an_empty_pool(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    normalized, manifest = write_corpus(
        tmp_path / "corpus", [normalized_row(1, split="train")]
    )

    with pytest.raises(EvaluationPoolError, match="no normalized games"):
        freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")


def test_incompatible_source_preprocessing_is_rejected(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    record = json.loads(manifest.read_text())
    record["preprocessing_version"] = 1
    manifest.write_text(json.dumps(record))

    with pytest.raises(EvaluationPoolError, match="preprocessing version"):
        freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")


def test_designating_a_generation_records_the_core_it_becomes(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """Being the core is a recorded fact, not an inference from being first."""

    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(
        _resolved(normalized, manifest, pool_id="core-one", designates_core=True),
        tmp_path / "pool",
    )

    record = json.loads(result.manifest_path.read_text())["core"]
    assert record["core_id"] == "core-one"
    assert record["game_ids_sha256"] == result.game_ids_sha256
    assert record["games"] == result.games
    # The designating generation lists no ids, because its own are the core's.
    assert "game_ids" not in record
    assert result.core_id == "core-one"


def test_an_undesignated_generation_carries_no_core(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(_resolved(normalized, manifest), tmp_path / "pool")

    assert json.loads(result.manifest_path.read_text())["core"] is None
    assert load_pool(tmp_path / "pool").core is None
    assert result.core_id is None


def test_a_designated_core_loads_as_the_pool_it_designated(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    result = freeze_pool(
        _resolved(normalized, manifest, pool_id="core-one", designates_core=True),
        tmp_path / "pool",
    )

    pool = load_pool(tmp_path / "pool")

    assert pool.core is not None
    assert pool.core.core_id == "core-one"
    assert pool.core.game_ids == frozenset(pool.game_ids)
    assert pool.core.game_ids_sha256 == result.game_ids_sha256


def test_a_later_generation_carries_the_core_forward_unchanged(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """The continuous line: one set of games, in every generation after it."""

    rows = [normalized_row(index, split="test") for index in range(240)]
    smaller, smaller_manifest = write_corpus(tmp_path / "smaller", rows[:120])
    grown, grown_manifest = write_corpus(tmp_path / "grown", rows)

    designated = freeze_pool(
        _resolved(
            smaller,
            smaller_manifest,
            pool_id="core-one",
            sample_fraction=0.25,
            designates_core=True,
        ),
        tmp_path / "first",
    )
    freeze_pool(
        _resolved(
            grown,
            grown_manifest,
            pool_id="generation-two",
            sample_fraction=0.25,
            predecessor=str(tmp_path / "first"),
            predecessor_game_ids_sha256=designated.game_ids_sha256,
        ),
        tmp_path / "second",
    )

    first = load_pool(tmp_path / "first")
    second = load_pool(tmp_path / "second")

    assert second.core is not None and first.core is not None
    assert second.core.core_id == "core-one"
    assert second.core.game_ids == first.core.game_ids
    assert second.core.game_ids < frozenset(second.game_ids)


def test_a_second_designation_over_an_existing_core_is_refused(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """Designating twice would end every series the first designation holds."""

    rows = [normalized_row(index, split="test") for index in range(240)]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    first = freeze_pool(
        _resolved(
            normalized,
            manifest,
            pool_id="core-one",
            sample_fraction=0.25,
            designates_core=True,
        ),
        tmp_path / "first",
    )

    with pytest.raises(EvaluationPoolError, match="already carries the core"):
        freeze_pool(
            _resolved(
                normalized,
                manifest,
                pool_id="core-two",
                sample_fraction=0.5,
                designates_core=True,
                predecessor=str(tmp_path / "first"),
                predecessor_game_ids_sha256=first.game_ids_sha256,
            ),
            tmp_path / "second",
        )


def test_a_core_whose_games_do_not_match_its_digest_is_refused(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    result = freeze_pool(
        _resolved(normalized, manifest, pool_id="core-one", designates_core=True),
        tmp_path / "pool",
    )
    record = json.loads(result.manifest_path.read_text())
    record["core"]["game_ids_sha256"] = "0" * 64
    result.manifest_path.write_text(json.dumps(record))

    with pytest.raises(EvaluationPoolError, match="core whose games do not match"):
        load_pool(tmp_path / "pool")


def test_designation_reports_the_axes_it_fixes_permanently(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """Speed is the axis the breadth corpus exists for, so it must be readable."""

    normalized, manifest = corpus(tmp_path)

    result = freeze_pool(
        _resolved(normalized, manifest, pool_id="core-one", designates_core=True),
        tmp_path / "pool",
    )

    assert set(result.coverage_axes) == {
        "speed_games",
        "clock_presence_games",
        "results",
    }
    assert sum(result.coverage_axes["speed_games"].values()) == result.games


def test_a_core_with_no_id_is_refused(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    """An unnamed core would be inherited under that non-name by every cut after."""

    normalized, manifest = corpus(tmp_path)
    result = freeze_pool(
        _resolved(normalized, manifest, pool_id="core-one", designates_core=True),
        tmp_path / "pool",
    )
    record = json.loads(result.manifest_path.read_text())
    del record["core"]["core_id"]
    result.manifest_path.write_text(json.dumps(record))

    with pytest.raises(EvaluationPoolError, match="records a core with no id"):
        load_pool(tmp_path / "pool")


def test_a_core_whose_games_are_not_ids_is_refused(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
) -> None:
    normalized, manifest = corpus(tmp_path)
    result = freeze_pool(
        _resolved(normalized, manifest, pool_id="core-one", designates_core=True),
        tmp_path / "pool",
    )
    record = json.loads(result.manifest_path.read_text())
    record["core"]["game_ids"] = 5
    result.manifest_path.write_text(json.dumps(record))

    with pytest.raises(EvaluationPoolError, match="games are not a list"):
        load_pool(tmp_path / "pool")


def test_a_core_reaching_outside_the_generation_carrying_it_is_refused(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """Containment against the predecessor's pool does not cover its core.

    The two are different sets once a generation has been cut past designation,
    so a predecessor whose recorded core names a game it does not itself hold
    passes the containment check and would hand on a core nothing can score.
    """

    rows = [normalized_row(index, split="test") for index in range(240)]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    first = freeze_pool(
        _resolved(
            normalized,
            manifest,
            pool_id="core-one",
            sample_fraction=0.25,
            designates_core=True,
        ),
        tmp_path / "first",
    )
    record = json.loads((tmp_path / "first" / "manifest.json").read_text())
    escaped = sorted({*load_pool(tmp_path / "first").game_ids, 2**63 - 1})
    record["core"]["game_ids"] = escaped
    record["core"]["game_ids_sha256"] = pool_module.game_ids_sha256(escaped)
    (tmp_path / "first" / "manifest.json").write_text(json.dumps(record))

    with pytest.raises(EvaluationPoolError, match="this generation does not hold"):
        freeze_pool(
            _resolved(
                normalized,
                manifest,
                pool_id="generation-two",
                sample_fraction=0.5,
                predecessor=str(tmp_path / "first"),
                predecessor_game_ids_sha256=first.game_ids_sha256,
            ),
            tmp_path / "second",
        )


def test_a_missing_predecessor_is_refused_before_the_corpus_is_scanned(
    tmp_path: Path,
    corpus: Callable[[Path], tuple[Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Everything the predecessor decides is readable without touching a shard.

    The scan behind it is twenty minutes at best on a real corpus, which is
    long enough that learning a path is wrong at the end of one is what makes a
    re-cut something to avoid.
    """

    normalized, manifest = corpus(tmp_path)

    def _refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the corpus was scanned before the predecessor was read")

    monkeypatch.setattr(pool_module, "_scan_shards", _refuse)

    with pytest.raises(EvaluationPoolError, match="do not exist"):
        freeze_pool(
            _resolved(
                normalized,
                manifest,
                predecessor=str(tmp_path / "absent"),
            ),
            tmp_path / "pool",
        )

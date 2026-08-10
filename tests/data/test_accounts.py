from pathlib import Path

import pytest

from anthro_chess.data.accounts import (
    MarkedAccountError,
    MarkedAccounts,
    account_digest,
    load_marked_accounts,
    resolve_snapshot_path,
)

ARCHIVE_A = "a" * 64
ARCHIVE_B = "b" * 64


def _snapshot(
    *usernames: str, archives: tuple[str, ...] = (ARCHIVE_A,)
) -> MarkedAccounts:
    return MarkedAccounts(
        covers_archives=archives,
        queried_at="2026-08-08",
        accounts_total=10,
        accounts_queried=4,
        slots_total=100,
        slots_queried=75,
        digests=frozenset(account_digest(username) for username in usernames),
    )


def test_digests_ignore_the_case_a_pgn_happened_to_print() -> None:
    assert account_digest("BearpawNeptune") == account_digest("bearpawneptune")
    assert account_digest("  Punishing ") == account_digest("punishing")
    assert account_digest("someone") != account_digest("someone-else")


def test_round_trips_a_snapshot_without_storing_usernames(tmp_path: Path) -> None:
    snapshot = _snapshot("Cheater", "AlsoCheater", archives=(ARCHIVE_A, ARCHIVE_B))

    path = snapshot.write(tmp_path / "marked.txt")
    text = path.read_text(encoding="utf-8")
    reloaded = load_marked_accounts(path)

    assert "Cheater" not in text
    assert "AlsoCheater" not in text
    assert reloaded.digests == snapshot.digests
    assert reloaded.covers_archives == (ARCHIVE_A, ARCHIVE_B)
    assert reloaded.accounts_marked == 2
    assert reloaded.contains("cheater")
    assert not reloaded.contains("someone-honest")


def test_carries_the_coverage_the_census_had_reached(tmp_path: Path) -> None:
    """A snapshot states what it asked about rather than implying totality."""

    path = _snapshot("Cheater").write(tmp_path / "marked.txt")

    snapshot = load_marked_accounts(path)

    assert snapshot.accounts_queried == 4
    assert snapshot.accounts_total == 10
    assert snapshot.slot_coverage == pytest.approx(0.75)


def test_refuses_an_archive_the_snapshot_never_counted(tmp_path: Path) -> None:
    path = _snapshot("Cheater").write(tmp_path / "marked.txt")
    snapshot = load_marked_accounts(path)

    snapshot.require_archive(ARCHIVE_A)
    with pytest.raises(MarkedAccountError, match="does not cover archive"):
        snapshot.require_archive(ARCHIVE_B)


def test_rejects_a_snapshot_whose_body_contradicts_its_header(
    tmp_path: Path,
) -> None:
    path = _snapshot("Cheater", "AlsoCheater").write(tmp_path / "marked.txt")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(MarkedAccountError, match="but its header claims"):
        load_marked_accounts(path)


def test_rejects_a_snapshot_that_does_not_say_what_it_covered(tmp_path: Path) -> None:
    path = _snapshot("Cheater").write(tmp_path / "marked.txt")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"slots_queried": 75,', ""), encoding="utf-8")

    with pytest.raises(MarkedAccountError, match="how much of them it asked about"):
        load_marked_accounts(path)


def test_rejects_a_snapshot_built_with_another_salt(tmp_path: Path) -> None:
    path = _snapshot("Cheater").write(tmp_path / "marked.txt")
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("anthro-marked-accounts-v1", "other"), encoding="utf-8"
    )

    with pytest.raises(MarkedAccountError, match="digest salt"):
        load_marked_accounts(path)


def test_resolves_a_snapshot_against_the_selection_that_names_it() -> None:
    relative = Path("marked-accounts/lichess.txt")

    resolved = resolve_snapshot_path(relative, "configs/data/lichess-blitz.toml")

    assert resolved == Path("configs/data/marked-accounts/lichess.txt")
    absolute = Path("/srv/marked.txt")
    assert resolve_snapshot_path(absolute, None) == absolute
    with pytest.raises(MarkedAccountError, match="nothing to resolve it against"):
        resolve_snapshot_path(relative, None)

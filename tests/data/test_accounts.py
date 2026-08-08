from pathlib import Path

import pytest

from anthro_chess.data.accounts import (
    MarkedAccountError,
    MarkedAccounts,
    account_digest,
    load_marked_accounts,
    marked_accounts_from_usernames,
    resolve_snapshot_path,
    scan_archive_accounts,
)

ARCHIVE_A = "a" * 64
ARCHIVE_B = "b" * 64


def _snapshot(*usernames: str, archive: str = ARCHIVE_A) -> MarkedAccounts:
    return marked_accounts_from_usernames(
        usernames,
        archive_sha256=archive,
        queried_at="2026-08-08",
        accounts_queried=10,
    )


def test_digests_ignore_the_case_a_pgn_happened_to_print() -> None:
    assert account_digest("BearpawNeptune") == account_digest("bearpawneptune")
    assert account_digest("  Punishing ") == account_digest("punishing")
    assert account_digest("someone") != account_digest("someone-else")


def test_round_trips_a_snapshot_without_storing_usernames(tmp_path: Path) -> None:
    snapshot = _snapshot("Cheater", "AlsoCheater")

    path = snapshot.write(tmp_path / "marked.txt")
    text = path.read_text(encoding="utf-8")
    reloaded = load_marked_accounts(path)

    assert "Cheater" not in text
    assert "AlsoCheater" not in text
    assert reloaded.digests == snapshot.digests
    assert reloaded.covers_archive == ARCHIVE_A
    assert reloaded.accounts_marked == 2
    assert reloaded.contains("cheater")
    assert not reloaded.contains("someone-honest")


def test_refuses_an_archive_the_snapshot_never_asked_about(tmp_path: Path) -> None:
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


def test_rejects_a_snapshot_built_with_another_salt(tmp_path: Path) -> None:
    path = _snapshot("Cheater").write(tmp_path / "marked.txt")
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("anthro-marked-accounts-v1", "other"), encoding="utf-8"
    )

    with pytest.raises(MarkedAccountError, match="digest salt"):
        load_marked_accounts(path)


def test_resumes_a_query_without_re_asking_about_answered_accounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthro_chess.data import accounts as accounts_module

    names = [f"player{index:02d}" for index in range(10)]
    asked: list[str] = []

    def fake_post(batch: list[str]) -> list[dict[str, object]]:
        asked.extend(batch)
        if "player03" in batch:
            raise MarkedAccountError("source refused")
        return [
            {"username": name, "tosViolation": name == "player01"} for name in batch
        ]

    monkeypatch.setattr(accounts_module, "_post_usernames", fake_post)
    resume = tmp_path / "marked.txt.partial"

    with pytest.raises(MarkedAccountError):
        accounts_module.query_marked_accounts(
            names, batch_size=2, pause_seconds=0.0, resume_path=resume
        )

    # One batch landed before the second refused, and its answer survived.
    assert asked == names[:4]
    asked.clear()

    def fake_post_again(batch: list[str]) -> list[dict[str, object]]:
        asked.extend(batch)
        return [
            {"username": name, "tosViolation": name == "player07"} for name in batch
        ]

    monkeypatch.setattr(accounts_module, "_post_usernames", fake_post_again)

    marked = accounts_module.query_marked_accounts(
        names, batch_size=2, pause_seconds=0.0, resume_path=resume
    )

    # Resumed from the completed batch, so the refused one is retried and
    # nothing already answered is paid for twice.
    assert asked == names[2:]
    assert marked == {"player01", "player07"}


def test_refuses_progress_recorded_against_another_account_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthro_chess.data import accounts as accounts_module

    def fake_post(batch: list[str]) -> list[dict[str, object]]:
        return [{"username": name, "tosViolation": True} for name in batch]

    monkeypatch.setattr(accounts_module, "_post_usernames", fake_post)
    resume = tmp_path / "marked.txt.partial"
    accounts_module.query_marked_accounts(
        ["alpha", "beta"], batch_size=2, pause_seconds=0.0, resume_path=resume
    )

    with pytest.raises(MarkedAccountError, match="a different account list"):
        accounts_module.query_marked_accounts(
            ["gamma", "delta"], batch_size=2, pause_seconds=0.0, resume_path=resume
        )


def test_resolves_a_snapshot_against_the_selection_that_names_it() -> None:
    relative = Path("marked-accounts/lichess.txt")

    resolved = resolve_snapshot_path(relative, "configs/data/lichess-blitz.toml")

    assert resolved == Path("configs/data/marked-accounts/lichess.txt")
    absolute = Path("/srv/marked.txt")
    assert resolve_snapshot_path(absolute, None) == absolute
    with pytest.raises(MarkedAccountError, match="nothing to resolve it against"):
        resolve_snapshot_path(relative, None)


def test_scans_both_player_tags_from_a_plain_archive(tmp_path: Path) -> None:
    archive = tmp_path / "games.pgn"
    archive.write_text(
        '[Event "Rated Blitz game"]\n[White "One"]\n[Black "Two"]\n\n1. e4 e5 1-0\n'
        '[Event "Rated Blitz game"]\n[White "Two"]\n[Black "Three"]\n\n1. d4 d5 0-1\n',
        encoding="utf-8",
    )

    assert scan_archive_accounts(archive) == ["One", "Three", "Two"]

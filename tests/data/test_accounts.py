from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from anthro_chess.data.accounts import (
    LICHESS_USERS_BATCH,
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

    def fake_post(batch: list[str], token: str | None) -> list[dict[str, object]]:
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

    def fake_post_again(batch: list[str], token: str | None) -> list[dict[str, object]]:
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

    def fake_post(batch: list[str], token: str | None) -> list[dict[str, object]]:
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


def test_paces_from_the_limiter_and_a_token_buys_twice_the_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthro_chess.data import accounts as accounts_module

    assert accounts_module.sustainable_pause(300, authenticated=False) == pytest.approx(
        30.0
    )
    assert accounts_module.sustainable_pause(300, authenticated=True) == pytest.approx(
        15.0
    )

    seen: list[str | None] = []
    slept: list[float] = []

    def fake_post(batch: list[str], token: str | None) -> list[dict[str, object]]:
        seen.append(token)
        return []

    names = [f"player{index:04d}" for index in range(2 * LICHESS_USERS_BATCH)]
    monkeypatch.setattr(accounts_module, "_post_usernames", fake_post)
    monkeypatch.setattr(accounts_module.time, "sleep", slept.append)
    monkeypatch.setenv(accounts_module.LICHESS_TOKEN_VARIABLE, "  secret  ")
    accounts_module.query_marked_accounts(names)
    monkeypatch.setenv(accounts_module.LICHESS_TOKEN_VARIABLE, "")
    accounts_module.query_marked_accounts(names)

    # The pace a token buys is the one an unpaced call actually waits. The
    # tolerance is the real clock: the pause is an interval, so however long
    # the batch took comes off it.
    assert seen == ["secret", "secret", None, None]
    assert slept == [pytest.approx(15.0, abs=0.5), pytest.approx(30.0, abs=0.5)]


def test_refuses_a_batch_larger_than_the_source_answers_for() -> None:
    from anthro_chess.data import accounts as accounts_module

    with pytest.raises(MarkedAccountError, match="drops the excess silently"):
        accounts_module.query_marked_accounts(
            ["alpha"], batch_size=accounts_module.LICHESS_USERS_BATCH + 1
        )


def test_rides_out_one_refusal_and_stops_at_the_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthro_chess.data import accounts as accounts_module

    slept: list[float] = []
    attempts = 0

    def always_refuses(request: Request, timeout: float | None = None) -> None:
        nonlocal attempts
        attempts += 1
        raise HTTPError(accounts_module.LICHESS_USERS_ENDPOINT, 429, "", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(accounts_module, "urlopen", always_refuses)
    monkeypatch.setattr(accounts_module.time, "sleep", slept.append)

    with pytest.raises(MarkedAccountError, match="try again tomorrow"):
        accounts_module._post_usernames(["alpha"], None)

    assert attempts == 2
    assert slept == [accounts_module._BURST_WINDOW_SECONDS]


def test_the_pause_is_the_interval_rather_than_time_added_to_each_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthro_chess.data import accounts as accounts_module

    clock = iter([0.0, 4.0, 100.0, 115.0, 200.0])
    slept: list[float] = []

    monkeypatch.setattr(accounts_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(accounts_module.time, "sleep", slept.append)
    monkeypatch.setattr(
        accounts_module,
        "_post_usernames",
        lambda batch, token: [],
    )

    accounts_module.query_marked_accounts(
        ["alpha", "beta", "gamma"], batch_size=1, pause_seconds=10.0
    )

    # A four-second request leaves six of the ten, and a request slower than
    # the pause leaves none rather than a negative wait.
    assert slept == [6.0, 0.0]


def test_gives_up_immediately_on_a_refusal_that_is_not_a_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthro_chess.data import accounts as accounts_module

    def forbidden(request: Request, timeout: float | None = None) -> None:
        raise HTTPError(accounts_module.LICHESS_USERS_ENDPOINT, 403, "", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(accounts_module, "urlopen", forbidden)

    with pytest.raises(MarkedAccountError, match="cannot query account status"):
        accounts_module._post_usernames(["alpha"], None)


def test_sends_the_token_as_a_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    from anthro_chess.data import accounts as accounts_module

    captured: list[Request] = []

    def fake_urlopen(request: Request, timeout: float | None = None) -> BytesIO:
        captured.append(request)
        return BytesIO(b"[]")

    monkeypatch.setattr(accounts_module, "urlopen", fake_urlopen)

    accounts_module._post_usernames(["alpha"], "secret")
    accounts_module._post_usernames(["alpha"], None)

    assert captured[0].get_header("Authorization") == "Bearer secret"
    assert captured[1].get_header("Authorization") is None


def test_scans_both_player_tags_from_a_plain_archive(tmp_path: Path) -> None:
    archive = tmp_path / "games.pgn"
    archive.write_text(
        '[Event "Rated Blitz game"]\n[White "One"]\n[Black "Two"]\n\n1. e4 e5 1-0\n'
        '[Event "Rated Blitz game"]\n[White "Two"]\n[Black "Three"]\n\n1. d4 d5 0-1\n',
        encoding="utf-8",
    )

    assert scan_archive_accounts(archive) == ["One", "Three", "Two"]

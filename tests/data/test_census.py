from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from anthro_chess.data import census as census_module
from anthro_chess.data.census import (
    LICHESS_USERS_BATCH,
    ArchiveAccounts,
    CensusError,
    PinnedArchive,
    account_games,
    append_answers,
    count_archive_accounts,
    daily_account_allowance,
    read_account_games,
    read_answers,
    read_census,
    run_census,
    snapshot_from_census,
    sustainable_pause,
    write_account_games,
)

ARCHIVE_A = "a" * 64
ARCHIVE_B = "b" * 64


def _archive(tmp_path: Path, name: str, *games: tuple[str, str]) -> PinnedArchive:
    from hashlib import sha256

    path = tmp_path / name
    path.write_text(
        "".join(
            f'[Event "Rated Blitz game"]\n[White "{white}"]\n[Black "{black}"]\n\n'
            "1. e4 e5 1-0\n"
            for white, black in games
        ),
        encoding="utf-8",
    )
    return PinnedArchive(
        path=path,
        counts_path=tmp_path / f"{name}.accounts.tsv",
        sha256=sha256(path.read_bytes()).hexdigest(),
    )


def _counted(tmp_path: Path, name: str, archive: str, **games: int) -> PinnedArchive:
    counts_path = tmp_path / f"{name}.accounts.tsv"
    write_account_games(
        counts_path,
        ArchiveAccounts(archive_sha256=archive, games_by_account=dict(games)),
    )
    return PinnedArchive(path=tmp_path / name, counts_path=counts_path, sha256=archive)


def test_counts_both_player_tags_as_one_account_whatever_case_it_printed(
    tmp_path: Path,
) -> None:
    archive = _archive(
        tmp_path, "games.pgn", ("One", "Two"), ("TWO", "Three"), ("two", "One")
    )

    assert count_archive_accounts(archive.path) == {"one": 2, "two": 3, "three": 1}


def test_takes_the_pass_over_an_archive_at_most_once(tmp_path: Path) -> None:
    archive = _archive(tmp_path, "games.pgn", ("One", "Two"))

    counted = account_games(archive)
    archive.path.unlink()

    assert account_games(archive) == counted
    assert read_account_games(archive.counts_path).games_by_account == {
        "one": 1,
        "two": 1,
    }


def test_counts_an_archive_again_when_the_selection_pins_a_different_one(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path, "games.pgn", ("One", "Two"))
    account_games(archive)

    widened = _archive(tmp_path, "games.pgn", ("One", "Two"), ("Three", "Four"))

    assert len(account_games(widened).games_by_account) == 4


def test_refuses_an_archive_that_is_not_the_one_pinned(tmp_path: Path) -> None:
    archive = _archive(tmp_path, "games.pgn", ("One", "Two"))
    impostor = PinnedArchive(
        path=archive.path, counts_path=archive.counts_path, sha256=ARCHIVE_B
    )

    with pytest.raises(CensusError, match="not the .* this selection pins"):
        account_games(impostor)


def test_rejects_counts_written_by_something_this_code_cannot_read(
    tmp_path: Path,
) -> None:
    path = _counted(tmp_path, "counts", ARCHIVE_A, one=1).counts_path
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"format_version": 1', '"format_version": 9'))

    with pytest.raises(CensusError, match="delete it to count that archive again"):
        read_account_games(path)


def test_keeps_every_answer_and_drops_only_a_torn_one(tmp_path: Path) -> None:
    path = tmp_path / "answers.tsv"

    append_answers(path, {"alpha": True, "beta": False}, "2026-08-10")
    append_answers(path, {"gamma": False}, "2026-08-11")
    with path.open("a", encoding="utf-8") as store:
        store.write("delta\t1")

    assert read_answers(path) == {"alpha": True, "beta": False, "gamma": False}


def test_refuses_answers_it_cannot_read_rather_than_re_asking_silently(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answers.tsv"
    path.write_text("alpha\tmaybe\t2026-08-10\n", encoding="utf-8")

    with pytest.raises(CensusError, match="not an answer"):
        read_answers(path)


def test_queues_the_busiest_unanswered_accounts_across_every_counted_archive(
    tmp_path: Path,
) -> None:
    first = _counted(tmp_path, "first", ARCHIVE_A, alpha=5, beta=30, gamma=1)
    second = _counted(tmp_path, "second", ARCHIVE_B, alpha=40, delta=20)
    answers = tmp_path / "answers.tsv"
    append_answers(answers, {"beta": True}, "2026-08-10")

    census = read_census([first, second], answers)

    # Counts add across archives, so alpha outranks the busiest single archive.
    assert census.queue(10) == ["alpha", "delta", "gamma"]
    assert census.queue(1) == ["alpha"]
    assert census.archives == (ARCHIVE_A, ARCHIVE_B)
    assert census.accounts_total == 4
    assert census.slots_total == 96


def test_coverage_is_a_share_of_the_archives_the_snapshot_will_cover(
    tmp_path: Path,
) -> None:
    counted = _counted(tmp_path, "counts", ARCHIVE_A, alpha=30, beta=10)
    answers = tmp_path / "answers.tsv"
    append_answers(answers, {"alpha": True, "elsewhere": True}, "2026-08-10")

    census = read_census([counted], answers)
    snapshot = snapshot_from_census(census, queried_at="2026-08-11")

    # An account no counted archive holds is neither covered nor marked here.
    assert census.accounts_queried == 1
    assert snapshot.slots_queried == 30
    assert snapshot.slots_total == 40
    assert snapshot.accounts_marked == 1
    assert snapshot.covers_archives == (ARCHIVE_A,)


def test_records_every_account_asked_about_including_the_unanswered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = tmp_path / "answers.tsv"

    def fake_post(batch: list[str], token: str | None) -> list[dict[str, object]]:
        # The source omits accounts it has never heard of, and answers with its
        # own capitalization for the rest.
        return [
            {"id": name, "username": name.upper(), "tosViolation": name == "beta"}
            for name in batch
            if name != "gamma"
        ]

    monkeypatch.setattr(census_module, "_post_usernames", fake_post)

    run = run_census(
        ["alpha", "beta", "gamma"],
        answers,
        queried_at="2026-08-10",
        batch_size=3,
        pause_seconds=0.0,
    )

    assert run.accounts_asked == 3
    assert run.accounts_marked == 1
    assert not run.refused
    assert read_answers(answers) == {"alpha": False, "beta": True, "gamma": False}


def test_a_spent_allowance_keeps_what_it_answered_and_is_not_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = tmp_path / "answers.tsv"
    asked: list[str] = []

    def fake_post(batch: list[str], token: str | None) -> list[dict[str, object]]:
        if "player04" in batch:
            raise census_module.SourceExhausted("spent")
        asked.extend(batch)
        return [{"id": name, "tosViolation": name == "player01"} for name in batch]

    monkeypatch.setattr(census_module, "_post_usernames", fake_post)
    names = [f"player{index:02d}" for index in range(8)]

    run = run_census(
        names, answers, queried_at="2026-08-10", batch_size=2, pause_seconds=0.0
    )

    assert run.refused
    assert run.accounts_asked == 4
    assert asked == names[:4]
    assert set(read_answers(answers)) == set(names[:4])

    # A later run resumes from the accounts nobody has an answer for.
    counted = _counted(tmp_path, "counts", ARCHIVE_A, **{name: 1 for name in names})
    assert read_census([counted], answers).queue(10) == names[4:]


def test_paces_and_budgets_from_the_limiter_and_a_token_doubles_both(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert sustainable_pause(300, authenticated=False) == pytest.approx(30.0)
    assert sustainable_pause(300, authenticated=True) == pytest.approx(15.0)
    assert daily_account_allowance(300, authenticated=False) == 90_000
    assert daily_account_allowance(300, authenticated=True) == 180_000

    seen: list[str | None] = []
    slept: list[float] = []

    def fake_post(batch: list[str], token: str | None) -> list[dict[str, object]]:
        seen.append(token)
        return [{"id": name} for name in batch]

    names = [f"player{index:04d}" for index in range(2 * LICHESS_USERS_BATCH)]
    monkeypatch.setattr(census_module, "_post_usernames", fake_post)
    monkeypatch.setattr(census_module.time, "sleep", slept.append)
    monkeypatch.setenv(census_module.LICHESS_TOKEN_VARIABLE, "  secret  ")
    run_census(names, tmp_path / "answers.tsv", queried_at="2026-08-10")
    monkeypatch.setenv(census_module.LICHESS_TOKEN_VARIABLE, "")
    run_census(names, tmp_path / "answers.tsv", queried_at="2026-08-10")

    # The pace a token buys is the one an unpaced call actually waits. The
    # tolerance is the real clock: the pause is an interval, so however long
    # the batch took comes off it.
    assert seen == ["secret", "secret", None, None]
    assert slept == [pytest.approx(15.0, abs=0.5), pytest.approx(30.0, abs=0.5)]


def test_refuses_to_record_a_batch_the_source_answered_for_nobody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing re-asks, so a thin answer would be a permanent clean verdict."""

    answers = tmp_path / "answers.tsv"
    monkeypatch.setattr(census_module, "_post_usernames", lambda batch, token: [])

    with pytest.raises(CensusError, match="answered for none of"):
        run_census(["alpha"], answers, queried_at="2026-08-10", pause_seconds=0.0)

    assert not answers.exists()


def test_refuses_a_batch_larger_than_the_source_answers_for(tmp_path: Path) -> None:
    with pytest.raises(CensusError, match="drops the excess silently"):
        run_census(
            ["alpha"],
            tmp_path / "answers.tsv",
            queried_at="2026-08-10",
            batch_size=LICHESS_USERS_BATCH + 1,
        )


def test_the_pause_is_the_interval_rather_than_time_added_to_each_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = iter([0.0, 4.0, 100.0, 115.0, 200.0])
    slept: list[float] = []

    monkeypatch.setattr(census_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(census_module.time, "sleep", slept.append)
    monkeypatch.setattr(
        census_module,
        "_post_usernames",
        lambda batch, token: [{"id": name} for name in batch],
    )

    run_census(
        ["alpha", "beta", "gamma"],
        tmp_path / "answers.tsv",
        queried_at="2026-08-10",
        batch_size=1,
        pause_seconds=10.0,
    )

    # A four-second request leaves six of the ten, and a request slower than
    # the pause leaves none rather than a negative wait.
    assert slept == [6.0, 0.0]


def test_rides_out_one_refusal_and_calls_the_second_a_spent_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    attempts = 0

    def always_refuses(request: Request, timeout: float | None = None) -> None:
        nonlocal attempts
        attempts += 1
        raise HTTPError(census_module.LICHESS_USERS_ENDPOINT, 429, "", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(census_module, "urlopen", always_refuses)
    monkeypatch.setattr(census_module.time, "sleep", slept.append)

    with pytest.raises(census_module.SourceExhausted, match="the day's allowance"):
        census_module._post_usernames(["alpha"], None)

    assert attempts == 2
    assert slept == [census_module._BURST_WINDOW_SECONDS]


def test_gives_up_immediately_on_a_refusal_that_is_not_a_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(request: Request, timeout: float | None = None) -> None:
        raise HTTPError(census_module.LICHESS_USERS_ENDPOINT, 403, "", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(census_module, "urlopen", forbidden)

    with pytest.raises(CensusError, match="cannot query account status"):
        census_module._post_usernames(["alpha"], None)


def test_sends_the_token_as_a_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Request] = []

    def fake_urlopen(request: Request, timeout: float | None = None) -> BytesIO:
        captured.append(request)
        return BytesIO(b"[]")

    monkeypatch.setattr(census_module, "urlopen", fake_urlopen)

    census_module._post_usernames(["alpha"], "secret")
    census_module._post_usernames(["alpha"], None)

    assert captured[0].get_header("Authorization") == "Bearer secret"
    assert captured[1].get_header("Authorization") is None

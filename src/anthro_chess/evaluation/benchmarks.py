"""Every benchmark this build can run, and the one path that runs one.

A benchmark declares what it is once — the schema its selection is validated
against, which of the paths in that selection name shared artifacts, the errors
a bad reading raises, and the entry point that measures — and everything that
runs a benchmark reads the declaration rather than restating it.

This lived in ``suite.py`` as a suite-private adapter while the seven ``anthro
eval`` subcommands went around it, each naming the schema again, rooting the
same fields again, and writing out the same error tuple and the same call. The
convention was real and was held by repetition, so nothing could detect a
benchmark that drifted off it, and a cross-cutting addition to what an
invocation does cost seven edits.

Only the envelope. The measuring stays heterogeneous, because it is: the ladder
plays games, the inference benchmark reads no pool and times its own model
load, and decision decomposition consumes another step's output rather than a
selection of its own. A uniform interface over that variety would be worse than
the duplication it removed. What is uniform is the envelope around the
measuring, and only that lives here; :mod:`anthro_chess.evaluation.recording`
owns the other half, which is everything after a benchmark has finished.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anthro_chess.config import ConfigModel, ResolvedConfig, load_config
from anthro_chess.evaluation import roots

if TYPE_CHECKING:
    from anthro_chess.evaluation.games import GameRecord
    from anthro_chess.evaluation.results import DetailStore, ResultsStore


@dataclass(frozen=True)
class Benchmark:
    """What one benchmark declares to everything that runs it.

    The benchmarks are independent of their callers and stay that way: each
    owns its configuration schema, its entry point, and its errors. This is the
    one declaration of that, so a command and a sweep cannot disagree about
    what a benchmark takes, roots, or raises.
    """

    name: str
    #: ``None`` for a step whose input is another step's output.
    schema: type[ConfigModel] | None
    #: Which configured paths are rooted beneath the shared data root.
    artifact_fields: tuple[str, ...]
    #: Errors this benchmark raises for a bad reading. A caller converts them
    #: into its own failure — a failed sweep step, a command's exit status —
    #: rather than letting them end everything around it.
    errors: tuple[type[Exception], ...]
    #: The one of those the benchmark raises itself, and what the store's own
    #: errors are converted into while it runs. Declared rather than read as the
    #: first of ``errors``, which would make that order load-bearing with
    #: nothing saying so.
    error: type[Exception]
    #: ``(resolved, *, run_root, recording, ...) -> result``, called through
    #: :func:`run_benchmark`. A step configured by another step's output takes
    #: that payload instead and is invoked by whatever produced it.
    invoke: Callable[..., Any]
    #: Whether the benchmark writes results to the store at all. Decision
    #: decomposition does not: it has no result kind, and a decomposition over
    #: one payload is a diagnostic rather than a series.
    records_results: bool = True
    #: Returns the games this benchmark generated, when a later step reads them.
    games: Callable[[Any], tuple[GameRecord, ...]] | None = None
    #: Whether a configuration retained its games at all, checked while a sweep
    #: resolves so a dependent step cannot be starved an hour in.
    retains_games: Callable[[Any], bool] | None = None
    #: The step whose game payload this benchmark reads.
    games_from: str | None = None


def benchmark_registry() -> dict[str, Benchmark]:
    """Return every benchmark this build can run, keyed by its command name.

    Built on demand rather than at import, because the benchmarks pull in the
    model stack and a suite selection can be planned and printed without it.
    """

    from anthro_chess.evaluation import (
        CheckpointEvaluationConfig,
        CheckpointEvaluationError,
        InferenceBenchmarkConfig,
        InferenceBenchmarkError,
        LadderBenchmarkConfig,
        LadderBenchmarkError,
        LeakageError,
        NoveltyBenchmarkConfig,
        NoveltyBenchmarkError,
        PuzzleBenchmarkConfig,
        PuzzleBenchmarkError,
        RolloutBenchmarkConfig,
        RolloutBenchmarkError,
        TerminationBenchmarkConfig,
        TerminationBenchmarkError,
        benchmark_inference,
        benchmark_ladder,
        benchmark_novelty,
        benchmark_puzzles,
        benchmark_rollout,
        benchmark_termination,
        evaluate_checkpoint,
    )
    from anthro_chess.evaluation.decisions import DecisionDecompositionError
    from anthro_chess.evaluation.results import ResultsStoreError

    store_errors = (ResultsStoreError,)
    return {
        benchmark.name: benchmark
        for benchmark in (
            Benchmark(
                name="inference",
                schema=InferenceBenchmarkConfig,
                artifact_fields=(),
                errors=(InferenceBenchmarkError, *store_errors),
                error=InferenceBenchmarkError,
                invoke=benchmark_inference,
            ),
            Benchmark(
                name="run",
                schema=CheckpointEvaluationConfig,
                artifact_fields=roots.CHECKPOINT_ARTIFACT_FIELDS,
                errors=(CheckpointEvaluationError, LeakageError, *store_errors),
                error=CheckpointEvaluationError,
                invoke=evaluate_checkpoint,
            ),
            Benchmark(
                name="novelty",
                schema=NoveltyBenchmarkConfig,
                artifact_fields=roots.NOVELTY_ARTIFACT_FIELDS,
                errors=(NoveltyBenchmarkError, *store_errors),
                error=NoveltyBenchmarkError,
                invoke=benchmark_novelty,
            ),
            Benchmark(
                name="puzzles",
                schema=PuzzleBenchmarkConfig,
                artifact_fields=roots.PUZZLE_ARTIFACT_FIELDS,
                errors=(PuzzleBenchmarkError, *store_errors),
                error=PuzzleBenchmarkError,
                invoke=benchmark_puzzles,
            ),
            Benchmark(
                name="rollout",
                schema=RolloutBenchmarkConfig,
                artifact_fields=roots.ROLLOUT_ARTIFACT_FIELDS,
                errors=(RolloutBenchmarkError, *store_errors),
                error=RolloutBenchmarkError,
                invoke=benchmark_rollout,
                games=_rollout_games,
                retains_games=lambda config: bool(config.detail.retain_games),
            ),
            Benchmark(
                name="decisions",
                schema=None,
                artifact_fields=(),
                errors=(DecisionDecompositionError, OSError),
                error=DecisionDecompositionError,
                invoke=_invoke_decisions,
                records_results=False,
                games_from="rollout",
            ),
            Benchmark(
                name="termination",
                schema=TerminationBenchmarkConfig,
                artifact_fields=roots.TERMINATION_ARTIFACT_FIELDS,
                errors=(TerminationBenchmarkError, *store_errors),
                error=TerminationBenchmarkError,
                invoke=benchmark_termination,
            ),
            Benchmark(
                name="ladder",
                schema=LadderBenchmarkConfig,
                artifact_fields=roots.LADDER_ARTIFACT_FIELDS,
                errors=(LadderBenchmarkError, *store_errors),
                error=LadderBenchmarkError,
                invoke=benchmark_ladder,
            ),
        )
    }


def resolve_benchmark(
    benchmark: Benchmark,
    *,
    path: Path,
    overrides: Sequence[str] = (),
) -> ResolvedConfig[Any]:
    """Load one benchmark's selection and root the artifact paths it names.

    Rooting belongs to resolving rather than to each caller. A caller that
    forgot it would read a shipped selection's ``artifacts/`` path relative to
    the working directory, which resolves in a fresh clone and fails on a
    machine that keeps its artifacts anywhere else.
    """

    assert benchmark.schema is not None  # the caller screens a schema-less step
    resolved = load_config(benchmark.schema, path=path, overrides=overrides)
    return roots.resolve_artifact_roots(
        resolved,
        fields=benchmark.artifact_fields,
        overrides=overrides,
    )


def run_benchmark(
    benchmark: Benchmark,
    resolved_config: ResolvedConfig[Any],
    *,
    run_root: Path | None = None,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
    **measured: Any,
) -> Any:
    """Measure one benchmark against a resolved selection, and record it.

    Every benchmark that measures from a selection is invoked here, so what
    happens around a benchmark run is written once rather than by each of them:
    the recording is opened, held across the whole call — so a store error
    raised while measuring is converted like one raised while recording — and
    closed, which is what appends. The result is assembled from what was
    recorded, which is the last thing all seven used to do for themselves.

    ``measured`` carries what one benchmark's entry point takes beyond the
    shared call, which is a pre-loaded runner for the three that accept one.
    Neither the sweep nor a command passes any: it is how something that has
    already loaded a checkpoint measures with it rather than loading a second.
    """

    # Imported here for the reason the registry is built here: recording pulls
    # in the model stack, and a suite selection is planned and printed without
    # it.
    from anthro_chess.evaluation.recording import ResultRecording

    with ResultRecording(
        resolved_config,
        store=store,
        detail=detail,
        error=benchmark.error,
    ) as recording:
        result = benchmark.invoke(
            resolved_config,
            run_root=run_root,
            recording=recording,
            **measured,
        )
    return replace(result, **recording.fields)


def _rollout_games(result: Any) -> tuple[GameRecord, ...]:
    """Return every game a rollout retained, across its whole matrix."""

    return tuple(record for cell in result.cells for record in cell.records)


def _invoke_decisions(path: Path) -> Any:
    from anthro_chess.evaluation.decisions import decompose_game_records

    return decompose_game_records(path)


__all__ = [
    "Benchmark",
    "benchmark_registry",
    "resolve_benchmark",
    "run_benchmark",
]

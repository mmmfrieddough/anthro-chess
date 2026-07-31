# 0020: Declared Generation Settings Scope Generated-Play Series

Date: 2026-07-30

## Status

Accepted as initial design direction.

## Context

Decision 0013 built series fingerprints from realized inputs: the metric's
definition version, and a digest over the content of the games scored. Decision
0018 added one more realized input for efficiency metrics, the **declared
workload**, on the grounds that a latency figure taken at forty plies and one
taken at eighty measure different quantities.

Generated-play rollouts do not fit either half cleanly, and building the first
rollout benchmark is what made that concrete.

**They score no content.** A rollout plays new games. When it continues frozen
human prefixes it does read pool games, but as an *input to generation* rather
than as the content the metric was computed over. A data component would claim
the metric measured those games, which it did not.

**Their declared settings decide the quantity.** A rollout at temperature 0.2
and one at temperature 1.2 measure different things, as do rollouts at two
conditioning ratings, at two ply limits, or from two different position sources.
Without those in identity, a temperature grid would land every cell on one
series and a report would render a dial change as a checkpoint improvement.

The rollout matrix has to stay configurable, so freezing the grid into the
benchmark version was not an option: the issue driving this work requires
independently configurable rating and temperature grids, and the whole point of
the grid is to vary them.

## Decision

Generated-play series are scoped by their **declared generation settings**,
through the same workload mechanism decision 0018 introduced for efficiency.

The workload concept generalizes from "the settings that decide what was timed"
to "the settings that decide what was measured". Two costs may declare
themselves workload-scoped: timing execution, and generating games.

### What Is In The Workload

The arm, the conditioning rating, the temperature, the ply limit, whether colors
were swapped, whether the harness claimed draws, whether the model could resign,
and the identity of the position source. For a human-prefix arm the position
source carries the prefix depth and the pool, view, and game-id digest the
prefixes were projected from, so continuing a different set of openings or a
different depth is a different series.

A human-reference curve reading carries the same workload with one substitution
and one addition. The single conditioning rating is replaced by the whole rating
grid, because for a curve the grid is the axis rather than a setting held fixed.
And the declared curve shape joins it — the evaluation grid and the frozen
bandwidth — since a distance estimated at one smoothing and one estimated at
another are not the same quantity. That is the same reasoning that makes the
bandwidth a declared constant rather than configuration: it belongs to the
benchmark version, so changing it ends every curve series deliberately rather
than silently.

### What Is Not

Seed count, games per position, and concurrency. Generating more games estimates
the same distribution more precisely, exactly as scoring more games does, so
sample counts stay provenance. Concurrency changes only which kernels resolve a
decision — the measured suites showed identical games at every concurrency, with
recorded policy probabilities moving under `3e-6` — so putting it in identity
would end a series for a throughput change.

### A Generated Metric Declares Its Own Data Dependency

Cost alone cannot answer whether a generated reading consumes pool content,
because the two existing kinds differ: decision decomposition generates
decisions at positions a human reached and does read a projection, while a
rollout from the standard start reads none. So generating is the one cost for
which naming a projection is the metric's own statement rather than implied.
Every other cost still has to agree with its projection.

A rollout's human prefixes are still recorded on the envelope as a dataset
reference. That is provenance a reader consults to see which openings were
continued; it is deliberately not a fingerprint input, and the prefix truncation
reaches the digested rows so it cannot claim a depth it did not use.

### The Environment Stays Coordinates

Unchanged from 0018. Device, precision, Torch version, and coarse platform key
are recorded and are not in the fingerprint, so a rollout taken on a laptop and
one taken on a workstation stay on one line with the movement attributable.

## Consequences

One mechanism covers both workload-scoped kinds, so a report that already knows
how to attribute an efficiency delta needs nothing new to read a rollout series.

Every cell of a rollout matrix is its own series and its own stored result. That
is more records than a single-envelope benchmark writes, and it is the point: a
cell is the unit that has a comparable history, and the seeds inside it are the
replicates whose spread is that cell's evaluation noise.

The cost is that a rollout series is easy to end by accident, since the grid is
configuration rather than code. A reader who changes a temperature to explore
and then records the run gets a new series rather than a continued one. That is
the correct outcome and the reason the exploratory path can skip recording
entirely.

The raw rollout scalars are informational by construction. Human games are the
reference for whether generated play looks human, so declaring a direction on a
draw rate or a game length would assert a target the project has deliberately
declined to hardcode. One exception is the unfinished-game rate, which is a
defect rather than a behavior: an adjudicated game has no result or termination
to compare against anything.

The distances against matched human play are the other exception, and they are
where the family gets its direction. Closer to humans of the same rating is
better, which is the project's stated goal rather than an invented target, so
those metrics are the ones that rank two checkpoints. That is also why the raw
scalars are worth recording anyway: a distance says a checkpoint is unlike
humans, and only the scalars beside it say in which direction it erred.

## References

- `docs/evaluation.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0014-evaluation-result-storage.md`
- `docs/decisions/0018-workload-scoped-efficiency-series.md`

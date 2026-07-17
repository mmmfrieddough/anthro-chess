# Data

Anthro Chess should treat data as a reproducible pipeline output, not as a set
of hand-managed files. The data system should support many sources over time
while keeping a compact, source-agnostic normalized format for training and
evaluation.

The initial bulk source is expected to be the Lichess open database, especially
the universal exports with centisecond clock comments. The schema should not be
Lichess-specific. It should be a compact superset of useful chess-game concepts
that different sources can populate to different degrees.

## Data Principles

- Keep raw downloads, normalized records, and training-ready shards separate.
- Prefer reproducible scripts over manual data files.
- Store observed facts and compact labels; recompute deterministic chess state
  unless profiling shows a real bottleneck.
- Preserve source provenance without repeating large strings such as full URLs
  in hot training shards.
- Use explicit missingness for unavailable fields instead of overloading real
  values such as `0`.
- Allow examples with partial metadata. Games without clocks can still train
  move prediction, and games without ratings can still be useful in contexts
  where rating conditioning is not required.
- Keep the durable local dataset within a few hundred GB on consumer hardware.
- Record schema versions, preprocessing versions, source versions, and sampling
  recipes so training runs are reproducible.

## Pipeline Shape

The intended pipeline has three broad data layers:

```text
raw/          downloaded source archives, unchanged where practical
normalized/   source-agnostic game records in compact shards
training/     optional packed shards optimized for dataloading
manifests/    source, schema, checksum, split, and sampling metadata
```

Raw data may include compressed PGN archives, API JSON, source index files, and
other source-native artifacts. Raw files are useful for reproducibility but are
allowed to be treated as rebuildable cache when local storage is constrained.

Normalized data should be the durable local corpus. It should contain generic
fields and compact encodings, not source-specific PGN text.

Training shards may be generated from normalized data when performance requires
them. They can be less self-describing than normalized shards as long as their
manifests fully specify schema versions, offsets, checksums, and source inputs.

### Current Sample Path

The current implementation provides an importable PGN preparation API and the
thin `anthro data prepare` command. It validates standard games through
`python-chess`, converts moves with the shared action codec, writes one
source-agnostic Parquet row per game, and records source, input, output,
configuration, action-vocabulary, filtering, and deterministic split
provenance in a separate manifest.

The checked-in Lichess sample and its source selection provide an offline
reproduction path. Exact schema fields, versions, defaults, and filters are
owned by the data package and checked-in configuration rather than duplicated
here.

## Primary Source

The main initial source should be the Lichess open database:

- main database: <https://database.lichess.org/>
- universal centisecond-clock exports: <https://database.lichess.org/db-univ/>
- universal export counts: <https://database.lichess.org/db-univ/counts.txt>

The Lichess universal export currently provides the strongest fit for the core
timed model because it has very large game volume and centisecond clock comments
using `%clkc` in many timed games. The export spans 2013-01 through 2021-06 and
has roughly 3.7B games total. The 2017-04 through 2021-06 portion has roughly
3.35B games.

A local sample of the 2017-04 universal export suggested that about half of all
records were standard rated games with ratings, numeric time controls, and
centisecond clock comments. That implies a rough clean timing corpus on the
order of 1.5B-2.0B games, or tens to hundreds of billions of plies. The exact
usable count should be computed by the ingestion pipeline rather than assumed.

Lichess records need filters and tags. Early and universal exports can include
casual games, variants, unknown ratings, correspondence games, AI-level games,
tournament games, and games without usable clocks. Bot players are often
identifiable through title metadata such as `BOT`.

## Other Sources

Other data sources can be imported into the same normalized schema when useful.
They are not expected to replace Lichess as the initial bulk source.

FICS Games Database:

- <https://www.ficsgames.org/download.html>
- Online human games with ratings, rating deviations in later data, time
  controls, and some move-time information.
- Useful as a secondary online source and out-of-distribution evaluation data.

Chess.com public API:

- <https://www.chess.com/news/view/published-data-api>
- Player/month archive API with JSON metadata and PGN.
- Useful for supplemental games, cross-site validation, and high-profile player
  collections. It is not a simple bulk source like Lichess.

PGN Mentor:

- <https://www.pgnmentor.com/files.html>
- Historical and master-game PGN collections by player, opening, and event.
- Useful for famous-player or high-level style data, but usually lacks clocks
  and online-rating context.

Engine-game sources such as CCRL or TCEC may be useful for evaluation,
pipeline testing, or future engine-vs-human classifiers. They should not be
silently mixed into the main human-imitation training data.

## Logical Schema

The normalized schema should separate game-level metadata from per-ply sequence
data.

Required game-level fields should be small and structural:

- `schema_version`;
- internal `game_id`;
- compact `source_id`;
- compact `source_game_key` or source-native identifier;
- `ruleset`;
- `initial_position`;
- `result`;
- `ply_count`;
- compact action sequence or offsets into a packed action array.

Useful optional game-level fields include:

- player identifiers and names;
- player titles and bot flags;
- source ratings, rating namespaces, rating systems, rating differences, and
  rating deviations;
- normalized rating when the source rating is on a trusted or converted scale;
- rated/casual marker;
- time initial value and increment;
- termination reason;
- event or tournament identifiers;
- opening/ECO metadata;
- source license and provenance metadata;
- source final position for validation only.

Required per-ply fields should also be small:

- `game_id` or shard-local game index;
- `ply_index`;
- side to move, if not implied by ply index and initial position;
- compact action id.

Useful optional per-ply fields include:

- clock remaining after the move;
- derived move time;
- clock precision;
- timing validity flags;
- opening, structure, style, or player-style labels;
- evaluation metadata when a source provides it.

The schema should distinguish unavailable values from meaningful zero values.
For example, no clock data is different from a move that consumed zero
milliseconds, and unknown rating is different from a numeric rating.

## Missing Fields

Missing fields are allowed and should be represented explicitly. A missing
value must not be encoded as a real value such as `0`, empty string, or a
default rating. For each optional field, the normalized record should make it
possible to distinguish:

- the value is present;
- the value is absent because the source does not provide it;
- the value was expected but rejected because parsing or validation failed.

Model-facing encodings should preserve that distinction where it matters. For
example, clock context may use nullable values plus presence indicators, or
learned missing/untimed tokens, rather than pretending an unknown clock is a
zero clock.

Missing targets should also create loss masks. A record should contribute to
every objective it supports and contribute no loss to objectives it cannot
support. Games without clock data can train move prediction, but they should not
train the move-time head. Games without trusted ratings can train general move
imitation, but should not train rating-conditioned behavior directly.

Timing is both an input and a target. When clock data is present, it should be
available to the move model because human move choice depends on clock pressure.
When clock data is absent, the model should receive an explicit missing or
untimed representation, and the timing target should be masked out.

The ingestion pipeline should report coverage for important optional fields by
source, split, rating band, time setting, and other relevant slices. Partial
data is useful, but the training recipes should keep rich subsets, such as games
with reliable clocks, from being drowned out by larger subsets that lack those
fields.

## Identifiers And Provenance

Use compact internal identifiers for training data. An internal `game_id` can be
a deterministic `uint64` or `uint128` hash of the source id and source game key.
It is useful for joins, de-duplication, splits, debugging, and manifests.

`source_id` should be a compact enum, not a repeated source string. Full source
URLs should usually be reconstructable from a source template and
`source_game_key`, or stored only in debug/provenance tables.

Player identifiers should be kept when available because they may later support
player-style data. They should also use compact source-specific ids or
dictionary encoding in training shards.

## Rating Scale

The default rating scale should be Lichess-like because Lichess is expected to
provide the initial bulk of rating-conditioned training data.

The normalized schema should preserve original source ratings without assuming
that every source rating is directly comparable. Useful fields include:

- `source_rating_value`;
- `source_rating_namespace`, such as `lichess_blitz` or `chesscom_rapid`;
- `source_rating_system`, such as Glicko-2, Glicko, or unknown;
- `normalized_rating`, when available;
- `normalization_version`, when conversion was applied.

For initial training, use Lichess ratings directly as the normalized rating.
Other sources can still contribute move, style, player, opening, evaluation, or
general validation data even when their ratings are left out of
rating-conditioned training.

If non-Lichess ratings become important for rating-conditioned training, convert
them through an explicitly versioned normalization step. Simple affine mappings
or monotonic lookup tables are acceptable starting points, but the conversion
should be treated as approximate calibration metadata rather than ground truth.

## Actions

Training-ready data should store actions as normalized ids rather than SAN or
raw PGN text.

For standard chess and Chess960, ordinary moves can be represented as
from-square, to-square, and optional promotion. Castling can still be encoded as
the king move as long as chess logic interprets it correctly.

The action vocabulary should also support non-move or variant-specific actions:

- resignation;
- drops, such as crazyhouse `P@e4`;
- future variant-specific actions if variants become supported.

PGN/SAN should remain in raw source data or debug views. The training path
should use parsed, validated, compact action ids.

## Variants And Starting Positions

Standard chess should be the main initial target. The schema should still be
able to represent variants without redesign.

Lichess variant exports use ordinary PGN with a `Variant` header. Chess960
games also include `FEN` and `SetUp "1"` headers for the non-standard initial
position. Crazyhouse uses a `Variant "Crazyhouse"` header and drop notation
such as `P@e4`.

The normalized schema should therefore carry:

- `ruleset`;
- `initial_position`, defaulting to the standard start when omitted;
- action ids that are valid for the ruleset.

Unsupported variants can be filtered into separate corpora until the chess
logic and action vocabulary support them.

## Storage Format

Normalized data should start as sharded Parquet or Arrow files compressed with
zstd. This gives compact columnar storage, nullable fields, list columns,
schema metadata, and easy inspection with tools such as DuckDB, PyArrow, or
Polars.

A simple normalized row can represent one game with list columns for actions
and optional clocks:

```text
game_id
source_id
ruleset_id
initial_position
ratings and time-control metadata
actions: list<uint16 or uint32>
clock_remaining_ms: nullable list<int32>
```

If Parquet list columns become too slow or awkward for training, generate
packed training shards:

```text
games.parquet
actions.bin
clock_remaining_ms.bin
```

The `games.parquet` file can hold offsets and lengths into contiguous binary
arrays. This format is less self-describing, so it should be treated as a
derived cache with explicit manifests.

## Derived State And Legal Masks

Do not store full board positions or dense legal masks for every ply in the
canonical dataset.

The canonical data should store compact action sequences plus initial position
and ruleset. Deterministic chess logic should reconstruct board state and legal
moves during preprocessing, training, evaluation, and runtime.

Precomputed board states, legal masks, or checkpoint positions may be useful
for small evaluation sets, debugging, or derived training caches if profiling
shows the dataloader is the bottleneck. They should not become the default
source of truth.

The current model-facing encoding API reconstructs normalized standard games
into one typed example per ply. Each example carries the compact exact pre-move
state, trajectory alignment, normalized rating context, explicit timing
missingness, and legal action ids. Its versioned serialized identity is the
compatibility source of truth for future manifests, run records, and
checkpoints; exact field names and token mappings live with the implementation
rather than being duplicated here.

The initial sequence loader reads those normalized games into either full-game
sequences or contiguous fixed-length chunks. It packs framework-neutral numeric
batches, keeps nullable context behind explicit presence masks, reconstructs
legal actions per ply, and pads variable lengths behind attention and loss
masks. Its deterministic epoch order and explicit next-example cursor are the
restart boundary for training checkpoints.

## Approximate Scale

The project should reason about data in plies as well as games. A rough
conversion is:

```text
1M games   ~= 50M-90M plies
10M games  ~= 0.5B-0.9B plies
100M games ~= 5B-9B plies
```

Rough scale targets:

- `10k-100k` games: smoke tests for parsing, training, inference, and UCI.
- `1M` games: first real model and end-to-end evaluation loop.
- `10M+` games: useful rating and timing conditioning.
- `100M+` games: mature large-scale model, if compute and storage allow.

For `100M` games, compact normalized or training-ready storage should be
roughly in the tens to low hundreds of GB if actions, clocks, ratings, and
metadata are encoded numerically. Keeping raw compressed PGNs as well may push
local storage toward a few hundred GB.

Avoid repeated strings, URLs, SAN move text, per-ply FENs, full board states,
and dense legal masks in hot training shards.

## Sampling And Weighting

The full normalized corpus should preserve valid data, including
overrepresented groups. Training should use sampling recipes to decide how
often different slices appear.

Important sampling axes include:

- rating band;
- initial time;
- increment;
- game length or ply-count bucket;
- phase or ply index;
- opening family, when labels are available;
- source;
- ruleset;
- clock-data availability.

Balancing should avoid the full cross product becoming too sparse. For example,
it is reasonable to balance broadly by rating and time-control buckets, but
usually not to require every rating/time/opening/length combination to appear
equally.

Overrepresented buckets should generally be downsampled at training time rather
than physically pruned from the normalized corpus. This lets long training runs
still rotate through many examples while giving scarce useful groups more
learning pressure.

Some records should still be filtered or separated early:

- corrupt or unparseable games;
- unsupported variants for a standard-chess run;
- known bots or AI games from the main human corpus;
- malformed clocks or impossible metadata;
- abandoned zero-move games unless a specific task needs them.

Training runs should log both the intended recipe and the effective sample
distribution they actually saw.

Sampling recipes should be compatible with fine-grained training resume. Prefer
shard orders, within-shard shuffles, and sample cursors that can be
reconstructed from saved training state. Avoid making exact recovery depend on
large opaque random queues or dataloader internals when a simpler deterministic
recipe would give nearly equivalent coverage.

## Sequence Length And Phase Bias

Chess data naturally overrepresents early positions because every game starts
at move one while fewer games reach late middlegames or endgames.

The safest initial training setup is to train sequences from move one so
training matches live inference, where the game normally begins at move one and
the runtime accumulates history.

Use length bucketing to reduce padding waste and to make phase-aware evaluation
easy. Evaluation should report metrics by phase or ply bucket so good opening
performance does not hide weak late-game behavior.

Sampled midgame or endgame windows may be useful later for efficiency or
late-game coverage, but they introduce a train/inference mismatch if live
inference always has full history from move one. Treat windowed training as an
optimization to validate, not as the default assumption.

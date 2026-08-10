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

### Shared Machine-Local Data

Large corpora that should be reused across worktrees may live in a shared
machine-local directory outside every repository checkout. Set
`ANTHRO_CHESS_DATA_ROOT` to that directory. When a data command omits its
artifact directory, the CLI uses
`$ANTHRO_CHESS_DATA_ROOT/<configured-artifact-name>`. Acquisition writes the
verified archive under the archive selection's `raw/` directory. Preparation
reads that archive by default, where a selection pins exactly one, and writes
`normalized/` plus `manifests/` under the configured prepared-artifact
directory. This lets multiple prepared selections reuse one verified archive.
Explicit input and output paths still take precedence.

Worktrees should read and write the same verified archives, normalized shards,
and manifests directly beneath the shared root rather than copying them into
each checkout. Avoid running concurrent writers against the same corpus
directory.

### Current Sample Path

The current implementation provides an importable PGN preparation API and the
thin `anthro data prepare` command. It validates standard games through
`python-chess`, converts moves with the shared action codec, writes one
source-agnostic Parquet row per game, and records source, inputs, output,
configuration, action-vocabulary, filtering, and deterministic split
provenance in a separate manifest.

The checked-in Lichess sample and its source selection provide an offline
reproduction path. Exact schema fields, versions, defaults, and filters are
owned by the data package and checked-in configuration rather than duplicated
here.

### Current Baseline Corpus Path

The current implementation also provides the thin `anthro data acquire`
command and a checked-in bounded Lichess selection under `configs/data/`.
Acquisition downloads the exact configured monthly archive into the ignored
artifact root, verifies it against Lichess's published SHA-256 digest, and
reuses it only while that identity still matches.

Preparation reads Zstandard-compressed PGN directly, so it does not need a
second uncompressed copy. The baseline selection accepts one explicit Lichess
speed and rating namespace, rejects missing or invalid source ratings and bot
games, stops at a deterministic accepted-game bound, and writes bounded
Parquet shards through the existing shared PGN parser and action codec.
Game-id hashing keeps split assignments stable and ensures a duplicate source
game cannot cross the split boundary.

The initial recipe takes accepted games in source order until that bound. The
source is already limited to one month, so this trades some within-month
temporal coverage for a much shorter first preparation run while retaining
broad player, rating, position, game-length, and time-control variation. If
held-out results show that the selection is too small or narrow, the bound or
sampling recipe can change in configuration without creating another ingestion
path.

The manifest records every output shard and checksum, filter rejections, split
counts, ply ranges, and rating, time-control, and clock coverage. Exact release,
digest, selection size, filters, split recipe, and shard sizing remain owned by
the checked-in configuration. Raw archives and generated outputs remain
outside Git, and ordinary tests continue to use local fixtures.

### Building One Corpus From Many Archives

A selection may pin many archives, and preparation appends one of them per run:
each run takes one input, writes that archive's shards beside whatever is
already there, and rewrites the manifest to span every archive that has been
prepared. This is what lets a selection larger than the machine's disk be built
at all — fetch a month, prepare it, delete the archive, continue — and a run
names its own `--input` rather than being handed a default, because a selection
spanning archives has no single one.

Three properties make that safe to interrupt and resume. Shard names carry the
input's digest, so no two archives collide and a retried archive overwrites only
its own shards. The manifest records each archive's own digest, counts,
rejections and coverage, derives the corpus-wide totals from those parts rather
than carrying a running tally, and is replaced atomically, so the whole can
never disagree with the pieces and a kill mid-write cannot lose the record of
what is in. And an archive the manifest already records is left alone rather
than prepared twice, so re-running an interrupted pass from its beginning costs
nothing and changes nothing — including an archive every filter rejected, which
is recorded as an empty append rather than failing the pass.

One corpus directory still takes one writer at a time. An append reads the
manifest and rewrites it, so two runs against the same directory can each write
one that omits the other's archive, and the loser's shards are then swept as
orphans.

Preparation therefore only ever adds to a corpus. A selection whose source,
filters, split or termination choices differ from the ones the manifest recorded
is refused rather than half-applied, and the accepted-game bound counts the
corpus rather than each archive, so pinning 51 archives does not silently
multiply it by 51. Rebuilding under a changed selection means removing the
artifact directory or preparing into another one.

See `docs/decisions/0046-a-corpus-is-appended-one-archive-at-a-time.md`.

## Splits

Normalized games are partitioned three ways. `train` and `validation` behave as
usual: validation is consumed during training for validation loss and
checkpoint selection. `test` is held back from training entirely and is the
source of the frozen evaluation pool, so benchmark comparisons are not reported
on data the training loop has been selecting against. The training
configuration rejects a `test` selection outright.

Assignment is a pure function of the split seed and the internal game id.
Growing a corpus, changing its filters, or raising its game bound therefore
never moves an existing game between splits, which is what lets a frozen
benchmark stay safe as the corpus widens.

That guarantee depends entirely on the split seed. Changing it reassigns every
game and can move a previously held-out test game into training. Treat the seed
as frozen once a benchmark pool has been built from a selection, and prefer a
new pool version over a new seed. Exact fractions, seeds, and the assignment
algorithm live in the data configuration and `anthro_chess.data`.

See `docs/decisions/0011-held-out-test-partition.md`.

## Corpus Expansion

Corpus growth is the main expected source of change to evaluation inputs, so how
it is sequenced matters to more than data volume.

Expansion happens in two passes, breadth before depth. The **breadth** pass
widens the corpus across the axes the project intends to keep measuring, such as
time control, rating range, timing-data presence, and temporal or source spread.
The **depth** pass scales volume within those axes. Breadth comes first because
it is the irreversible one: a frozen evaluation reference can never measure an
axis it contains no games for, while adding volume to an axis already present
stays possible at any time.

Breadth is sized by evaluation power rather than by training needs. Each axis
needs enough held-out games to resolve the effects the project will want to
detect on it, which is a computable quantity given measured sampling noise, and
which is a much smaller number than training volume. Game-level benchmarks bind
here well before position-level ones do, since their unit is a game.

Expansion must preserve containment. The baseline recipe accepts games in source
order until a configured bound, so relaxing a filter without also raising that
bound can push previously accepted games past the cutoff and drop them.
Evaluation comparability depends on each pool generation being a superset of the
last, so an expansion should be configured to retain everything previously
accepted and the generation cut should verify containment rather than assume it.

Widening the corpus across an axis and training on that axis are separate steps,
and separating them resolves what otherwise looks like a conflict. Time control
is the case that matters first. The corpus and the pool should widen across
speeds before the core generation is cut, because breadth is the irreversible
half and a reference without an axis can never measure it. But training on a
mixture of speeds before the policy conditions on time control produces a model
that matches no speed exactly, including the one it matches today.

Both are satisfiable at once: widen the corpus and the pool, keep training
selection narrow, and slice benchmarks by speed from the start. Speed is
derivable from the schema's time fields, so slicing is a view-layer derivation
that needs no schema change. Training selection widens when the policy
conditions on time control, and the staged order gives that conditioning a
before-and-after picture to be judged against.

## Selecting Within A Corpus

Training selection filters within one broad corpus rather than relying on
preparation-time filters to produce a narrower one. It is a load-time property
of a run, not a property of the prepared artifact.

This is what makes the value of data measurable. Comparing a model trained on one
speed against a model trained on several requires both to be scored against one
evaluation reference containing both. Preparing two narrower corpora instead
produces two different references, so the comparison is not valid and the
fingerprints will not match. With selection as a load-time dial, the two runs
differ only in what they trained on, and the difference shows up sliced by axis.

The same dial answers how much data is worth acquiring. Subsampling the selection
deterministically gives a data-scaling curve against a fixed reference, which is
more informative for planning acquisition than any single comparison.

A model trained on a narrow selection will score poorly on the axes it never saw.
That is the measurement working rather than a regression, which is why these
comparisons need the axis slice and not only the aggregate.

The selection filters on the axes worth comparing models across, currently time
control and rating, and subsamples by ranking on a digest of the game id. That
rank is what makes a fraction reproducible on any machine and makes a smaller
fraction a subset of a larger one, so a data-scaling curve is a series of
nested selections rather than unrelated samples. A selection that matches no
games fails rather than starting a run on nothing.

Each run records the selection it resolved, not only the one it requested: the
counts it kept, why it excluded the rest, and a digest of the selected game ids.
The digest is what lets a later run confirm it reproduced the same games rather
than only the same configuration, and it is what tells two runs over one corpus
apart in the results store, where a result reaches its run through the
checkpoint's run id. Exact field names, axes, and defaults live in
`anthro_chess.data` rather than here.

Ply-count, result, and opening filters are deliberately absent. Benchmarks
measure the model's own distribution over those, so narrowing training on them
distorts the very quantity being read; unlike the axes above, that distortion is
not a comparison anyone wants to make.

See `docs/decisions/0013-benchmark-result-comparability.md`.

## Primary Source

The main initial source should be the Lichess open database:

- main database: <https://database.lichess.org/>
- universal centisecond-clock exports: <https://database.lichess.org/db-univ/>
- universal export counts: <https://database.lichess.org/db-univ/counts.txt>

The first bounded baseline uses a standard rated monthly export rather than the
much larger universal archive. It selects the first month in which the standard
exports include clock comments, then isolates one speed namespace. This keeps
the acquisition practical enough for the first training proof while preserving
a direct path to a larger or higher-precision selection through configuration
once downstream evidence justifies it.

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

The corpus the breadth pass builds is that universal export, 2017-04 through
2021-06, chosen for its clock precision.
`docs/decisions/0045-centisecond-clocks-from-a-closed-export.md` records why,
and what ending in mid-2021 costs.

Lichess records need filters and tags. Early and universal exports can include
casual games, variants, unknown ratings, correspondence games, AI-level games,
tournament games, and games without usable clocks. Bot players are often
identifiable through title metadata such as `BOT`.

## Admitting A New Source

A source may be mixed into the human corpus when its games are exchangeable with
existing games conditional on what the model conditions on. Any source-linked
variable that affects how people play and is not a conditioning input becomes a
confound the model silently averages over, and it lands on whichever benchmark
measures that behavior.

The test is not whether a source is high quality. It is whether it differs from
the existing corpus only along axes the model can represent.

A different site usually passes. The project is not imitating the players of one
platform, so the site itself is not a behavior axis worth preserving, and rating
scale differences are handled by normalization.

A different time control does not pass until the policy conditions on time
control, because time control genuinely changes both opening choice and game
length.

Curated historical and master collections fail on two axes that conditioning on
rating and time control does not reach. Opening theory moved across a century,
so era shifts the opening distribution on its own. And curated collections keep
notable and decisive games while online exports are exhaustive, so their result
and length distributions differ by construction of the collection rather than
by anything available to condition on. Their weighting risk is also conditional
rather than marginal: such a collection can be a negligible share of the corpus
while being most of the games in the high-rating, long-time-control region,
which is exactly where the reference is thinnest.

Admission is testable rather than assumed. Classify a candidate source's games
and compare its rating-conditional opening distribution against the existing
corpus. Agreement within the measured noise floor is evidence of exchangeability
on that axis; a systematic offset is evidence against it. The same comparison
diagnoses rating normalization, since a source whose 1600 has the repertoire of
the existing corpus's 1400 has a mapping problem rather than unusual players.

See `docs/decisions/0016-sampling-axes-versus-measured-distributions.md`.

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
- Fails the admission test above on era and curation selection, so it is not a
  candidate for the main human corpus. Its likely value is player-specific
  reference data for preference controls rather than bulk training data.

Engine-game sources such as CCRL or TCEC may be useful for evaluation,
pipeline testing, or future engine-vs-human classifiers. They should not be
silently mixed into the main human-imitation training data.

## Logical Schema

The normalized schema should separate game-level metadata from per-ply sequence
data.

A record is divided by what varies. Game-level values are scalars written once:
identity and provenance, the ruleset and initial position, the result, and the
rating and time-control metadata a decision may condition on. Per-ply values are
list columns aligned to the action sequence rather than rows keyed by ply, so a
game is one row and its plies travel with it — which is what lets a batch's rows
be read with a single columnar take.

Optional values carry a status beside them rather than a sentinel, because
absent and zero are different facts: no clock data is not a move that consumed
zero milliseconds, and an unknown rating is not a numeric one. A rating that is
missing, one present on an untrusted scale, and one that was converted are three
cases a reader has to be able to separate. So most optional values are a pair,
and the schema carries more columns than a list of concepts would suggest.

`anthro_chess.data.schema` declares those columns and their types, and is where
their names are written down.

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

Player identifiers are kept when a source provides them, as a fixed-width salted
digest of the account rather than the name. Membership is what a corpus-level
account filter needs and a digest serves it as well as a name does, so the
corpus does not repeat a source's usernames in readable form. That obscures
rather than protects, for the reason `anthro_chess.data.accounts` gives about
the snapshot digests it shares a salt with: the salt is public and the account
space is the archive's, so anyone holding the archive can rebuild the mapping.

Carrying them is what makes account-level filtering a property of the rows
rather than of the parse. A filter that can only run while reading PGN has to be
decided before preparation and costs a full re-parse to revisit, and whether
splitting on game id leaks a player between train and test cannot be asked at
all. Both stay answerable while the evaluation core is still unfrozen.

## Rating Scale

The default rating scale should be Lichess-like because Lichess is expected to
provide the initial bulk of rating-conditioned training data.

The normalized schema should preserve original source ratings without assuming
that every source rating is directly comparable. Each player's source rating is
kept beside the namespace and rating system it came from — a Lichess blitz
Glicko-2 number and a Chess.com rapid one are not the same quantity, and a
reader that treats them as one has silently pooled two scales. The normalized
value is a separate column rather than an overwrite, so revisiting the
conversion never destroys what the source actually said.

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
- claiming a draw by repetition or the fifty-move rule;
- drops, such as crazyhouse `P@e4`;
- future variant-specific actions if variants become supported.

PGN/SAN should remain in raw source data or debug views. The training path
should use parsed, validated, compact action ids.

The action vocabulary identity is stamped into both data and model artifacts, so
adding an action invalidates existing artifacts and starts a new benchmark
comparability series. Batch vocabulary additions into one deliberate change
rather than paying that cost repeatedly.

## Termination

Sources report how a game ended inconsistently and usually incompletely.
Lichess collapses resignation, agreed draw, stalemate, and checkmate into one
termination value, and collapses clock expiry and player abandonment into
another. Preprocessing should therefore derive a termination category rather
than relying on source text, while keeping the raw source value for provenance.

The derivation replays the game and combines the result, the source termination
field, and exact chess logic on the final position. It is defined over what a
PGN reports rather than over any one source's status vocabulary. Most categories
fall out directly: a decisive game whose source termination indicates normal
play is a resignation unless the final position is checkmate, and a drawn game
is an automatic draw when exact chess logic already considers the final position
terminal, a claimed draw when a claim was merely available, and a draw by
agreement otherwise.

Exact chess logic takes precedence wherever the final position is already
terminal, because the position is proof and the source field is not. A position
that is terminal in a way the reported result contradicts means the source
disagrees with itself, and the ending is recorded as unknown rather than
resolved in either direction. An ending the source genuinely cannot support
classifying is also recorded as unknown rather than guessed.

The derived category is stored alongside the raw source value, together with
whether the ending was attributable to the side to move. Terminal actions
belong in the action sequence when the game ended through a player's decision
and that player held the move. Resignations made on the opponent's clock, which
some platforms allow, have no decision point to attach to and are excluded from
the action sequence while the game's moves are kept. A claimed draw is excluded
for a second reason as well: the derivation accepts a claim the rules allow
only alongside an announced move, which the claim action cannot express, so the
action is appended only when the final position is claimable on its own. Each
game records whether a terminal action was appended and, when it was not, why,
so every exclusion is auditable rather than silent. Endings no player decided
stay distinct from decisions made off turn.

A terminal action is an action but not a ply. The stored ply count is the move
count, so appending one never changes a game's length, its ply filters, or the
prefix depth a benchmark takes from it. The per-ply columns stay aligned
one-to-one with the action sequence, so a trailing terminal action carries an
explicitly unavailable clock observation rather than a synthesized one, and
coverage over what the source reported counts move plies only.

Per-ply encoding scores a terminal action at the position the last move left,
against the actions that position enabled: its legal moves, resignation, and a
draw claim where the rules already allow one. Terminal actions are enabled at
every step rather than only where one was taken, because a player could always
have resigned; a model putting probability on resigning mid-game is making an
available choice rather than an illegal one.

Abandonment and clock expiry are not resignations and must not be relabelled as
such, even where clock traces make abandonment identifiable. Abandonment keeps
its own derived category so evaluation can compare against a reference
distribution that separates it from behavior the model can actually produce.
Separating the two uses the only evidence a PGN carries: the losing player's
remaining time as a share of their initial time at their last move, tested
against a configured threshold. Where a source exports no clocks the ending
stays the clock expiry the source itself reported, since promoting it to
unknown would discard a classification the source did support. The split
between the two populations depends on the source and time control, so
preparation reports the derived composition and how much of the time-forfeit
population the threshold was able to judge, rather than presenting the
threshold as settled.

Draw offers are not recorded by any source in scope and are out of scope
entirely. See
[`0017-derived-termination-and-terminal-actions.md`](decisions/0017-derived-termination-and-terminal-actions.md)
for the measurements and the reasoning.

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

One game is one row, with list columns carrying its actions and optional
clocks, so the format matches the unit a batch is read in.

Two storage choices are not what a reader wants back, and both earn the
indirection only because they were measured. A clock trace is differenced
against the same player's previous reading, which takes about a third off the
column. And the columns whose values never repeat are exempted from dictionary
encoding, which on those measured a quarter of the column spent for nothing. Reading a
clock therefore goes through `anthro_chess.data.schema`, which owns the codec
and the reasoning behind it; `anthro_chess.data.artifacts` owns the encoding
exemption.

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

The current model-facing encoding API reconstructs each normalized standard
game once. Every ply carries the compact exact pre-move state, both players'
observed action history, the side-to-move player's optional rating as a
decision target, explicit timing missingness, and the legal action ids a caller
asked to have reconstructed. Historical timestep contexts contain neither
player's rating, and loss is enabled on every valid ply. The same position
construction builds target-free live history without inventing an action target
or opponent rating; Anthro's single runtime target rating is attached only to
the current decision. The encoding's versioned serialized identity is the
compatibility source of truth for future manifests, run records, and
checkpoints; exact field names and token mappings live with the implementation
rather than being duplicated here.

The canonical normalized-artifact schema lives in `anthro_chess.data.schema`.
Preparation writes that schema, while loaders and other consumers select the
explicit projection they need from the same column contract. A normalized field
does not need to be a model input, but it should serve reconstruction, targets,
filtering, evaluation, provenance, compatibility, or debugging rather than be
retained without a concrete downstream purpose.

The sequence loading layer reads those normalized games into either full-game
sequences or contiguous fixed-length chunks. It packs framework-neutral numeric
batches, keeps nullable context behind explicit presence masks, and pads
variable lengths behind attention and loss masks. Length buckets keep similarly
sized sequences together, reducing padding without changing the examples. A
deterministic epoch plan and an explicit next-batch cursor are the restart
boundary for training checkpoints.

A batch is contiguous columns, each no wider than the values it carries, and
the per-ply encoding hands over its board as bytes so that a run of boards
joins into one rather than being read square by square. Both boundaries a batch
crosses are paid in that shape: a worker sends buffers, and the tensor
conversion wraps a column and copies it rather than visiting every timestep.
Widening to what a model indexes with belongs on the far side of that copy, so
what crosses to a device is the width the loader chose.

Where a nullable model input has a natural reserved token, the encoding assigns
it and the column carries that token instead of a value beside a presence flag.
`docs/decisions/0038-the-encoding-owns-token-vocabularies-the-model-owns-transforms.md`
owns where the encoding's business ends and the model's begins, and what that
means for checkpoint compatibility.

A live game's history carries the same column form and accumulates it while the
game is played, one row per ply beside the ply it encodes. What that buys is not
shared with the loader, which sees each ply once: a game asks for a batch after
every ply, so building one from the plies would re-walk the whole prefix per
decision and make a game quadratic in its own length. Extending buffers instead
makes a decision's batch a memory copy, flat in the history behind it.

Per-ply legal actions are reconstructed only when a caller reads them. Scoring
is the only consumer and training is not one, so a training loader asks for none
and the encoding it drives builds none — which is most of what decoding a game
costs, before any of what packing and pickling one costs. Asking is one signal,
given where the reconstruction happens; a batch then carries the set exactly
when the plies in it do. Refusing a target the position does not allow is asked
about that one candidate rather than of the set, so it holds either way.
Padding is right-aligned throughout, which is what lets a padded row's outputs
be ignored rather than masked away.

Two loaders provide that boundary, and a selection picks one by declaring a
streaming section or leaving it out. Both produce batches through the same
encoding and collation and expose the same identities and cursor; they differ
in what they hold.

The **eager** loader reconstructs and retains every selected per-ply encoding
before the first batch. A per-ply encoding is far larger than the normalized
row it came from, so this suits checked-in fixtures and bounded proof slices
and reaches neither the memory nor the startup time a corpus needs.

The **shard-backed** loader decodes a batch at a time. It first indexes the
selection from columns cheap enough to read for a whole corpus, which is
possible because a game's decoded length follows from its ply count and whether
a terminal action was appended, so no game is decoded to plan against it. An
epoch then orders row groups, orders the games inside each one, and cuts that
stream into planning windows. A window is where length buckets fill and flush,
so every example in a batch comes from one row group and a batch is read with a
single columnar take. Flushing at a window boundary rather than an epoch
boundary is the one visible cost: each window ends with a short batch per
occupied bucket, which `drop_last` drops and otherwise leaves slightly small.
What stays resident is one row group's projected columns, the index, and the
batches in flight, none of which grows with corpus size.
Because the plan follows from the index alone, a resumed run replays it to its
saved cursor without reading or decoding anything.

The two produce different orders and neither is a defect. A global shuffle over
a corpus means a seek per example, so the shard-backed loader shuffles row
groups and shuffles within them instead. Their identities differ accordingly,
which is what stops a run from continuing across the two and training on an
order it did not record.

Decoding is the expensive half and it parallelizes, so the shard-backed loader
can build batches in worker processes. Rows travel to a worker and a packed
batch comes back, which keeps every Parquet read sequential in one process.
Worker count and prefetch depth change how fast the same batches arrive and
never which examples share one, so they stay out of the identity a resumed run
has to match. Preparation's shard and row-group sizing is the remaining bound,
because a row group is the unit a batch's rows are read from.

**The depth is a rate, not an order.** That is worth stating because the two
dials were once coupled in a way that made it look otherwise. Jobs were
submitted only until `prefetch_batches` of them were outstanding, so the depth
also capped how many workers could ever hold a job: above it a worker never
received one, and a sweep of worker counts at a fixed depth measured the depth
rather than decode capacity. Doubling the depth at the *same* eight workers was
51% faster than adding workers had been. The loader now keeps
`workers + prefetch_batches` jobs outstanding, one per worker so none waits on
the consumer. Two 200-step runs under strict determinism, one either side of
that change, reached bit-identical parameters across all 47 tensors and the same
validation record — which is what says a throughput dial did not become an
ordering one.

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

Not every axis may be reweighted. Resampling is safe on an axis the model is
explicitly conditioned on, because it changes the marginal distribution of an
input rather than the conditional behavior benchmarks read. Resampling an axis
the model must reproduce unconditionally moves the very distribution its
benchmark measures, so the benchmark reports the sampling recipe instead of the
model. Weighting the loss by an axis is the same operation as resampling it.
`docs/decisions/0016-sampling-axes-versus-measured-distributions.md` owns this
rule and the axes currently closed under it.

Sampling axes available for balancing:

- rating band, a conditioning input;
- initial time and increment, once the policy conditions on time control;
- phase or ply index;
- source;
- ruleset;
- clock-data availability.

Axes that must not be reweighted, because a benchmark measures the model's own
distribution over them and the model does not condition on them:

- opening family;
- game length or ply-count bucket;
- result, termination, and repetition pattern.

Length-bucketed batching is not resampling and remains available: grouping
similar lengths within a batch serves the efficiency goal without changing how
often any game is seen.

Balancing should avoid the full cross product becoming too sparse. For example,
it is reasonable to balance broadly by rating and time-control buckets, but
usually not to require every rating/time/length combination to appear equally.

Overrepresented buckets should generally be downsampled at training time rather
than physically pruned from the normalized corpus. This lets long training runs
still rotate through many examples while giving scarce useful groups more
learning pressure.

Some records should still be filtered or separated early:

- corrupt or unparseable games;
- unsupported variants for a standard-chess run;
- known bots or AI games from the main human corpus;
- games played by an account the source has marked for breaking its rules;
- malformed clocks or impossible metadata;
- games a source cut short on a rules report rather than play;
- abandoned zero-move games unless a specific task needs them.

Filter for validity, never for balance. Every entry above is a statement that a
record is not a usable human game. Dropping valid games because their result,
termination, length, or repetition pattern is an inconvenient shape is a
different act, and it belongs in training selection rather than in preparation.

The distinction matters because preparation runs before split assignment, so a
preparation filter removes games from every split at once. It shifts the
training data and the evaluation reference in the same direction, and a
benchmark cannot detect a bias it shares with its own reference. Training
selection touches only the train split, so the same distortion becomes visible
as a mismatch against a clean reference, and it stays reversible. A preparation
filter is therefore a definitional statement about what counts as human play
for every later benchmark, and it should be recorded as one.

Engine assistance is the case that most tests that rule, and it is the
deliberate exception. Every game played by an account the source has marked is
rejected during preparation, so those games leave every split rather than only
the training one. The reason is that a reference containing engine-assisted
play measures the wrong target for a project whose whole subject is human-like
play, and the human-versus-engine classifier would otherwise draw its human
class from it. That is a definitional statement about what counts as human
play, it removes roughly a tenth of the games a month offers, and it is
recorded as one in
[`0041-games-of-marked-accounts-leave-the-corpus.md`](decisions/0041-games-of-marked-accounts-leave-the-corpus.md).

The baseline selection does not name a snapshot yet, so it prepares unfiltered
until one exists. Building it means asking the source about every account in an
archive, which it rate limits hard enough to take several sessions; the
selection carries the setting commented out rather than pointing at a file that
is not there, because preparation refuses a missing or non-covering snapshot
rather than quietly preparing without it.

The filter acts on accounts rather than on moves because no method separates
assisted moves from honest ones within a game at any useful confidence, while
the source publishes an account-level judgement for free. What that judgement
is, why it is pinned and checked in rather than queried during preparation, and
why widening the corpus fails loudly instead of quietly under-filtering are
owned by `configs/data/marked-accounts/`.

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

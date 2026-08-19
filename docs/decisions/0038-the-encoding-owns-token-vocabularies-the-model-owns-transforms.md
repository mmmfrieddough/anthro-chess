# 0038: The Encoding Owns Token Vocabularies, The Model Owns Transforms

Date: 2026-08-08

## Status

Accepted. Settles `#233`, and is the rule the next "should this move into the
loader?" question is answered against.

`0070-one-decision-per-pass-and-history-in-the-token-depth.md` removed the
previous-action input, so one of the two mappings below no longer exists. The
rule is unchanged and the surviving mapping still follows it: the repetition
count that replaced the input is the encoding's, and the perspective flip that
arrived with it is the model's.

## Context

`#197` found three fixups the model recomputed on every forward pass, each a
deterministic function of what the loader had already handed over:

1. the en-passant `+1` offset, reserving token 0 for "no en-passant square";
2. the previous-action fill value, substituting a reserved embedding row for
   "no previous action";
3. `log1p` on `halfmove_clock` and `fullmove_number`.

All three are the same shape of observation — the model is deriving something
per step that nothing per step decides — and it is tempting to treat them as one
item. They are not. Two of them are the encoding stating which integer means
what, and the third is the model stating how it wants to read an integer whose
meaning is already settled.

## The line

**A token vocabulary is the encoding's.** "Which row of the embedding does this
input occupy, and which row means it is absent" is a fact about the
representation. It is total, it is integer-to-integer, and every reader has to
agree on it or read the wrong row. Splitting it — the loader emits a square and
a flag, the model assembles the row — means the mapping is written in one place
and half-applied in another, and a second reader has to reproduce the second
half correctly to see the same input.

**A feature transform is the model's.** `log1p` on a counter is one of several
defensible ways to feed an unbounded integer to a network: a normalization, a
bucketing, and a raw value are all admissible, and which one is right is decided
by what trains well rather than by what the encoding means. The counter is
already exact and already unambiguous in the batch; the transform is the
model's opinion about it.

## Decision

**Items 1 and 2 move into the encoding. Item 3 stays in the model.**

`en_passant_token` and `previous_action_token` in `anthro_chess.data.encoding`
are now the single statement of both mappings, read by the sequence loader, by
the live-game decision history, and by the model sizing its embedding tables.
Absence is a row rather than a flag, so each input travels as one column
instead of a value beside a presence mask.

`log1p` on the two rule counters stays in the model's square-token encoder,
which is likewise where a rating that travels as an integer is turned into a
representation. Moving it would have made the encoding responsible for a
modelling choice, and would have widened two `int16` columns to `float32` on the
host side of a step whose measured problem is the host side.

## What this is not

It is not an argument from cost. `#198` measured forward and backward at 4.7 ms
against 0.8 ms of batch construction, and these are elementwise kernels on a
`(batch, sequence)` tensor. Moving them saves a few launches, which is not the
reason. Where the mapping is written is the reason; the saved kernels are a
consequence.

Nor is it a claim that the batch got much smaller. Removing two presence columns
takes a legal-action-free training batch at 16 by 150 from 267,877 to 262,995
bytes, about 1.8%, and takes a decision history's row from nine `int64` columns
to seven.

## Consequences

`encoding_identity()` does not change, and no checkpoint is refused.

That is the part worth stating plainly, because `#233` raised the opposite
possibility. What `encoding_identity()` protects is that a checkpoint's weights
still mean what they meant, and an embedding row's meaning is exactly what did
not change here: row 0 of the en-passant table was "no en-passant square"
before and after, and the row past the action vocabulary was "no previous
action" before and after. Only where the index is computed moved. Model outputs
for the same weights and the same games are bit-identical across both tensor
boundaries, which is what makes the identity's stability a verified claim rather
than an assumption.

The rule generalizes: a change that alters which row an input occupies is an
encoding change and moves `encoding_identity()` with it. A change that alters
only where an unchanged row index is computed does not.

# 0023: Series-Separated TensorBoard Checkpoint History

Date: 2026-07-31

## Status

Accepted. Refines `0014-evaluation-result-storage.md`.

## Context

Decision 0014 rejected putting cross-version checkpoint history into
TensorBoard because TensorBoard does not understand Anthro Chess comparability
fingerprints. A direct export would draw a continuous line through measurements
that decision 0013 says are different series.

The committed results store has since become the authoritative history and
records the raw fingerprint beside every scalar. That makes a safe projection
possible without asking TensorBoard to understand comparability.

## Decision

Checkpoint history may be projected into TensorBoard as a disposable view, with
one TensorBoard run per metric fingerprint. Runs for the same metric use the
same scalar tag, so TensorBoard groups them into one chart while every
fingerprint remains a separate line.

The step axis is the checkpoint's ordinal by first appearance in the results
store. Re-scoring an older checkpoint writes at its existing ordinal rather
than moving it to the end of history.

The projection is regenerated from the store and is never authoritative. Its
output stays outside the committed store, and the command replaces only a
directory marked as its own prior output.

This refines decision 0014's prohibition on cross-version TensorBoard history.
The reason for that prohibition still governs the design; raw measurements may
not be exported onto one continuous line.

## Consequences

TensorBoard provides a useful long-running visual view without becoming a
results database or weakening series identity.

Each fingerprint change produces another run and line, including a fingerprint
joined by an explicit bridge. TensorBoard therefore does not visualize bridge
semantics; the authoritative delta and history reports remain the place to read
bridged seams, noise floors, checkpoint labels, explicit absences, and
provenance.

The x-axis is an ordinal rather than a labeled checkpoint axis. A missing
metric appears as a missing chart or point rather than as an explicit absence.

## References

- `docs/evaluation.md`
- `docs/decisions/0013-benchmark-result-comparability.md`
- `docs/decisions/0014-evaluation-result-storage.md`

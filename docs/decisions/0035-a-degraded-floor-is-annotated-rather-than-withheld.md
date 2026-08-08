# 0035: A Floor The Paired Estimator Could Not Produce Is Annotated, Not Withheld

Date: 2026-08-08

## Status

Accepted. Supersedes the paired-floor consequence in
`0019-external-puzzle-calibration-set.md`, and settles the reporting half of
`#294` under `0033-pairing-is-a-correctness-fix-not-a-resolution-lever.md`.

## Context

The paired checkpoint-pair estimator needs both readings' retained per-unit
contributions. Those live in the machine-local detail tier while the summary
record is committed, so a comparison against a reading taken on another machine
routinely has nothing to difference. Decision 0033 keeps the estimator and names
this as work rather than as grounds for removing it.

Decision 0019 already stated a rule for that case: the paired floor is unknown,
and an independent-input floor must not be substituted. The code never did
either. It substituted silently where an independent-input floor existed, and
where the detail payload was merely absent it raised — ending the whole report
rather than one row, so `anthro eval report` over the committed store failed
outright on any machine that had not recorded it.

So three behaviors were in play against one written rule, and the question 0019
answered has to be answered again rather than assumed.

## Two ways to be loud

The substitution has to stop being silent. Beyond that there are two shapes, and
`#294` left the choice to the work.

**Withhold the floor.** What 0019 says. A report that cannot pair reports no
data-sampling floor, and the verdict is `unknown`.

**Annotate the reading.** Keep whatever floor applies, name the estimator that
produced it, and say on the row that the paired floor was expected and why it is
missing.

## Decision

**Annotate.** The reading stands with the floor it has, and the row states that
the floor is not the paired one and what stopped it.

The reason is the direction of the error. An unpaired floor drops the covariance
two checkpoints scored on one sample share, so it is about 1.9x too wide — it
errs conservative, and only in one direction. A delta that clears it has cleared
the paired floor as well, and that verdict is unaffected. Only `within` is
weakened, because the delta may be a real improvement the wider floor covers.
Withholding the floor discards the half of the instrument that still works: it
turns a finding a reader could rely on into `unknown`, which says less rather
than more.

It also decides what a missing detail payload is. A payload this machine does
not hold is the ordinary state of a reading recorded elsewhere, so it degrades
its own rows and nothing else, in the same way `#234`, `#252` and `#253` say a
step that cannot do its job fails alone rather than ending the sweep. A payload
that is present and disagrees with its own record still raises, because that is
the store contradicting itself rather than a file that never arrived.

Annotating is only honest if a report can tell the two absences apart, and a
missing payload cannot say what it would have contained. So a metric whose floor
is the paired one declares that in the metric registry, beside the existing
declaration that no sampling floor can exist. Two readings whose contributions
never reached this machine are then distinguishable from two readings that never
had any, which is the distinction the whole annotation rests on.

## Consequences

Every floor names the estimator that produced it, so "which estimator" is a
field rather than a sentence to parse. A delta qualified by a substituted floor
carries the reason in the noise column, in the row's note, and as its own key in
the machine-readable record.

The 1.9x error is still an error. This decision makes it visible; it does not
make it acceptable. `#293` remains the work that lets a pair survive the machine
boundary, and until it lands a cross-machine comparison of a paired metric
reports `unknown` with the reason stated rather than a number.

The declaration and what a benchmark retains can drift, and only one direction
is quiet: retaining a metric the registry does not declare would keep pairing
working and stop the failure being reported anywhere it broke. The two retention
tests assert the agreement, so the drift fails where it happens.

## References

- `docs/evaluation.md`
- `docs/decisions/0019-external-puzzle-calibration-set.md`
- `docs/decisions/0026-conservative-dispersion-bounds.md`
- `docs/decisions/0033-pairing-is-a-correctness-fix-not-a-resolution-lever.md`

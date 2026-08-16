# Vision

Anthro Chess is a project to build a controllable chess bot that plays like a
human opponent in chess games, with optional support for realistic timed play.

The target is not maximum engine strength. The target is a usable model that
plays sensible chess across the range of human strength, showing human-like
timing, imperfection, and configurable skill.

## Product Goal

The final product should be able to:

- play complete legal games against real players;
- operate in both untimed games and games with clocks;
- choose a move and, when timing is enabled, a realistic time-to-move;
- expose dials for target rating, time behavior, randomness, and optional soft
  preferences;
- run and train within practical high-end consumer hardware constraints.

## Experience Goal

The bot should feel like a human opponent, not like a conventional chess engine
with artificial weakness layered on top.

Human-like behavior should mainly emerge from training on human games with
ratings, moves, and clock data when available. The system should learn patterns
of move choice and optional time usage from that data rather than
hand-engineering specific categories of human imperfection.

## Competence Target

The model should span human strength, from casual play up to the strongest play
its training data contains. Reaching that upper end is wanted rather than
incidental.

**Human play is the ceiling by intent, not by concession.** The model learns from
human games and predicts human moves, so the strength it can reach is bounded by
the strength in its data — and that bound is the goal rather than a shortfall. A
stronger model bought by training toward engine evaluations would be a different
product, so producing best-engine moves stays outside what this project is for.

Strength is a setting rather than a fixed property, so the upper end counts only
where the rating control reaches it. A model that plays well but cannot be asked
to play weakly has not met this target either.

## Training Target

Training should be practical on high-end consumer GPUs:

- main iterations should ideally take days, not months;
- a larger final training run may take weeks;
- architecture choices should favor building a usable bot over answering
  research questions.

## Boundaries

Anthro Chess is intended to be:

- a practical chess-playing project;
- a human-like chess opponent for timed or untimed play;
- a standalone opponent, sparring partner, or training sandbox;
- a model controlled through explicit user-facing settings;
- a system where board state is constructed by exact chess logic instead of
  inferred from raw notation;
- a project whose success is judged by whether it is fun, playable, and
  plausibly human.

Anthro Chess is not intended to provide assistance in games against other
people where outside chess help is disallowed, or to misrepresent bot-generated
moves as unaided human play.

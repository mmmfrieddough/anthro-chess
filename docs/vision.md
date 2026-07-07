# Vision

Anthro Chess is a project to build a controllable chess bot that plays like a
human opponent in chess games, with optional support for realistic timed play.

The target is not maximum engine strength. The target is a usable model that can
play sensible chess against casual through somewhat strong humans while showing
human-like timing, imperfection, and configurable skill.

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

The model should be strong enough to play sensibly against casual through
somewhat strong human players. It does not need to challenge expert engine users
or produce best-engine moves.

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

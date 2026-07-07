# Preference Controls

This document describes the design of optional soft preference controls.

## Goal

Preference controls should let a user bias the bot toward broad human-play
concepts without changing the selected rating or forcing incoherent moves.

Examples:

- opening families, such as Sicilian, French, Caro-Kann, Queen's Gambit, or
  King's Indian structures;
- broader style concepts, such as aggression, solidity, development, fianchetto
  preference, simplification, gambit play, or sacrifice tolerance.

These controls should be soft. If the current position does not support a
preference, the model should be able to pivot naturally.

## Non-Goals

- Do not hardcode move-selection rules for specific openings.
- Do not rewrite move logits with hand-authored opening logic.
- Do not make preference sliders a hidden way to change model strength.
- Do not require all preference categories to share one labeling strategy.
- Do not treat a whole game as belonging to an opening after the opening or
  related structure is no longer relevant.

## Data Shape

Preference data should be derived as multi-label metadata over positions or
position windows.

A single ply may have labels such as:

```text
sicilian_family
open_center
opposite_side_castling
kingside_attack
material_imbalance
```

The labels are not the training target for the base model. They are used to
find, train, or evaluate preference-control mechanisms.

Each labeled example should keep enough context for matching and calibration:

- game id;
- ply index;
- side to move;
- bot color or player color for the candidate label;
- board state or normalized position key;
- rating band;
- time-control information when available;
- move number or phase;
- active preference labels;
- source of each label, such as known opening, board structure, or event rule.

## Opening Labels

Opening labels should be position-based rather than copied from a game-level
PGN tag.

The likely source is an opening-position table such as
`lichess-org/chess-openings`, which provides ECO code, name, PGN line, UCI line,
and EPD for known opening positions.

Classification process:

1. Build a lookup from normalized opening position keys to opening metadata.
2. Parse each game and reconstruct the exact board before and after each ply.
3. For each candidate ply, walk backward through earlier positions until a
   known opening position is found.
4. Map the matched opening metadata into Anthro Chess categories.
5. Attach the category only while the opening or related structure is active.

Category mapping should be owned by this project. Raw opening names are useful
input data, but user-facing sliders should be higher-level and cleaner than
raw ECO or variation names.

Example mapping:

```text
Sicilian Defense: Najdorf Variation
  -> sicilian_family
  -> open_sicilian
  -> sharp_asymmetric_opening

French Defense: Advance Variation
  -> french_family
  -> pawn_chain
  -> closed_center

King's Indian Defense
  -> indian_defense
  -> fianchetto_structure
  -> kingside_attack_potential
```

Opening-family labels should use conservative windows. For example, use early
opening and early middlegame positions where the matched opening or its
characteristic structure is still present. Broader structural labels can
continue after the named opening label expires.

## Non-Opening Labels

Non-opening preference labels should be designed case by case.

Structural labels can be derived from the current board:

- fianchetto setup;
- closed center;
- open center;
- isolated queen's pawn;
- opposite-side castling;
- material imbalance;
- advanced kingside pawns;
- developed minor pieces;
- early queen activity.

Event labels can be derived from move history or changes in material:

- gambit accepted or declined;
- early pawn sacrifice;
- piece sacrifice;
- exchange sacrifice;
- early simplification;
- repeated trade offers;
- delayed development.

Some labels may need both structure and history. For example, aggression may
combine attacking pawn advances, piece concentration near the king, willingness
to sacrifice material, and avoidance of simplifying trades.

It is acceptable for these labels to be imperfect. The important requirement is
that each label can produce useful positive and negative sets for discovering or
learning controls.

## Matched Contrast Sets

Preference steering can start by comparing labeled positions against broad
unlabeled controls. This is the simplest baseline and may be enough for useful
sliders.

Matched controls are useful when broad controls produce obvious confounds. They
should be added only where they make the steering signal cleaner, not as a
requirement for every label.

Useful matching dimensions:

- rating band;
- color;
- side to move;
- move number or phase;
- time-control class if timing data is used;
- similar broad opening context when isolating a non-opening concept.

For example, if a `sicilian_family` direction mostly learns "early black
opening position" instead of a Sicilian-specific tendency, a cleaner control set
could compare against other `1. e4` black-defense positions instead of unrelated
late endgames.

## Steering Methods

Candidate methods:

1. Activation-difference steering:
   - collect model activations for positive and matched-control examples;
   - compute a direction such as `mean_positive - mean_control`;
   - apply `slider_value * direction` at selected model layers during
     inference.

2. Supervised steering vectors:
   - train a lightweight classifier or probe for a preference label from hidden
     activations;
   - use the learned direction or representation to steer the model.

3. Sparse-autoencoder features:
   - train sparse autoencoders on hidden activations;
   - identify features correlated with preference labels;
   - use selected features as slider directions.

4. Learned steering controller:
   - train a small module that maps user preference settings and current hidden
     state to an activation delta;
   - keep it constrained and calibrated so it remains a preference mechanism,
     not a separate move engine.

All methods should modify internal representations before the move head rather
than hardcoding move choices after the model runs. Legal move masking still
applies after the model produces move logits.

## Model Integration

The primary integration path is activation-space steering during inference.
The base model can be trained normally, then preference controls can be derived
from hidden activations and applied before the move head.

Preference settings should therefore be represented at runtime as steering
configuration, not as ordinary model input metadata by default.

Direct input conditioning is a fallback option. If activation steering does not
produce stable or useful controls, a later model could be trained with explicit
preference settings as part of its input. That would be a different approach,
not the default design.

Implementation should record:

- which layer or layers are steered;
- steering vector normalization;
- slider-to-strength scaling;
- whether multiple sliders combine additively or through a learned controller;
- whether steering is active for all plies or only relevant phases;
- how steering interacts with KV caching during live inference.

## Application Controls

The application should expose preference settings as optional sliders or similar
continuous controls.

Good UI behavior:

- keep target rating separate from preferences;
- keep temperature separate from preferences;
- group related sliders, such as openings, structures, and style;
- allow zero/neutral values;
- make preferences feel like tendencies, not commands;
- avoid exposing raw ECO codes or overly specific variation names as the main
  interface.

Example runtime configuration:

```text
target_rating: 1500
temperature: 0.85
preferences:
  sicilian_family: 0.7
  french_family: 0.1
  fianchetto_structure: 0.4
  aggression: 0.6
  simplification: -0.2
```

## Evaluation

Preference controls need their own evaluation before they are exposed in the
application. See `docs/evaluation.md` for the main evaluation design.

At minimum, preference-control evaluation should check that the intended
preference increases, unrelated preferences do not drift too much, target rating
stays approximately calibrated, legal masking behavior remains stable, timing
remains plausible when timing is enabled, and generated games still look human
rather than mechanically forced.

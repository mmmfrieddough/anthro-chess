# 0003: Action-Conditioned Timing

Date: 2026-07-11

## Status

Accepted as initial design direction.

## Context

Anthro Chess predicts both what action to play and, when timing is enabled, how
long to wait before submitting it. If action and move time are predicted as
fully independent samples from the same context, generated games can produce
awkward pairs: an unusually strong or surprising move played instantly, or an
obvious simple move after an implausibly long delay.

The project considered independent output heads and also the opposite ordering,
where time is sampled before the action. Both are possible, but they make the
relationship between action choice and timing less direct.

## Decision

Factor the joint output as action first, then time conditioned on that action:

```text
p(action_t, move_time_t | context_t)
  = p(action_t | context_t) * p(move_time_t | context_t, action_t)
```

The shared causal model produces a context state. The action head predicts the
next action policy. When timing is enabled, the time head predicts a sampleable
move-time distribution from the same context state plus an embedding of the
selected action.

During training, the time target is conditioned on the observed human action:

```text
action_loss = -log p(human_action | context)
time_loss   = -log p(human_time | context, human_action)
```

During inference, the runtime samples a legal action first, then samples move
time conditioned on the sampled action.

## Consequences

This keeps timing attached to the move actually being submitted. It also keeps
time pressure available to the action head through context inputs such as clock
state, increment, and prior timing.

There remains a train/inference distinction: the time head trains on observed
human actions and runs on model-sampled actions. Evaluation should therefore
include move-time coherence checks in generated games instead of only
independent action and timing losses.

Timing remains optional. Untimed games should not require time targets or
move-time inference.

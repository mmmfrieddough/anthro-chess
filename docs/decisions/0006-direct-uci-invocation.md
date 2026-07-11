# 0006: Direct UCI Invocation

Date: 2026-07-11

## Status

Accepted as initial design direction.

## Context

Anthro Chess should work with local chess GUIs and other tools that understand
the Universal Chess Interface. UCI is a text protocol over a child process's
standard input and standard output. In typical use, the GUI launches the engine
process directly, sends commands, receives protocol responses, and manages
engine options through `setoption`.

An alternative would be a small UCI adapter process that talks to a separately
running Anthro service over another protocol. That could keep a heavy model
loaded outside the GUI-owned process, but it would add another protocol,
another process lifecycle, and more synchronization complexity.

## Decision

Use direct UCI invocation as the default compatibility approach.

The UCI executable should load the runtime and model itself, advertise supported
options, accept `setoption` updates, keep standard output reserved for UCI
protocol messages, and send normal logs to files or standard error.

UCI remains an interface layer over the runtime. It is not the model's native
representation, and the model should not learn raw UCI commands.

## Consequences

Local GUI compatibility is straightforward: users point the GUI at the engine
executable, and the executable owns runtime/model startup.

Startup cost may matter if model loading becomes heavy. That should be handled
inside the direct executable path first, rather than introducing an adapter
protocol prematurely.

UCI options are GUI-to-engine configuration for the current process. They should
not be treated as a bidirectional synchronized settings system for a separate
web UI.

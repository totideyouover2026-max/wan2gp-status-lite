# Changelog

## 1.0.1 — 2026-09-01

- Made simultaneous Status Pro and Status Lite installation safe: Pro takes precedence and Lite remains dormant without installing duplicate callback or download observers.
- Resolved Wan2GP's native `gen_status` component directly, with an insertion-order-safe sibling fallback that skips either Status plugin container.

## 1.0.0 — 2026-09-01

- Created Status Lite from the Status Pro 1.0.5 live stage engine.
- Preserved stage selection, elapsed time, ETA, step speed, memory telemetry, model downloads, component details, model lifecycle, and responsive collapse behavior.
- Preserved accurate standalone and inline LTX 2/2.5 post-processing model/component reporting.
- Preserved task-bound step observation and minimized-window recovery.
- Kept sliding and post-processing subwindows within one ephemeral live task.
- Removed generation history, browser/session run persistence, history settings, prompt retention, import/export, and old-run gallery navigation.
- Gave Lite an independent plugin name, runtime namespace, DOM IDs, manifest, documentation, and tests.

# Changelog

## 1.1.2 — 2026-09-22

- Bound authoritative stage timing to task identity and execution epoch before generation callbacks can be recorded.
- Prevented stale Save timing and native DOM status from carrying into later queued or separate generations.
- Preserved measurable Encode activity for fast-starting queued tasks and isolated legacy fallback from authoritative V13 telemetry.
- Prevented the completed Save view from flashing against the idle completion view after generation finishes.

## 1.1.1 — 2026-09-14

- Fixed WanGP V13 model, module, and LoRA downloads that could fail because the download observer accepted only the older two-argument call.
- The observer now forwards all native download arguments unchanged, preserving progress generators, filename display settings, future options, and compatibility with older WanGP releases.
- Fixed YuE2 score and semantic-audio token progress creating large numbers of duplicate live activities.
- Token, tile, and layer progress is now phase-local telemetry rather than denoising step performance, and YuE2 audio decoding appears under Decode.

## 1.1.0 — 2026-09-13

- Added WanGP V13 native phase progress, nearby WangpProgress support, and unit-aware step, layer, and tile counters within the existing seven-stage timeline. Legacy WanGP progress remains supported.
- Expanded live activity detail across Inputs, Encode, Generate, Decode, and Enhance, while preserving repeated prompts, passes, and windows.
- Kept text-encoder and VAE tile callbacks separate from denoising step timing, cache-skip observations, and phase numbering.
- Added genuine V13 VAE Decode progress while preserving cancellation, minimized-window recovery, Qwen Encode fallback, and Status Pro precedence.

## 1.0.0 — 2026-09-05

- A stopped queue no longer revives Prepare from lingering abort text or completed asset activity. Active downloads and model unloading remain visible.
- Temporary or incomplete telemetry snapshots no longer reset an active generation.
- Classified gallery-applied LTX 2/2.5 `Distilled refinement` as Enhance rather than Prepare.
- Preserved the earlier upsampling-start duration when LTX returns to Enhance after its Inputs and Encode phases.
- Made simultaneous Status Pro and Status Lite installation safe: Pro takes precedence and Lite remains dormant without installing duplicate callback or download observers.
- Resolved Wan2GP's native `gen_status` component directly, with an insertion-order-safe sibling fallback that skips either Status plugin container.
- Created Status Lite from the Status Pro 1.0.5 live stage engine.
- Preserved stage selection, elapsed time, ETA, step speed, model downloads, component details, model lifecycle, and responsive collapse behavior.
- Preserved accurate standalone and inline LTX 2/2.5 post-processing model/component reporting.
- Preserved task-bound step observation and minimized-window recovery.
- Kept sliding and post-processing subwindows within one ephemeral live task.
- Removed generation history, browser/session run persistence, history settings, prompt retention, import/export, and old-run gallery navigation.
- Gave Lite an independent plugin name, runtime namespace, DOM IDs, manifest, documentation, and tests.

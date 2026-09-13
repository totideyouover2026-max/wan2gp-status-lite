# Status Lite 1.1.0

Status Lite 1.1.0 adds WanGP V13 compatibility to the lightweight, live-only Status panel.

## What's new

- Native WanGP phase state now takes priority, with V13 WangpProgress and legacy progress markup retained as fallbacks.
- Unit-aware counters can show text-encoder layers, VAE tiles, and denoising steps.
- Richer activities appear inside Inputs, Encode, Generate, Decode, and Enhance while the same seven-stage timeline remains unchanged.
- Repeated prompts, passes, and windows remain individually visible.
- Genuine V13 VAE Decode progress is displayed when available. Older blocking Decode operations continue to show elapsed activity without an invented percentage or ETA.
- Layer and tile callbacks remain separate from denoising step timing, cache-skip observations, and phase numbering.
- Cancellation, minimized-window recovery, Qwen Encode fallback, and safe coexistence with Status Pro remain supported.

Older WanGP releases continue to work through the legacy fallback, though they may naturally provide fewer phase details.

## Live-only operation

Status Lite does not save completed runs and has no generation History, prompt retention, run import/export, or previous-run gallery navigation. Temporary task data is discarded when WanGP reports completion. Only the panel's collapsed or expanded preference is stored locally.

Status Pro takes priority when both editions are enabled, leaving Lite dormant without duplicate observers or a competing panel.

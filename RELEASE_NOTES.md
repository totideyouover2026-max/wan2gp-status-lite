# Status Lite 1.1.2

Status Lite 1.1.2 stabilises task-owned live stage timing across queued and separate generations. Task identity and execution epoch are bound before progress callbacks, so fast Encode activity is retained and timing from a previous task cannot create or extend stages in the next task.

Authoritative WanGP V13 state now takes precedence over stale native DOM status. Save is shown while saving, then yields cleanly to the completed presentation without flashing between the two views.

## Status Lite 1.1.1

Status Lite 1.1.1 is a focused download compatibility hotfix. WanGP V13 model, module, and LoRA downloads now pass every native argument through the Status observer unchanged, including progress generators and filename display settings. Older WanGP download calls remain supported.

YuE2 score and semantic-audio token updates now remain single, advancing live activities. Token, tile, and layer counters remain phase-local rather than entering denoising performance, while genuine acoustic-synthesis steps remain recorded and YuE2 audio decoding appears under Decode.

## Status Lite 1.1.0

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

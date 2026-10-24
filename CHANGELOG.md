# Changelog

## Unreleased

- Prevented a previous window or sample's lingering Save status from trapping later Viggle and other sliding-window work in Save when generation uses the polling fallback.

- Recovered live progress when a host bypasses the installed generation wrapper, while rejecting unchanged progress left over from the previous queued job.

- Prevented delayed status polls from resetting a later job to Prepare after an abort and queue handover; worker events now own task, sample and window timing when the generation observer is installed.
- Kept worker ownership authoritative between queued jobs so a previous image's Save status cannot seed the next video's timing, including H3 runs with the model kept loaded.
- Published task ownership from the generation observer for hosts without the native active-task field, while retaining the legacy fallback when the observer is unavailable.

## 1.3.0 — 2026-10-17

- Kept streaming voice/audio progress in one semantic activity as durations and counters change, including standalone-to-denoising callback handoffs, while preserving distinct parts, segments and passes.
- Preserved the last reported counter when an activity ends instead of forcing it to its maximum; recognised native speech waveform decoding as Decode.

- Unified Enhance tracking across registered spatial/temporal upscalers, audio processors, film grain and VAE-based upscaling; preserved arbitrary processor status text, loading and internal encode/decode work in their owning enhancement scope.
- Fixed processor ownership across later jobs and reused task IDs, retained progress units with older callbacks, and resumed generation after early VAE-upscaler preparation.
- Added a processor-family regression matrix and a CPU-only audit against upstream WanGP dispatcher implementations (checked against de302eec64b6).

- Tracked PiD loading and VAE conditioning in Enhance through the shared spatial/temporal dispatchers, recognised native PiD statuses, and recovered processor ownership from WanGP's active media-control scope when the generation entry wrapper is bypassed.

- Reported H3's transformer handoff before phase-two latent upscaling as Enhance, exposed its layer counts, and recognised native low/high-resolution, tiled and audio-refinement generation statuses.

- Kept worker stage timing, start/completion timestamps and recent completed executions available while a window is minimized; restored missed runs/windows without charging hidden work to Prepare or adding the refocus delay.
- Requested a fresh backend status snapshot on focus and visibility restoration, and applied completed server timings before finalizing live or History records.
- Captured task-owned status labels and counters from the same native event, including prompt enhancement and input preprocessing.
- Prevented denoising counters or invalid tile totals from producing false Decode percentages; repaired Qwen 2.1 tile totals from the actual latent decode dimensions and retained completed counts when a total is unknown.
- Tracked soundtrack, voice processing and film grain in Enhance, including audio processing after an intermediate Save, and accumulated repeated H3 Encode passes.
- Fixed spatial and temporal postprocessing remaining in Prepare or Decode. All processors now enter Enhance at their actual call boundary, including processors that emit no progress.
- Kept processor-internal loading, encoding, denoising and decoding in Enhance, with task-owned frame/tile counters and cleanup after failures.
- Recognised RIFE, DLSS and additional enhancement labels in both stage matchers, and preserved unit-aware LTX callbacks.

## 1.2.2 — 2026-10-06

- Kept MiniMax H3 Ref2VA RefMod preview decoding in Encode until generation begins; output decoding remains in Decode.
- Restarted live tracking for each sample added with WanGP's One More button, even when the queue task stays the same.

## 1.2.1 — 2026-09-27

- Restarted stage timing at each sliding-window boundary so Save from one window cannot hide progress in the next.

## 1.2.0 — 2026-09-27

- Added total elapsed time to the header and a display-only final denoising step countdown based on the displayed average step time.
- Improved IndexTTS stage reporting and the model unloading/loading handoff.
- Display denoising as `Step n/N` from the first active step, while keeping layer and tile counters with their own phases.
- Put expanded metrics on one row on wide panels, removed duplicate elapsed-time metrics, and prevented stale Encode layers from appearing during Denoising.

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

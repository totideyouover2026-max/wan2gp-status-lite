# Status Lite 1.0.0

Status Lite delivers Status Pro's live stage timeline for users who do not need generation history.

## Included in the initial release

- Selectable stages, live timings and ETA, model/component identities, LTX post-processing detail, multi-pass/subwindow step observations, download progress, and recovery after the window has been minimized.
- Gallery-applied LTX 2/2.5 `Distilled refinement` stays in Enhance and preserves the earlier upsampling-start time across Inputs and Encode.
- A stopped queue no longer revives Prepare from lingering abort text or completed asset activity. Active downloads and model unloading remain visible.
- Temporary or incomplete telemetry snapshots preserve the active generation until valid completion telemetry arrives.
- Safe coexistence with Status Pro: Pro takes precedence when both are enabled, and Lite remains dormant without installing duplicate observers or a competing panel.
- WanGP's native status component is resolved independently of plugin insertion order.
- An independent plugin identity, responsive layout, and collapsed/expanded preference.

## Live-only operation

Completed-run data is not retained. There is no history UI, prompt memory, run import/export, or gallery navigation from prior runs. The ephemeral task object is discarded as soon as Wan2GP reports the task complete. Only the panel's collapsed/expanded preference is stored locally.

Status Lite and Status Pro are alternative presentations, so most users only need one. Keeping both installed is supported; disable Pro and restart Wan2GP whenever you want Lite to become the active edition.

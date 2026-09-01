# Wan2GP Status Lite

Status Lite is the live, stage-based Wan2GP status panel from Status Pro, packaged for users who do not need generation history.

It keeps the live experience:

- Selectable Prepare, Inputs, Encode, Generate, Decode, Enhance, and Save stages
- Live elapsed time, progress, step count, rolling step speed, and ETA
- Per-stage model/component details, including accurate LTX 2/2.5 post-processing identities
- Multi-pass and sliding/subwindow awareness
- Recovered stage and step telemetry after a window is minimized or backgrounded
- Process RAM, VRAM, and GPU-memory observations
- Model download file progress, transfer cycles, speed, and ETA
- Model loading/unloading, abort, failure, and optional-stage handling
- Responsive and collapsible layout

It intentionally does **not** include:

- Generation or task history
- Browser/session run records
- History import or export
- Gallery navigation from old runs
- Prompt-retention or history-retention controls

The only browser preference Status Lite stores is whether its panel is collapsed. Run telemetry exists only while the current Wan2GP task is active and is discarded when that task finishes.

## Install

Copy this folder into Wan2GP's `plugins` directory, then restart Wan2GP and enable **Status Lite** in the plugin manager.

Status Lite and Status Pro are alternatives. Install or enable one of them, not both, because both observe the same Wan2GP generation callbacks.

## Live stage behavior

Status Lite reads Wan2GP's native progress UI and a small plugin-owned telemetry bridge. The bridge supplies queue settings, model components, process/GPU memory, download activity, and callback step samples that the native progress text does not expose.

Sliding windows and post-processing subwindows remain part of the same live task. Their observed steps are accumulated while the task runs; Status Lite does not split them into saved records.

If the browser or app window is minimized, Gradio may pause visible DOM updates. When the window returns, Status Lite reconciles the backend step samples and rebuilds any stages it missed while backgrounded.

## Privacy and storage

Status Lite does not save prompts, settings, paths, outputs, performance records, or completed runs. It uses one local browser key for the collapsed-panel preference:

`wangp.status-lite.collapsed.v1`

## Requirements

- Wan2GP with plugin support
- A modern browser or the Wan2GP app webview
- No additional Python dependencies beyond Wan2GP's environment

## License

See [LICENSE](LICENSE).

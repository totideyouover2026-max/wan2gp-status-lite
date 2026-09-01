# Status Lite user guide

## What the panel shows

The header reports the current Wan2GP activity, total elapsed time, observed generation steps, and ETA when enough information is available. Select any visible stage card to inspect its status and timing.

Stages appear only when relevant. Inputs, Enhance, and Save are optional; a model or workflow that does not report one of them will not be forced to display it as active work.

## Model and component details

Prepare shows the transformer or primary processing model. Inputs and Decode show the applicable VAE components. Encode shows the text encoder. Status Lite uses the task's actual processing metadata, so a gallery-applied LTX upscaler is shown as LTX rather than inheriting the model selected on the generation form.

## Steps, passes, and subwindows

The live counter accumulates callback observations for the active queue task. Multi-pass and LTX spatial-upscaler windows retain their phase/window labels. A two-window, eight-step upscaler can therefore show sixteen observations while still describing the configured work as two passes of eight steps.

## Minimized or background windows

Browsers can throttle Gradio timers and DOM mutation delivery while minimized. Wan2GP's backend observations continue. When the page becomes visible or focused again, Status Lite immediately refreshes and reconstructs missed performance stages from the backend samples.

## Downloads

During model downloads, the panel shows known file sizes, received bytes, transfer-cycle observations, effective transfer rate, freshness, and ETA when enough transfer data exists. Download telemetry is limited to downloads made by Wan2GP in the current process.

## Collapse preference

Use the arrow in the top-right corner to collapse or expand the panel. This is the only Status Lite value saved in browser storage.

## No history

After a task completes, its temporary settings and performance samples are discarded. Status Lite has no history drawer, run database, prompt memory, imports, exports, or old-output gallery actions. Use Status Pro instead if you need those features.

## Troubleshooting

- If the panel does not appear, confirm Status Lite is enabled and restart Wan2GP.
- Do not enable Status Lite and Status Pro together.
- If download detail is unavailable after an upstream Wan2GP change, generation-stage tracking should still operate.
- If a stage says Wan2GP did not report intermediate progress, the underlying model or workflow did not expose a finer-grained callback for that stage.

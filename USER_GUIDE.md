# Status Lite user guide

## What the panel shows

The header reports the current Wan2GP activity, total elapsed time, observed generation steps, and ETA when enough information is available. Select any visible stage card to inspect its status and timing.

The selected stage card and its activity details show stage timing. **Total elapsed** appears in the header, including when the panel is collapsed.

Stages appear only when relevant. Inputs, Enhance, and Save are optional; a model or workflow that does not report one of them will not be forced to display it as active work.

WanGP V13 can provide activity counters such as `Encoding Text Prompt · 50/50 layers`, `Denoising · Step 8/8`, and `VAE Decoding · 28/28 tiles`. These activities remain inside the existing seven top-level stages. Inputs, Encode, Generate, Decode, and Enhance can each show their detailed sub-activities, including repeated prompts, passes, or windows.

Status Lite prefers WanGP's native phase information, with V13 WangpProgress and older progress markup retained as fallbacks. Layer and tile counters do not affect denoising step speed, cache-skip observations, or phase numbers. Genuine V13 Decode progress is shown when available; older workflows without it continue to show elapsed Decode activity.

## Model and component details

Prepare shows the transformer or primary processing model. Inputs and Decode show the applicable VAE components. Encode shows the text encoder. Status Lite uses the task's actual processing metadata, so a gallery-applied LTX upscaler is shown as LTX rather than inheriting the model selected on the generation form.

For gallery-applied LTX upscaling, setup, conditioning and `Distilled refinement` all remain in Enhance. Their elapsed times accumulate in the same stage.

## Steps, passes, and subwindows

The live counter accumulates callback observations for the active queue task. Multi-pass and LTX spatial-upscaler windows retain their phase/window labels. A two-window, eight-step upscaler can therefore show sixteen observations while still describing the configured work as two passes of eight steps.

During denoising, the displayed step number describes the step currently running: an eight-step run starts at `Step 1/8` and reaches `Step 8/8` while the final step runs. The displayed total follows the configured step count. Layer, token, tile, and segment counters retain their reported values.

On the final denoising step, the ETA counts down from recent completed step times when enough samples are available. If the estimate expires before WanGP reports the next stage, it says **Nearly done**.

IndexTTS speech segment progress appears in Generate. Its vocoder conditioning and waveform generation appear in Decode. When switching from IndexTTS to another model, Prepare shows the outgoing unload followed by the incoming load.

Activity rows represent distinct processing phases. Changing generated-audio duration, elapsed time or progress counts update the current row; real parts, segments and passes remain separate. A completed activity retains its last reported counter, so speech ending before its maximum duration does not imply that the maximum frame or token count was reached.

Registered enhancement operations keep model loading and their internal preprocessing, prompt encoding, sampling and decoding in **Enhance**, both during generation and when applied to an existing output. This includes spatial/temporal upscalers, refinement, soundtrack generation, voice processing, audio cleanup and film grain. Processor names come from WanGP's registry. Missing progress counters show activity and elapsed time without an invented percentage. VAE-based upscalers also use Enhance; early upscaler preparation returns to the generation pipeline when it finishes.

H3 two-phase generation reports the intermediate model handoff and latent upscaling in **Enhance**, followed by **Encode** for phase-two prompt/reference conditioning and **Generate** for the second pass. Latent upscaler progress uses **layers**, separate from denoising steps. Final media decoding uses **Decode**.

## Minimized or background windows

Browsers and Pinokio can throttle page timers while minimized. Status Lite records stage timings and execution start/completion times in WanGP's generation worker independently of page polling. When visible or focused again, it requests a fresh backend snapshot and restores the live/completed stage timings, including a recent task that finished entirely while hidden. Refocusing does not extend the completed-run duration. The backend retains the most recent 16 task executions in memory; Status Lite does not create a browser History ledger. Rendering may pause while hidden, but worker timing collection continues.

## Downloads

During model downloads, the panel shows known file sizes, received bytes, transfer-cycle observations, effective transfer rate, freshness, and ETA when enough transfer data exists. Download telemetry is limited to downloads made by Wan2GP in the current process.

## Collapse preference

Use the arrow in the top-right corner to collapse or expand the panel. This is the only Status Lite value saved in browser storage.

## No history

After a task completes, its temporary settings and performance samples are discarded. Status Lite has no history drawer, run database, prompt memory, imports, exports, or old-output gallery actions. Use Status Pro instead if you need those features.

## Troubleshooting

- If the panel does not appear, confirm Status Lite is enabled and restart Wan2GP.
- If Status Lite and Status Pro are both enabled, Status Pro takes precedence and Lite intentionally does not display or install its observers. Disable Pro and restart Wan2GP to use Lite instead.
- If download detail is unavailable after an upstream Wan2GP change, generation-stage tracking should still operate.
- If a stage says Wan2GP did not report intermediate progress, the underlying model or workflow did not expose a finer-grained callback for that stage.
- Older WanGP releases remain supported, but may provide fewer phase details and counters than V13.

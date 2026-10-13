"""Native status event and decoder-counter regressions, shared by both editions."""
import copy
import json
import shutil
import subprocess
import threading
import types
import unittest
from unittest.mock import patch

from test_enhancement_stages import helpers
from test_release_smoke import _javascript_with_exports


class StatusEventTests(unittest.TestCase):
    def test_decode_does_not_reuse_finished_generation_steps(self):
        scope = helpers()
        for unit in (None, "steps"):
            gen = {"progress_phase": ("VAE Decoding", 8), "num_inference_steps": 8,
                   "last_progress_args": [(8, 8), "VAE Decoding", 8] + ([unit] if unit else []),
                   "phase_progress_units": (8, unit)}
            snapshot = scope["_native_progress_snapshot"](gen)
            self.assertIsNone(snapshot["current"])
            self.assertIsNone(snapshot["total"])
            self.assertIsNone(snapshot["progress"])
        valid = scope["_validated_phase_progress"]("VAE Decoding", 4, 12, "tiles")
        self.assertEqual((valid["current"], valid["total"], valid["progress"]), (4, 12, 4 / 12 * 100))

    def test_invalid_counter_preserves_completed_tiles_without_percentage(self):
        scope = helpers()
        for current, total in ((15, 1), (13, 12), (4, 0), (4, None)):
            value = scope["_validated_phase_progress"]("VAE Decoding", current, total, "tiles")
            self.assertEqual(value["current"], current)
            self.assertIsNone(value["total"])
            self.assertIsNone(value["progress"])
        for current, total in ((float("nan"), 10), (None, 10), (-1, 10)):
            value = scope["_validated_phase_progress"]("VAE Decoding", current, total, "tiles")
            self.assertIsNone(value["current"])
            json.dumps(value, allow_nan=False)

    def test_generation_events_are_atomic_and_ignore_stale_global_fields(self):
        scope = helpers()
        timer = scope["_StageTimingTelemetry"]()
        snapshots = []
        state = {"gen": {"progress_phase": ("Encoding Text Prompt", 32),
                         "last_progress_args": [(32, 32), "Encoding Text Prompt", 32, "layers"]}}
        original_state = copy.deepcopy(state)
        def body(task, send_cmd, state=None):
            for label, current, total, unit in (("Enhancing Prompt", 7, 128, "tokens"),
                ("Extracting Face Movements", None, None, None), ("Encoding Text Prompt", 2, 32, "layers"),
                ("Denoising", 2, 8, "steps"), ("VAE Decoding", 3, 30, "tiles")):
                args = [(current, total), label, total, unit] if current is not None else [0, label]
                send_cmd("progress", args)
                snapshots.append(timer.snapshot())
            return True
        owner = types.SimpleNamespace(_stage_timing=timer, _generation_timing_observer_installed=False, generate_media=body)
        owner.set_global = lambda name, value: setattr(owner, name, value)
        scope["_install_generation_timing_observer"](owner)
        owner.generate_media({"id": "task", "params": {}}, lambda *args: None, state=state)
        self.assertEqual([s["last_stage"] for s in snapshots], ["encode", "input", "encode", "denoise", "decode"])
        self.assertEqual(snapshots[-1]["progress"]["total"], 30)
        self.assertIsNone(snapshots[1]["progress"]["current"])
        self.assertEqual(state, original_state)  # Observation does not rewrite WanGP's shared fields.
        epoch = snapshots[-1]["execution_epoch"]
        timer.start_task("next")
        self.assertFalse(timer.observe_progress("task", epoch, "Saving output", stage_id="save"))
        self.assertIsNone(timer.snapshot()["progress"])

    def test_repeated_encode_accumulates_and_stale_epoch_is_rejected(self):
        scope = helpers()
        timer = scope["_StageTimingTelemetry"]()
        epoch = timer.start_task("h3", now=0)
        for stage, now in (("encode", 1), ("denoise", 4), ("post", 9), ("encode", 12), ("denoise", 14)):
            self.assertTrue(timer.observe_stage("h3", stage, now=now, execution_epoch=epoch))
        self.assertEqual(timer.snapshot(now=15)["stages"]["encode"]["elapsed"], 5)
        self.assertEqual(timer.snapshot(now=15)["stages"]["encode"]["run_count"], 2)
        self.assertFalse(timer.observe_progress("h3", epoch + 1, "Encoding Text Prompt", 1, 32, "layers", "encode"))
        self.assertEqual(timer.snapshot()["last_stage"], "denoise")

    def test_qwen_tile_total_matches_actual_latent_loop_not_declared_total(self):
        scope = helpers()
        vae = types.SimpleNamespace(use_tiling=True, spatial_compression_ratio=16,
            tile_sample_min_height=512, tile_sample_min_width=512,
            tile_sample_stride_height=384, tile_sample_stride_width=384)
        latents = types.SimpleNamespace(shape=(1, 16, 1, 64, 112))
        self.assertEqual(scope["_qwen21_decoder_tile_total"](vae, [latents]), 15)
        latents.shape = (2, 16, 1, 64, 112)
        self.assertEqual(scope["_qwen21_decoder_tile_total"](vae, latents), 30)
        vae.use_tiling = False
        self.assertEqual(scope["_qwen21_decoder_tile_total"](vae, latents), 2)
        latents.shape = (1, 16, 5, 64, 112)
        self.assertIsNone(scope["_qwen21_decoder_tile_total"](vae, latents))

    def test_qwen_adapter_repairs_owned_totals_and_cleans_up_on_failure(self):
        scope = helpers()
        context = threading.local()
        calls = []
        def decode(latents, **kwargs):
            self.assertEqual(context.decoder_tile_total, 15)
            calls.append(latents)
            raise RuntimeError("decoder failed")
        vae = type("AutoencoderKLQwenImage21", (), {})()
        vae.use_tiling = True; vae.spatial_compression_ratio = 16
        vae.tile_sample_min_height = vae.tile_sample_min_width = 512
        vae.tile_sample_stride_height = vae.tile_sample_stride_width = 384
        vae.decode_to_cpu_uint8 = decode
        scope["_install_qwen21_tile_observer"](types.SimpleNamespace(vae=vae), context)
        installed = vae.decode_to_cpu_uint8
        scope["_install_qwen21_tile_observer"](types.SimpleNamespace(vae=vae), context)
        self.assertIs(installed, vae.decode_to_cpu_uint8)
        latents = [types.SimpleNamespace(shape=(1, 16, 1, 64, 112))]
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            vae.decode_to_cpu_uint8(latents)
        self.assertEqual(calls, [latents])
        self.assertIsNone(context.decoder_tile_total)
        timer = scope["_StageTimingTelemetry"]()
        snapshots = []
        def body(task, send_cmd, state=None):
            owner._generation_context.decoder_tile_total = 15
            for current in range(1, 16):
                send_cmd("progress", [(current, 1), "VAE Decoding", 1, "tiles"])
                snapshots.append(timer.snapshot()["progress"])
            return True
        owner = types.SimpleNamespace(_stage_timing=timer, _generation_timing_observer_installed=False, generate_media=body)
        owner.set_global = lambda name, value: setattr(owner, name, value)
        scope["_install_generation_timing_observer"](owner)
        owner.generate_media({"id": "task", "params": {}}, lambda *args: None)
        self.assertEqual([p["current"] for p in snapshots], list(range(1, 16)))
        self.assertEqual({p["total"] for p in snapshots}, {15})
        self.assertEqual(snapshots[-1]["progress"], 100)

    def test_audio_and_film_grain_modules_track_real_processor_boundaries(self):
        scope = helpers()
        timer = scope["_StageTimingTelemetry"]()
        observations = []
        def audio(method, **kwargs):
            self.assertEqual(timer.snapshot()["last_stage"], "post")
            self.assertEqual(kwargs["progress_callback"]("Audio - Denoising", 2, 8), "progress forwarded")
            kwargs["status_callback"]("Replace Voice")
            observations.append(timer.snapshot()["progress"])
            return "audio output"
        def grain(images, grain_intensity=0, saturation=0.5):
            observations.append(timer.snapshot())
            return images
        audio_module = types.SimpleNamespace(generate_soundtrack=audio, replace_voice_tracks=audio, process_audio_file=audio)
        grain_module = types.SimpleNamespace(add_film_grain=grain)
        owner = types.SimpleNamespace(_stage_timing=timer, _generation_timing_observer_installed=False,
                                      _postprocessing_step_observer_installed=False)
        owner.set_global = lambda name, value: setattr(owner, name, value)
        def import_module(name):
            return audio_module if name.endswith("audio_processors") else grain_module
        with patch.object(scope["importlib"], "import_module", side_effect=import_module):
            scope["_install_postprocessing_step_observer"](owner)
        first = audio_module.generate_soundtrack
        scope["_install_postprocessing_step_observer"](owner)
        self.assertIs(first, audio_module.generate_soundtrack)
        def body(task, send_cmd, state=None):
            send_cmd("status", "Saving File intermediate.mp4")
            self.assertEqual(audio_module.generate_soundtrack("mmaudio", progress_callback=lambda *a: "progress forwarded",
                             status_callback=lambda *a: "status forwarded"), "audio output")
            self.assertEqual(timer.snapshot()["last_stage"], "post")
            send_cmd("status", "Saving File final.mp4")
            self.assertEqual(timer.snapshot()["last_stage"], "save")
            self.assertEqual(grain_module.add_film_grain("pixels", 0), "pixels")
            self.assertEqual(timer.snapshot()["last_stage"], "save")
            self.assertEqual(grain_module.add_film_grain("pixels", 0.2), "pixels")
            self.assertEqual(timer.snapshot()["last_stage"], "post")
            self.assertIsNone(timer.snapshot()["enhancement"])
            return True
        owner.generate_media = body
        scope["_install_generation_timing_observer"](owner)
        owner.generate_media({"id": "task", "params": {}}, lambda *args: None)
        self.assertEqual(observations[0]["stage"], "post")
        self.assertEqual(observations[0]["phase"], "Replace Voice")
        self.assertIsNone(observations[1]["enhancement"])
        self.assertEqual(observations[2]["enhancement"]["phase"], "Film Grain")

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_browser_owned_events_counters_reentry_and_fallback(self):
        js = _javascript_with_exports("freshState", "readLiveSnapshot", "readReportedPhaseStatus",
                                     "applySnapshot", "applyServerStageTiming", "formatCounter", "structuredStageId")
        script = r"""
const api = globalThis.__statusProReleaseTest;
function check(value, message) {if (!value) throw new Error(message);}
const ns = {state:api.freshState(), activeRun:{queue_task_id:'task', _stageTimingEpoch:1, settings:{}},
    runTelemetry:{execution_task_known:true, executing_task:{id:'task'},status:'',
        native_progress:{phase:'Encoding Text Prompt',current:32,total:32,unit:'layers'},
        stage_timing:{task_id:'task',execution_epoch:1,stages:{input:{active:true}},
            progress:{stage:'input',phase:'Extracting Face Movements',current:null,total:null,unit:null}}}};
let result = api.readLiveSnapshot(ns);
check(result.id === 'input' && result.rawName === 'Extracting Face Movements' && result.progress === null, 'stale native Encode overrode owned Inputs');
ns.runTelemetry.stage_timing.stages = {decode:{active:true}};
for (let current=0; current<=15; current++) {
    ns.runTelemetry.stage_timing.progress = {stage:'decode',phase:'VAE Decoding',current,total:15,unit:'tiles'};
    result = api.readLiveSnapshot(ns);
    check(api.formatCounter(result.steps) === `${current}/15 tiles`, 'counter reversed or total changed');
    check(result.progress === current / 15 * 100, 'incorrect tile percentage');
}
ns.runTelemetry.stage_timing.progress = {stage:'decode',phase:'VAE Decoding',current:15,total:null,unit:'tiles'};
result = api.readLiveSnapshot(ns);
check(result.progress === null && api.formatCounter(result.steps) === '15 tiles completed', 'invalid total fabricated progress');
ns.state.records.denoise.stepTotal=8;
api.applySnapshot(ns,result);
check(ns.state.steps.current === 15 && ns.state.steps.unit === 'tiles', 'unknown tile total replaced by denoising header counter');
ns.runTelemetry.stage_timing.progress = {stage:'decode',phase:'VAE Decoding',current:null,total:null,unit:null};
check(api.readLiveSnapshot(ns).progressScope === null, 'automatic denoising completion counted as decoder progress');
ns.runTelemetry.stage_timing.execution_epoch=2;
check(api.readReportedPhaseStatus(ns).id === 'encode', 'wrong epoch owned progress trusted');
ns.runTelemetry.stage_timing.execution_epoch=1;
ns.runTelemetry.stage_timing.stages={save:{active:true}};
ns.runTelemetry.stage_timing.progress={stage:'save',phase:'Saving File intermediate.mp4'};
api.applyServerStageTiming(ns,ns.runTelemetry);
api.applySnapshot(ns,api.readLiveSnapshot(ns));
ns.runTelemetry.stage_timing.stages={post:{active:true},save:{active:false,completed:true,elapsed:2}};
ns.runTelemetry.stage_timing.progress={stage:'post',phase:'Audio - Denoising',current:2,total:8,unit:'steps'};
api.applyServerStageTiming(ns,ns.runTelemetry);
api.applySnapshot(ns,api.readLiveSnapshot(ns));
check(ns.state.currentId === 'post', 'Save blocked audio Enhance');
ns.runTelemetry.stage_timing.stages={encode:{active:true,elapsed:5}};
ns.runTelemetry.stage_timing.progress={stage:'encode',phase:'Encoding Text Prompt',current:7,total:32,unit:'layers'};
api.applyServerStageTiming(ns,ns.runTelemetry);
api.applySnapshot(ns,api.readLiveSnapshot(ns));
check(ns.state.currentId === 'encode' && ns.state.records.encode.elapsed >= 5, 'second Encode rejected or duration reset');
for (const phase of ['Enhancing Prompt','Captioning Images','Encoding H3 phase 2 prompt and references']) check(api.structuredStageId(phase)==='encode',phase);
check(api.structuredStageId('Audio - VAE Decoding')==='post','soundtrack decoding escaped Enhance');
check(api.structuredStageId('Audio VAE Decoding')==='decode','model audio decode misclassified');
"""
        result = subprocess.run([shutil.which("node"), "-"], input=js+script, text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

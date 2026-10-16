"""Processor boundary regressions shared by Status Pro and Lite."""
import ast
import inspect
import importlib
import json
import shutil
import subprocess
import threading
import time
import types
import unittest
from functools import wraps

from test_release_smoke import _source, _javascript_with_exports


def helpers():
    tree = ast.parse(_source())
    names = {"_StageTimingTelemetry", "_structured_stage_id", "_install_generation_timing_observer",
             "_install_postprocessing_step_observer", "_processor_execution", "_forward_processor_progress", "_new_performance_observer", "_observe_postprocessing_progress", "_validated_phase_progress", "_qwen21_decoder_tile_total",
             "_install_qwen21_tile_observer", "_install_h3_handoff_observer", "_install_step_observer", "_native_progress_snapshot"}
    definitions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    constants = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"STRUCTURED_STAGE_ORDER", "STRUCTURED_PHASE_STAGE_RULES",
                                              "LTX_POSTPROCESSING_MODEL_TYPES", "MAX_STEP_TELEMETRY"} for t in n.targets)]
    scope = {"importlib": importlib, "inspect": inspect, "time": time, "threading": threading, "wraps": wraps,
             "_telemetry_value": lambda v: v, "_memory_snapshot": lambda _: {}}
    exec(compile(ast.Module(body=constants + definitions, type_ignores=[]), "enhancement-test", "exec"), scope)
    scope["_task_owned_stage_id"] = lambda task, phase, timing: scope["_structured_stage_id"](phase)
    return scope


class EnhancementTests(unittest.TestCase):
    def owner(self, processor, temporal=False):
        scope = helpers()
        owner = types.SimpleNamespace(_stage_timing=scope["_StageTimingTelemetry"](),
                                      _generation_timing_observer_installed=False,
                                      _postprocessing_step_observer_installed=False, _latest_performance=None)
        name = "perform_temporal_upsampling" if temporal else "perform_spatial_upsampling"
        setattr(owner, name, processor)
        owner.set_global = lambda name, value: setattr(owner, name, value)
        scope["_install_postprocessing_step_observer"](owner)
        return scope, owner, name

    def generate(self, scope, owner, body):
        owner.generate_media = body
        scope["_install_generation_timing_observer"](owner)
        return owner.generate_media({"id": "task", "params": {"model_type": "flux"}}, lambda *args: None, state={"gen": {}})

    def test_real_enhancement_labels_match_python_and_javascript(self):
        labels = ["Spatial Upsampling", "Temporal Upsampling - RIFE", "DLSS 5 Neural Rendering",
                  "DLSS Frame Generation", "Adding Film Grain", "Applying Media Post Processing"]
        scope = helpers()
        for label in labels:
            self.assertEqual(scope["_structured_stage_id"](label), "post", label)
        node = shutil.which("node")
        if node:
            script = "const api = globalThis.__statusProReleaseTest;\n" + "\n".join(
                f'if (api.structuredStageId({json.dumps(label)}) !== "post") throw new Error({json.dumps(label)});'
                for label in labels)
            result = subprocess.run([node, "-"], input=_javascript_with_exports("structuredStageId") + script,
                                    text=True, encoding="utf-8", capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_inline_spatial_internal_phases_stay_in_enhance(self):
        seen = []
        def processor(sample, spatial_upsampling, progress_callback=None):
            # Entry already switched before the processor emits its first event.
            self.assertIsNotNone(owner._stage_timing.snapshot()["enhancement"])
            for phase in ("Loading model", "Encoding Prompt", "Denoising", "VAE Decoding", "Caching"):
                progress_callback(phase, 2, 12, "tiles")
                seen.append(owner._stage_timing.snapshot())
            return sample
        scope, owner, name = self.owner(processor)
        def body(task, send_cmd, state=None):
            send_cmd("status", "VAE Decoding")
            def callback(phase, current, total, unit):
                send_cmd("progress", [(current, total), phase, total, unit])
                return "forwarded"
            self.assertEqual(getattr(owner, name)("pixels", "flashvsr2", callback), "pixels")
            self.assertIsNone(owner._stage_timing.snapshot()["enhancement"])
            send_cmd("status", "Saving output")
            self.assertEqual(owner._stage_timing.snapshot()["last_stage"], "save")
            return True
        self.assertTrue(self.generate(scope, owner, body))
        self.assertTrue(all(x["last_stage"] == "post" for x in seen))
        self.assertEqual(seen[-1]["enhancement"], {"phase": "Caching", "current": 2, "total": 12, "unit": "tiles", "progress": 2 / 12 * 100, "stage": "post"})
        self.assertEqual(set(seen[-1]["stages"]), {"prepare", "decode", "post"})
        self.assertIsNone(owner._generation_context.execution)

    def test_temporal_standalone_and_silent_processors_enter_enhance(self):
        for temporal in (False, True):
            with self.subTest(temporal=temporal):
                def processor(*args, **kwargs):
                    snapshot = owner._stage_timing.snapshot()
                    self.assertEqual(snapshot["last_stage"], "post")
                    self.assertIsNone(snapshot["enhancement"]["current"])
                    return "result"
                # Use actual positional signatures, as WanGP does.
                if temporal:
                    def entry(sample, previous_last_frame, temporal_upsampling, fps, progress_callback=None):
                        return processor()
                else:
                    def entry(sample, spatial_upsampling, progress_callback=None):
                        return processor()
                scope, owner, name = self.owner(entry, temporal)
                def body(task, send_cmd, state=None):
                    args = ("pixels", None, "rife2", 24) if temporal else ("pixels", "dlss5")
                    self.assertEqual(getattr(owner, name)(*args), "result")
                    return True
                self.assertTrue(self.generate(scope, owner, body))

    def test_temporal_frame_progress_and_callback_return_are_preserved(self):
        def processor(sample, previous_last_frame, temporal_upsampling, fps, progress_callback=None):
            self.assertEqual(progress_callback("Temporal Upsampling - RIFE", 3, 12, "frames"), "callback result")
            snapshot = owner._stage_timing.snapshot()
            self.assertEqual(snapshot["last_stage"], "post")
            self.assertEqual(snapshot["enhancement"]["unit"], "frames")
            return sample, previous_last_frame, fps * 2
        scope, owner, _ = self.owner(processor, temporal=True)
        def body(task, send_cmd, state=None):
            callback = lambda *args: "callback result"
            self.assertEqual(owner.perform_temporal_upsampling("pixels", "last", "rife2", 24, callback),
                             ("pixels", "last", 48))
            return True
        self.generate(scope, owner, body)

    def test_callback_from_finished_processor_cannot_update_next_task(self):
        retained = []
        def processor(sample, spatial_upsampling, progress_callback=None):
            retained.append(progress_callback)
            return sample
        scope, owner, _ = self.owner(processor)
        def body(task, send_cmd, state=None):
            owner.perform_spatial_upsampling(None, "flashvsr2", lambda *args: None)
            owner._stage_timing.start_task("next", new_execution=True)
            retained[0]("VAE Decoding", 3, 12)
            self.assertEqual(owner._stage_timing.snapshot()["last_stage"], "prepare")
            self.assertIsNone(owner._stage_timing.snapshot()["enhancement"])
            return True
        self.generate(scope, owner, body)

    def test_exception_and_nested_processor_scopes_cleanup(self):
        def processor(sample, spatial_upsampling, progress_callback=None):
            if spatial_upsampling == "outer":
                with self.assertRaisesRegex(RuntimeError, "processor failed"):
                    owner.perform_spatial_upsampling(sample, "inner")
                self.assertIsNotNone(owner._stage_timing.snapshot()["enhancement"])
                raise RuntimeError("processor failed")
            raise RuntimeError("processor failed")
        scope, owner, _ = self.owner(processor)
        def body(task, send_cmd, state=None):
            with self.assertRaisesRegex(RuntimeError, "processor failed"):
                owner.perform_spatial_upsampling(None, "outer")
            self.assertIsNone(owner._stage_timing.snapshot()["enhancement"])
            send_cmd("status", "VAE Decoding")
            self.assertEqual(owner._stage_timing.snapshot()["last_stage"], "decode")
            return True
        self.generate(scope, owner, body)

    def test_ltx_units_forwarded_without_counting_tiles_as_steps(self):
        calls = []
        def processor(sample, spatial_upsampling, progress_callback=None):
            progress_callback("Upsampling - VAE decoding", 4, 12, "tiles")
            for step in range(1, 9):
                progress_callback("Upsampling - Denoising", step, 8, "steps")
            return sample
        scope, owner, _ = self.owner(processor)
        def body(task, send_cmd, state=None):
            owner.perform_spatial_upsampling(None, "ltx252", progress_callback=lambda *args: calls.append(args))
            return True
        self.generate(scope, owner, body)
        self.assertEqual(calls[0], ("Upsampling - VAE decoding", 4, 12, "tiles"))
        self.assertEqual([s["step"] for s in owner._latest_performance["steps"]], list(range(1, 9)))

    def test_unowned_and_retired_execution_cannot_change_next_task(self):
        def processor(sample, spatial_upsampling, progress_callback=None):
            if callable(progress_callback):
                progress_callback("VAE Decoding", 1, 2)
            return sample
        scope, owner, _ = self.owner(processor)
        self.assertEqual(owner.perform_spatial_upsampling("outside", "dlss5"), "outside")
        self.assertIsNone(owner._stage_timing.snapshot()["last_stage"])
        def body(task, send_cmd, state=None):
            owner._stage_timing.start_task("next", new_execution=True)
            owner.perform_spatial_upsampling("retired", "dlss5")
            self.assertEqual(owner._stage_timing.snapshot()["last_stage"], "prepare")
            self.assertIsNone(owner._stage_timing.snapshot()["enhancement"])
            return True
        self.generate(scope, owner, body)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_browser_uses_owned_processor_progress_and_rejects_stale_context(self):
        javascript = _javascript_with_exports("readReportedPhaseStatus", "readLiveSnapshot", "freshState")
        script = r"""
const api = globalThis.__statusProReleaseTest;
const ns = {state: api.freshState(), activeRun: {queue_task_id: "task", _stageTimingEpoch: 1, settings: {}},
    runTelemetry: {execution_task_known: true, executing_task: {id: "task"}, status: "",
        native_progress: {phase: "VAE Decoding", current: 99, total: 100, unit: "tiles"},
        stage_timing: {task_id: "task", execution_epoch: 1, stages: {post: {active: true}},
            enhancement: {phase: "Upsampling", current: null, total: null, unit: null}}}};
function check(condition, message) {if (!condition) throw new Error(message);}
let snapshot = api.readLiveSnapshot(ns);
check(snapshot.id === "post" && snapshot.progress === null && snapshot.steps.current === null, "stale VAE counter on entry");
for (const phase of ["Denoising", "VAE Decoding", "Temporal Upsampling - RIFE", "Unknown future processor"]){
    ns.runTelemetry.stage_timing.enhancement = {phase, current: 3, total: 12, unit: "frames"};
    snapshot = api.readLiveSnapshot(ns);
    check(snapshot.id === "post" && snapshot.progress === 25 && snapshot.steps.unit === "frames", phase);
}
ns.runTelemetry.model_lifecycle = {state: "unloading", model_name: "generation model"};
check(api.readLiveSnapshot(ns).id === "post", "model release displaced Enhance");
ns.runTelemetry.model_lifecycle = null;
ns.runTelemetry.stage_timing.execution_epoch = 2;
check(api.readReportedPhaseStatus(ns).id === "decode", "stale execution enhancement trusted");
ns.runTelemetry.stage_timing.execution_epoch = 1;
ns.runTelemetry.stage_timing.task_id = "different";
check(api.readReportedPhaseStatus(ns).id === "decode", "other task enhancement trusted");
ns.runTelemetry.stage_timing.task_id = "task";
ns.runTelemetry.stage_timing.enhancement = null;
ns.runTelemetry.stage_timing.stages = {save: {active: true}};
ns.runTelemetry.native_progress = {phase: "Saving output"};
check(api.readLiveSnapshot(ns).id === "save", "Save failed after processor finished");
"""
        result = subprocess.run([shutil.which("node"), "-"], input=javascript + script,
                                text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

"""H3 two-phase handoff regressions shared by Status Pro and Lite."""
import json
import shutil
import subprocess
import threading
import types
import unittest

from test_enhancement_stages import helpers
from test_release_smoke import _javascript_with_exports


class TwoPhaseTests(unittest.TestCase):
    def setup_pipeline(self, scope=None):
        scope = scope or helpers()
        timer = scope["_StageTimingTelemetry"]()
        epoch = timer.start_task("task", new_execution=True)
        context = threading.local()
        context.execution = ("task", epoch)
        observed = []
        class Pipeline:
            __module__ = "models.minimax_h3.pipeline"
            def _use_shared_components(self, value=None):
                observed.append(timer.snapshot())
                return value
        pipe = Pipeline()
        gen = {}
        handoff = scope["_install_h3_handoff_observer"](pipe, context, timer, gen, "task", epoch)
        return scope, timer, epoch, context, pipe, gen, handoff, observed

    def test_handoff_status_precedes_silent_unload_and_preserves_return(self):
        scope, timer, epoch, context, pipe, gen, handoff, seen = self.setup_pipeline()
        pipe._use_shared_components("initial")
        self.assertEqual(seen[-1]["last_stage"], "prepare")
        handoff["pass_no"] = 1
        timer.observe_progress("task", epoch, "VAE Decoding", 8, 8, stage_id="decode")
        self.assertEqual(pipe._use_shared_components("return"), "return")
        self.assertEqual(seen[-1]["last_stage"], "post")
        self.assertIn("preparing model handoff", seen[-1]["progress"]["phase"])
        self.assertIsNone(seen[-1]["progress"]["current"])
        self.assertEqual(gen["progress_phase"], (seen[-1]["progress"]["phase"], -1))
        timer.observe_progress("task", epoch, "Upscaling H3 latent between phases (7/35)", stage_id="post")
        self.assertEqual(timer.snapshot()["progress"]["unit"], "layers")
        timer.observe_progress("task", epoch, "Encoding H3 phase 2 prompt and references", stage_id="encode")
        timer.observe_progress("task", epoch, "Phase 2/2 Tiled", stage_id="denoise")
        handoff["pass_no"] = 2
        timer.observe_progress("task", epoch, "VAE Decoding", stage_id="decode")
        pipe._use_shared_components("final")
        self.assertEqual(seen[-1]["last_stage"], "decode")

    def test_spectrum_replay_stays_generate_until_real_handoff(self):
        scope, timer, epoch, context, pipe, gen, handoff, seen = self.setup_pipeline()
        handoff["pass_no"] = 1
        timer.observe_progress("task", epoch, "VAE Decoding", stage_id="decode")
        # Spectrum does not call _use_shared_components between capture and replay.
        timer.observe_progress("task", epoch, "Denoising First Phase | Spectrum smoothing replay", stage_id="denoise")
        self.assertNotIn("post", timer.snapshot()["stages"])
        timer.observe_progress("task", epoch, "VAE Decoding", stage_id="decode")
        pipe._use_shared_components()
        self.assertEqual(seen[-1]["last_stage"], "post")

    def test_unowned_retired_single_pass_and_aborted_handoffs_stay_decode(self):
        for case in ("unowned", "retired", "single", "aborted", "early_stop", "thread"):
            with self.subTest(case=case):
                scope, timer, epoch, context, pipe, gen, handoff, seen = self.setup_pipeline()
                handoff["pass_no"] = 1
                timer.observe_progress("task", epoch, "VAE Decoding", stage_id="decode")
                if case == "unowned": context.execution = None
                if case == "retired":
                    timer.finish_task("task")
                    timer.start_task("task", new_execution=True)
                if case == "single": handoff["pass_no"] = -1
                if case == "aborted": gen["abort"] = True
                if case == "early_stop": gen["early_stop"] = True
                if case == "thread":
                    worker = threading.Thread(target=pipe._use_shared_components)
                    worker.start(); worker.join()
                else:
                    pipe._use_shared_components()
                self.assertNotEqual(seen[-1]["last_stage"], "post")

    def test_real_builder_transformer_argument_uses_runtime_pipeline_and_pass(self):
        scope, timer, epoch, context, pipe, gen, handoff, seen = self.setup_pipeline()
        scope["_skip_count"] = lambda pipe: None
        state = {"gen": {"api_active_queue_task": {"id": "task"}}}
        # Match WanGP: its callback builder receives a transformer, while the
        # pipeline lives in the function's globals and changes when models load.
        builder_scope = {"wan_model": pipe}
        exec("def build_callback(state, transformer, *args, **kwargs):\n"
             "    def callback(*args, **kwargs):\n"
             "        state['gen']['progress_phase'] = ('VAE Decoding', 8)\n"
             "        return 'callback result'\n"
             "    return callback\n", builder_scope)
        owner = types.SimpleNamespace(_step_observer_installed=False, _stage_timing=timer,
                                      _generation_context=context, build_callback=builder_scope["build_callback"])
        owner.set_global = lambda name, value: setattr(owner, name, value)
        scope["_install_step_observer"](owner)
        for _ in range(2):
            callback = owner.build_callback(state, object(), num_inference_steps=8)
            timer.observe_progress("task", epoch, "Denoising First Phase", stage_id="denoise")
            # Positional pass_no must work as well as H3's usual keyword form.
            self.assertEqual(callback(7, None, False, False, -1, 1), "callback result")
            pipe._use_shared_components()
            self.assertEqual(seen[-1]["last_stage"], "post")
        self.assertEqual(len(seen), 2)  # Reinstalling does not stack wrappers.

    def test_native_phase_labels_and_layer_counter_validation(self):
        scope = helpers()
        for phase in ("Phase 1/2 Low Resolution", "Phase 2/2 High Resolution", "Phase 2/2 Tiled",
                      "Sample 2/3 - Phase 2/2 High Resolution", "Audio Refinement Extra Phase (3 steps, denoising 0.3)"):
            self.assertEqual(scope["_structured_stage_id"](phase), "denoise", phase)
        self.assertIsNone(scope["_structured_stage_id"]("Phase 2/2 Unknown work"))
        validate = scope["_validated_phase_progress"]
        counter = validate("Sample 1/2 - Upscaling H3 latent between phases (7/35)")
        self.assertEqual((counter["current"], counter["total"], counter["unit"], counter["progress"]), (7, 35, "layers", 20))
        self.assertIsNone(validate("Upscaling H3 latent between phases (36/35)")["progress"])
        self.assertIsNone(validate("Upscaling H3 latent between phases")["current"])

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_browser_two_phase_transitions_override_stale_decode(self):
        js = _javascript_with_exports("freshState", "readLiveSnapshot", "applySnapshot", "applyServerStageTiming",
                                     "structuredStageId", "phaseInfo", "formatCounter", "normalizedPhaseLabel")
        script = r"""
const api = globalThis.__statusProReleaseTest;
function check(v,m) {if (!v) throw new Error(m);}
const ns={state:api.freshState(),activeRun:{queue_task_id:'task',_stageTimingEpoch:1,settings:{model_type:'minimax_h3'}},
 runTelemetry:{execution_task_known:true,executing_task:{id:'task'},status:'',native_progress:{phase:'VAE Decoding',current:8,total:8,unit:'steps'}}};
const events=[['denoise','Denoising First Phase',2,8,'steps'],['decode','VAE Decoding'],
 ['post','Upscaling H3 latent between phases - preparing model handoff'],['post','Upscaling H3 latent between phases (7/35)',7,35,'layers'],
 ['encode','Encoding H3 phase 2 prompt and references'],['denoise','Phase 2/2 Tiled'],
 ['denoise','Denoising Second Phase | Phase 2/2 Tiled',1,12,'steps'],['decode','VAE Decoding']];
for (const [stage,phase,current,total,unit] of events) {
 ns.runTelemetry.stage_timing={task_id:'task',execution_epoch:1,stages:{[stage]:{active:true,elapsed:1}},
   progress:{stage,phase,current:current??null,total:total??null,unit:unit??null}};
 api.applyServerStageTiming(ns,ns.runTelemetry);
 const snapshot=api.readLiveSnapshot(ns);
 check(snapshot.id===stage && snapshot.rawName===api.normalizedPhaseLabel(phase,{current,total,unit}),'stale Decode hid '+phase);
 check(snapshot.rawMessage===phase,'native activity message was lost');
 api.applySnapshot(ns,snapshot);
 check(ns.state.currentId===stage,'transition rejected '+phase);
 if (unit==='layers') check(api.formatCounter(snapshot.steps)==='7/35 layers' && snapshot.progress===20,'latent layers lost');
}
for (const phase of ['Phase 1/2 Low Resolution','Phase 2/2 High Resolution','Phase 2/2 Tiled','Audio Refinement Extra Phase (3 steps, denoising 0.3)'])
 check(api.structuredStageId(phase)==='denoise',phase);
check(api.phaseInfo('Phase 2/2 High Resolution').phaseNumber===2,'native second phase number lost');
check(api.structuredStageId('Phase 2/2 Unknown work')===null,'unknown status guessed');
"""
        result = subprocess.run([shutil.which("node"), "-"], input=js+script, text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

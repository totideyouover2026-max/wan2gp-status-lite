"""Regression coverage for queued/worker boundaries with a resident model."""
import ast
import json
import types
import unittest
from unittest.mock import patch

from test_enhancement_stages import helpers
from test_release_smoke import _source


def snapshot_helpers():
    scope = helpers()
    names = {"_run_snapshot_json", "_recover_polled_stage_timing", "_task_model_type", "_TaskOutcomeTelemetry"}
    nodes = [n for n in ast.walk(ast.parse(_source()))
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    scope.update(json=json, _latest_performance_snapshot=lambda *args: None,
                 _output_records=lambda *args, **kwargs: [],
                 _task_telemetry=lambda task, **kwargs: {
                     "id": task["id"], "settings": dict(task.get("params", {})), "planned_stages": []},
                 MODEL_LIFECYCLE_TELEMETRY=types.SimpleNamespace(snapshot=lambda: {}))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "worker-boundary-test", "exec"), scope)
    return scope


class WorkerBoundaryTests(unittest.TestCase):
    def owner(self, scope, generate):
        owner = types.SimpleNamespace(_stage_timing=scope["_StageTimingTelemetry"](),
            _generation_timing_observer_installed=False, generate_media=generate,
            _runtime_id="test", _latest_performance=None,
            _task_outcomes=scope["_TaskOutcomeTelemetry"]())
        owner.set_global = lambda name, value: setattr(owner, name, value)
        scope["_install_generation_timing_observer"](owner)
        return owner

    def snapshot(self, scope, owner, state):
        saved = json.loads(scope["_run_snapshot_json"](owner, state))
        self.assertNotIn("error", saved)
        return saved

    def test_image_to_video_gap_cannot_seed_save_before_worker_starts(self):
        for reload_model in (False, True):
            with self.subTest(reload_model=reload_model):
                scope = snapshot_helpers()
                seen = []
                clock = [0]
                image = {"id": "image", "params": {"model_type": "minimax_h3_t2v", "image_mode": 1}}
                video = {"id": "video", "params": {"model_type": "minimax_h3_t2v", "image_mode": 0}}
                state = {"gen": {"queue": [image], "in_progress": True,
                                 "repeat_no": 1, "window_no": 1}}
                def generate(task, send_cmd, state=None):
                    if task is video:
                        seen.append(self.snapshot(scope, owner, state))
                        if reload_model:
                            send_cmd("status", "Loading model")
                        clock[0] = 4
                        send_cmd("progress", [(2, 32), "Encoding H3 Prompt", 32, "layers"])
                        seen.append(self.snapshot(scope, owner, state))
                        clock[0] = 6
                        send_cmd("progress", [(3, 10), "Denoising", 10, "steps"])
                        seen.append(self.snapshot(scope, owner, state))
                        clock[0] = 16
                        send_cmd("status", "VAE Decoding")
                        clock[0] = 18
                        send_cmd("status", "Saving File video.mp4")
                        clock[0] = 19
                    else:
                        send_cmd("status", "Saving File previous-image.jpg")
                    return True
                owner = self.owner(scope, generate)
                def publish(command, data):
                    if command == "status":
                        state["gen"]["status"] = data
                    elif command == "progress":
                        state["gen"]["progress_phase"] = (data[1], -1)
                with patch("time.monotonic", side_effect=lambda: clock[0]):
                    state["gen"]["api_active_queue_task"] = image
                    owner.generate_media(image, publish, state=state)
                    state["gen"]["progress_phase"] = ("Saving File previous-image.jpg", -1)
                    state["gen"].pop("api_active_queue_task")  # WanGP deletes the key between jobs.
                    state["gen"]["queue"] = [video]
                    clock[0] = 2
                    gap = self.snapshot(scope, owner, state)
                    self.assertTrue(gap["execution_task_known"])
                    self.assertIsNone(gap["executing_task"])
                    self.assertIsNone(gap["active_task"])
                    self.assertEqual(gap["stage_timing"]["task_id"], "image")
                    clock[0] = 3
                    state["gen"]["api_active_queue_task"] = video
                    ready = self.snapshot(scope, owner, state)
                    self.assertEqual(ready["stage_timing"]["last_stage"], "prepare")
                    owner.generate_media(video, publish, state=state)
                    final = self.snapshot(scope, owner, state)["stage_timing"]
                self.assertEqual([s["stage_timing"]["last_stage"] for s in seen],
                                 ["prepare", "encode", "denoise"])
                self.assertEqual(final["stages"]["denoise"]["elapsed"], 10)
                self.assertEqual(final["stages"]["save"]["elapsed"], 1)
                self.assertEqual(final["progress"]["phase"], "Saving File video.mp4")

    def test_observed_worker_is_authoritative_without_native_task_field(self):
        scope = snapshot_helpers()
        task = {"id": 1, "params": {"model_type": "minimax_h3_t2v"}}
        state = {"gen": {"in_progress": True, "queue": [task]}}
        seen = []
        def generate(task, send_cmd, state=None):
            # No native api_active_queue_task, as in older hosts.
            send_cmd("status", "Denoising")
            seen.append(self.snapshot(scope, owner, state))
            return True
        owner = self.owner(scope, generate)
        before = self.snapshot(scope, owner, state)
        self.assertTrue(before["execution_task_known"])
        self.assertIsNone(before["active_task"])
        owner.generate_media(task, lambda *args: None, state=state)
        after = self.snapshot(scope, owner, state)
        self.assertEqual(seen[0]["executing_task"]["id"], 1)
        self.assertEqual(seen[0]["stage_timing"]["last_stage"], "denoise")
        self.assertIsNone(after["executing_task"])
        self.assertEqual(state, {"gen": {"in_progress": True, "queue": [task]}})

    def test_unobserved_legacy_host_keeps_queue_fallback(self):
        scope = snapshot_helpers()
        owner = self.owner(scope, lambda *args, **kwargs: True)
        owner._generation_timing_observer_installed = False
        task = {"id": "legacy", "params": {}}
        saved = self.snapshot(scope, owner, {"gen": {"in_progress": True, "queue": [task],
                              "progress_phase": ("Denoising", -1)}})
        self.assertFalse(saved["execution_task_known"])
        self.assertEqual(saved["active_task"]["id"], "legacy")
        self.assertEqual(saved["stage_timing"]["last_stage"], "denoise")

    def test_poll_during_worker_entry_cannot_finish_the_new_execution(self):
        scope = snapshot_helpers()
        task = {"id": "entry", "params": {}}
        state = {"gen": {"in_progress": True, "queue": [task]}}
        seen = []
        def generate(task, send_cmd, state=None):
            send_cmd("status", "Denoising")
            seen.append(self.snapshot(scope, owner, state)["stage_timing"])
            return True
        owner = self.owner(scope, generate)
        original_begin = owner._task_outcomes.begin
        def begin(*args, **kwargs):
            original_begin(*args, **kwargs)
            # A browser poll can arrive after timing starts, before the worker
            # publishes ownership on an older host with no native task field.
            self.snapshot(scope, owner, state)
        owner._task_outcomes.begin = begin
        owner.generate_media(task, lambda *args: None, state=state)
        self.assertTrue(seen[0]["stages"]["denoise"]["active"])
        self.assertIsNone(seen[0]["completed_at"])

    def test_failed_worker_releases_ownership_and_reused_id_starts_fresh(self):
        scope = snapshot_helpers()
        task = {"id": "reused", "params": {}}
        state = {"gen": {"in_progress": True, "queue": [task]}}
        seen = []
        fail = [True]
        def generate(task, send_cmd, state=None):
            send_cmd("status", "Denoising")
            seen.append(self.snapshot(scope, owner, state))
            if fail[0]:
                raise RuntimeError("synthetic model error")
            return True
        owner = self.owner(scope, generate)
        with self.assertRaisesRegex(RuntimeError, "synthetic model error"):
            owner.generate_media(task, lambda *args: None, state=state)
        idle = self.snapshot(scope, owner, state)
        self.assertIsNone(idle["executing_task"])
        self.assertEqual(idle["task_outcomes"][-1]["error"], "synthetic model error")
        self.assertEqual(set(state["gen"]), {"in_progress", "queue"})
        fail[0] = False
        owner.generate_media(task, lambda *args: None, state=state)
        self.assertGreater(seen[1]["stage_timing"]["execution_epoch"],
                           seen[0]["stage_timing"]["execution_epoch"])
        self.assertTrue(seen[1]["stage_timing"]["stages"]["denoise"]["active"])
        self.assertIsNone(self.snapshot(scope, owner, state)["executing_task"])

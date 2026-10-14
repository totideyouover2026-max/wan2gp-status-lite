"""PiD processor dispatch and native worker ownership regressions."""
import json
import shutil
import subprocess
import threading
import types
import unittest
from unittest.mock import patch

from test_enhancement_stages import helpers
from test_release_smoke import _javascript_with_exports


class PiDStatusTests(unittest.TestCase):
    def runtime(self, stage):
        scope = helpers()
        timer = scope['_StageTimingTelemetry']()
        epoch = timer.start_task('task', new_execution=True)
        timer.observe_progress('task', epoch, 'VAE Decoding' if stage == 'decode' else 'Preparing', stage_id=stage)
        gen = {'api_active_queue_task': {'id': 'task'}, 'repeat_no': 1, 'window_no': 1}
        native_scope = threading.local()
        native_scope.control = types.SimpleNamespace(gen=gen)
        native = types.SimpleNamespace(current_control=lambda: getattr(native_scope, 'control', None))
        seen, forwarded = [], []
        def dispatcher(handler, sample, spatial_upsampling, *, main_offloadobj=None, **kwargs):
            # The real dispatcher loads the upscaler before its first callback.
            seen.append(timer.snapshot())
            progress = kwargs.get('progress_callback')
            if progress:
                progress('PiD VAE Encode')
                seen.append(timer.snapshot())
                self.assertEqual(progress('PiD', 3, 60), 'callback-result')
                seen.append(timer.snapshot())
            return sample, 'cache'
        spatial = types.SimpleNamespace(upscale_postprocessing=dispatcher)
        temporal = types.SimpleNamespace(temporal_upsample=lambda *a, **k: None)
        modules = {'postprocessing.spatial_upsamplers': spatial,
                   'postprocessing.temporal_upsamplers': temporal,
                   'shared.utils.media_control': native}
        def imports(name):
            if name not in modules: raise ImportError(name)
            return modules[name]
        owner = types.SimpleNamespace(_stage_timing=timer, _generation_context=threading.local(),
                                      _postprocessing_step_observer_installed=False, _latest_performance=None)
        # Keep a reference to the host helper before plugin installation. A
        # native module dispatcher must still be observed through that alias.
        def host_helper(sample, spatial_upsampling, progress_callback=None):
            return spatial.upscale_postprocessing(None, sample, spatial_upsampling, progress_callback=progress_callback)
        owner.perform_spatial_upsampling = host_helper
        owner.set_global = lambda key, value: setattr(owner, key, value)
        with patch.object(scope['importlib'], 'import_module', side_effect=imports):
            scope['_install_postprocessing_step_observer'](owner)
        def callback(*args):
            forwarded.append(args)
            return 'callback-result'
        return scope, owner, epoch, gen, native_scope, imports, host_helper, spatial, callback, seen, forwarded

    def test_native_dispatcher_enters_enhance_before_pid_loading_inline_and_late(self):
        for stage in ('decode', 'prepare'):
            with self.subTest(stage=stage):
                scope, owner, epoch, gen, native, imports, alias, spatial, cb, seen, forwarded = self.runtime(stage)
                with patch.object(scope['importlib'], 'import_module', side_effect=imports):
                    self.assertEqual(alias('pixels', 'pid_flux2_post*4', cb), ('pixels', 'cache'))
                self.assertEqual([x['last_stage'] for x in seen], ['post', 'post', 'post'])
                self.assertEqual(seen[0]['enhancement']['phase'], 'PiD Spatial Upsampling')
                self.assertIsNone(seen[0]['progress']['current'])  # Clear the generation tile counter.
                self.assertEqual(seen[1]['progress']['phase'], 'PiD VAE Encode')
                self.assertIsNone(seen[1]['progress']['progress'])
                self.assertEqual((seen[2]['progress']['current'], seen[2]['progress']['total']), (3, 60))
                self.assertEqual(forwarded, [('PiD VAE Encode', None, None), ('PiD', 3, 60)])
                self.assertIsNone(owner._stage_timing.snapshot()['enhancement'])
                self.assertIsNone(getattr(owner._generation_context, 'execution', None))

    def test_nested_host_and_dispatcher_scopes_cleanup_and_exception(self):
        scope, owner, epoch, gen, native, imports, alias, spatial, cb, seen, forwarded = self.runtime('decode')
        owner._generation_context.execution = ('task', epoch)
        with patch.object(scope['importlib'], 'import_module', side_effect=imports):
            owner.perform_spatial_upsampling('pixels', 'pid_flux_post*4', cb)
            self.assertIsNone(owner._stage_timing.snapshot()['enhancement'])
            def fail(*args): raise RuntimeError('PiD failed')
            with self.assertRaisesRegex(RuntimeError, 'PiD failed'):
                owner.perform_spatial_upsampling('pixels', 'pid_flux_post*4', fail)
        self.assertIsNone(owner._stage_timing.snapshot()['enhancement'])
        self.assertEqual(owner._stage_timing.snapshot()['last_stage'], 'post')

    def test_nested_dispatch_does_not_duplicate_ltx_performance_observers(self):
        scope, owner, epoch, gen, native, imports, alias, spatial, cb, seen, forwarded = self.runtime('decode')
        owner._generation_context.execution = ('task', epoch)
        created = []
        original = scope['_new_performance_observer']
        def create(task_id):
            value = original(task_id)
            created.append(value)
            return value
        scope['_new_performance_observer'] = create
        method = next(iter(scope['LTX_POSTPROCESSING_MODEL_TYPES']))
        with patch.object(scope['importlib'], 'import_module', side_effect=imports):
            owner.perform_spatial_upsampling('pixels', method, cb)
        self.assertEqual(len(created), 1)
        self.assertIs(owner._latest_performance, created[0])
        self.assertEqual([x['step'] for x in created[0]['steps']], [3])

    def test_native_scope_rejects_foreign_finished_and_unowned_workers(self):
        for case in ('foreign', 'finished', 'unowned', 'thread'):
            with self.subTest(case=case):
                scope, owner, epoch, gen, native, imports, alias, spatial, cb, seen, forwarded = self.runtime('decode')
                if case == 'foreign': gen['api_active_queue_task'] = {'id': 'other'}
                if case == 'finished': owner._stage_timing.finish_task('task')
                if case == 'unowned': native.control = None
                with patch.object(scope['importlib'], 'import_module', side_effect=imports):
                    if case == 'thread':
                        worker = threading.Thread(target=alias, args=('pixels', 'pid_qwen_post*4'))
                        worker.start(); worker.join()
                    else: alias('pixels', 'pid_qwen_post*4')
                self.assertNotIn('post', owner._stage_timing.snapshot()['stages'])

    def test_pid_labels_are_explicit_enhance_events_in_both_matchers(self):
        scope = helpers()
        labels = ['PiD', 'PiD VAE Encode', 'PiD Spatial Upsampling', 'Sample 1/3 - PiD VAE Encode']
        for label in labels:
            self.assertEqual(scope['_structured_stage_id'](label), 'post', label)
        self.assertEqual(scope['_structured_stage_id']('VAE Encoding'), 'input')
        self.assertEqual(scope['_structured_stage_id']('VAE Decoding'), 'decode')
        if shutil.which('node'):
            js = _javascript_with_exports('structuredStageId', 'stageIdFor')
            script = 'const api=globalThis.__statusProReleaseTest;\n' + '\n'.join(
                f'if(api.structuredStageId({json.dumps(label)})!=="post") throw new Error({json.dumps(label)});'
                for label in labels)
            script += '\nif(api.stageIdFor("PiD VAE Encode")!=="post") throw new Error("legacy matcher");'
            result = subprocess.run([shutil.which('node'), '-'], input=js+script, text=True, encoding='utf-8', capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

"""Shared enhancement ownership, dispatcher and VAE contract regressions."""
import inspect
import json
import shutil
import subprocess
import threading
import types
import unittest
from unittest.mock import patch

from test_enhancement_stages import helpers
from test_release_smoke import _javascript_with_exports


SPATIAL = ('Lanczos', 'DLSS5', 'FlashVSR', 'SeedVR2', 'PiD', 'H3 Face Refiner', 'Chain of Zoom', 'LTX Upscale', 'LTX Refine', 'Future Registered Upscaler')
TEMPORAL = ('RIFE', 'DLSSG', 'Future Temporal Processor')
AUDIO = ('MMAudio', 'PrismAudio', 'SeedVC', 'Audio Background Removal', 'Custom Soundtrack', 'Future Audio Processor')


class EnhancementCoverageTests(unittest.TestCase):
    def setup_owner(self, functions, context=True):
        scope = helpers()
        timer = scope['_StageTimingTelemetry']()
        epoch = timer.start_task('task', new_execution=True)
        owner = types.SimpleNamespace(_stage_timing=timer, _generation_context=threading.local(),
            _postprocessing_step_observer_installed=False, _latest_performance=None)
        if context: owner._generation_context.execution = ('task', epoch)
        owner.set_global = lambda key, value: setattr(owner, key, value)
        modules = {name: types.SimpleNamespace(**entries) for name, entries in functions.items()}
        def imports(name):
            if name not in modules: raise ImportError(name)
            return modules[name]
        with patch.object(scope['importlib'], 'import_module', side_effect=imports):
            scope['_install_postprocessing_step_observer'](owner)
        return scope, timer, epoch, owner, modules, imports

    def test_all_registered_processor_families_and_future_handlers_use_same_scope(self):
        for family, names in (('spatial', SPATIAL), ('temporal', TEMPORAL), ('audio', AUDIO)):
            for name in names:
                for start in ('prepare', 'decode', 'save'):
                    with self.subTest(family=family, processor=name, start=start):
                        seen, forwarded = [], []
                        def processing(*args, progress_callback=None, send_cmd=None, **kwargs):
                            seen.append(timer.snapshot())  # Before model loading or any callback.
                            send_cmd('status', 'Unrecognised download/transfer activity')
                            for phase, unit in (('Encoding Prompt', 'layers'), ('Brand-new preprocess operation', 'tiles'),
                                                ('Denoising', 'steps'), ('VAE Decoding', 'frames')):
                                progress_callback(phase, 2, 8, unit)
                                seen.append(timer.snapshot())
                            return 'pixels'
                        functions = {
                            'postprocessing.spatial_upsamplers': {'upscale_postprocessing': processing, 'method_progress_label': lambda value: name},
                            'postprocessing.temporal_upsamplers': {'temporal_upsample': processing, 'format_temporal_upsampling_label': lambda value: name},
                            'postprocessing.audio_processors': {'_dispatch_processor': processing, 'format_method_label': lambda value: name}}
                        scope, timer, epoch, owner, modules, imports = self.setup_owner(functions)
                        timer.observe_progress('task', epoch, 'Previous stage', 15, 15, 'tiles', stage_id=start)
                        module, entry, key = {
                            'spatial': ('postprocessing.spatial_upsamplers', 'upscale_postprocessing', 'spatial_upsampling'),
                            'temporal': ('postprocessing.temporal_upsamplers', 'temporal_upsample', 'temporal_upsampling'),
                            'audio': ('postprocessing.audio_processors', '_dispatch_processor', 'method')}[family]
                        # Match an older host's three-argument callback. Units
                        # must remain in plugin telemetry without a TypeError.
                        def callback(phase, current, total): forwarded.append((phase, current, total))
                        def send(command, data):
                            timer.observe_progress('task', epoch, data, stage_id=scope['_structured_stage_id'](data))
                            seen.append(timer.snapshot())
                        result = getattr(modules[module], entry)(**{key:name}, progress_callback=callback, send_cmd=send)
                        self.assertEqual(result, 'pixels')
                        self.assertTrue(all(s['last_stage'] == 'post' for s in seen))
                        self.assertIn(name, seen[0]['progress']['phase'])
                        self.assertIsNone(seen[0]['progress']['current'])
                        self.assertEqual(seen[1]['progress']['phase'], 'Unrecognised download/transfer activity')
                        self.assertEqual([s['progress']['unit'] for s in seen[2:]], ['layers', 'tiles', 'steps', 'frames'])
                        self.assertEqual(len(forwarded), 4)
                        self.assertIsNone(timer.snapshot()['enhancement'])

    def test_missing_callbacks_and_silent_processors_still_report_enhance(self):
        seen = []
        def processor(*args, **kwargs):
            seen.append(timer.snapshot())
            callback = kwargs.get('progress_callback')
            self.assertTrue(callable(callback))
            inspect.signature(callback).bind('custom', 1, 4, 'tiles')
            callback('Custom progress', 1, 4, 'tiles')
            seen.append(timer.snapshot())
            return 'done'
        scope, timer, epoch, owner, modules, imports = self.setup_owner({
            'postprocessing.spatial_upsamplers': {'upscale_postprocessing': processor}})
        self.assertEqual(modules['postprocessing.spatial_upsamplers'].upscale_postprocessing(spatial_upsampling='unknown'), 'done')
        self.assertEqual(seen[0]['last_stage'], 'post')
        self.assertEqual(seen[1]['progress']['unit'], 'tiles')
        self.assertEqual(seen[1]['progress']['progress'], 25)
        self.assertIsNone(owner._latest_performance)  # Tile work is not denoising steps.

    def test_vae_preparation_resumes_generation_and_session_decodes_are_enhance(self):
        seen = []
        class Session:
            def decode_inputs(self, value, **kwargs): return self.decode(value, **kwargs)
            def decode(self, value, progress_callback=None):
                seen.append(timer.snapshot())
                progress_callback('Model-specific VAE work', 1, 4, 'tiles')
                seen.append(timer.snapshot())
                return value
        session = Session()
        def prepare(handler, spatial_upsampling, send_cmd=None, **kwargs):
            seen.append(timer.snapshot())
            send_cmd('status', 'Loading an unrecognised upscaler')
            return session
        def resize(sample, spatial_upsampling):
            seen.append(timer.snapshot())
            return sample
        scope, timer, epoch, owner, modules, imports = self.setup_owner({
            'postprocessing.spatial_upsamplers': {'prepare_vae_upsampler':prepare, 'post_model_process_vae_upsampling':resize}})
        api = modules['postprocessing.spatial_upsamplers']
        self.assertIs(api.prepare_vae_upsampler(object(), 'custom*4', send_cmd=lambda *args: None), session)
        self.assertEqual(seen[0]['last_stage'], 'post')
        self.assertEqual(timer.snapshot()['last_stage'], 'prepare')
        self.assertIsNone(timer.snapshot()['enhancement'])
        timer.observe_progress('task', epoch, 'Encoding Text Prompt', stage_id='encode')
        timer.observe_progress('task', epoch, 'Denoising', stage_id='denoise')
        self.assertEqual(session.decode_inputs('pixels'), 'pixels')
        self.assertTrue(all(s['last_stage'] == 'post' for s in seen))
        self.assertEqual(timer.snapshot()['progress']['unit'], 'tiles')
        self.assertEqual(api.post_model_process_vae_upsampling('resized', 'custom*4'), 'resized')
        self.assertEqual(timer.snapshot()['last_stage'], 'post')
        self.assertIsNone(timer.snapshot()['enhancement'])

    def test_native_vae_upsampling_owns_output_decode_but_not_conditioning(self):
        seen = []
        class VAE:
            upsampling_set = 'custom_vae*2'
            def decode(self, value):
                seen.append(timer.snapshot()['last_stage'])
                return value
        scope, timer, epoch, owner, modules, imports = self.setup_owner({})
        vae = VAE()
        owner._observe_model_enhancements(types.SimpleNamespace(vae=vae))
        self.assertEqual(vae.decode('conditioning'), 'conditioning')
        timer.observe_progress('task', epoch, 'Denoising', stage_id='denoise')
        timer.observe_progress('task', epoch, 'VAE Decoding', stage_id='decode')
        self.assertEqual(vae.decode('output'), 'output')
        timer.observe_progress('task', epoch, 'Encoding second phase prompt', stage_id='encode')
        self.assertEqual(vae.decode('second-phase reference'), 'second-phase reference')
        self.assertEqual(seen, ['prepare', 'post', 'encode'])
        vae.upsampling_set = None
        timer.observe_progress('task', epoch, 'VAE Decoding', stage_id='decode')
        vae.decode('normal decode after disabling upscaling')
        self.assertEqual(seen[-1], 'decode')
        self.assertIsNone(timer.snapshot()['enhancement'])

    def test_native_controls_support_sequential_jobs_and_reused_task_ids(self):
        native = types.SimpleNamespace(control=None)
        native.current_control = lambda: native.control
        def processor(*args, **kwargs): return timer.snapshot()
        scope, timer, epoch, owner, modules, imports = self.setup_owner({
            'shared.utils.media_control': native.__dict__,
            'postprocessing.spatial_upsamplers': {'upscale_postprocessing': processor}}, context=False)
        api = modules['postprocessing.spatial_upsamplers']
        with patch.object(scope['importlib'], 'import_module', side_effect=imports):
            for task_id in ('task', 'second', 'second'):
                native.control = types.SimpleNamespace(gen={'api_active_queue_task':{'id':task_id}})
                snap = api.upscale_postprocessing(spatial_upsampling='any method')
                self.assertEqual(snap['task_id'], task_id)
                self.assertEqual(snap['last_stage'], 'post')
                timer.finish_task(task_id)
                self.assertEqual(scope['_processor_execution'](owner)[1], snap['execution_epoch'])
                stale = api.upscale_postprocessing(spatial_upsampling='any method')
                self.assertIsNone(stale['enhancement'])  # Same completed control cannot restart timing.
        self.assertGreater(timer.snapshot()['execution_epoch'], epoch)

    def test_unknown_phase_is_owned_only_inside_enhancement_scope(self):
        scope = helpers()
        timer = scope['_StageTimingTelemetry']()
        epoch = timer.start_task('task')
        self.assertFalse(timer.observe_progress('task', epoch, 'Completely new process', stage_id=None))
        timer.begin_enhancement('task', epoch, 'Enhancing')
        self.assertTrue(timer.observe_progress('task', epoch, 'Completely new process', 3, 8, 'tiles', stage_id=None))
        self.assertEqual(timer.snapshot()['progress']['stage'], 'post')
        timer.end_enhancement('task', epoch)
        self.assertFalse(timer.observe_progress('task', epoch, 'Other unknown phase', stage_id=None))

    def test_noop_methods_and_zero_grain_do_not_create_enhance(self):
        def noop(*args, **kwargs): return 'original'
        def grain(images, grain_intensity=0, saturation=0.5): return images
        scope, timer, epoch, owner, modules, imports = self.setup_owner({
            'postprocessing.temporal_upsamplers': {'temporal_upsample':noop},
            'postprocessing.film_grain': {'add_film_grain':grain}})
        self.assertEqual(modules['postprocessing.temporal_upsamplers'].temporal_upsample(temporal_upsampling=''), 'original')
        self.assertEqual(modules['postprocessing.film_grain'].add_film_grain('pixels'), 'pixels')
        self.assertNotIn('post', timer.snapshot()['stages'])
        self.assertEqual(modules['postprocessing.film_grain'].add_film_grain('pixels', 0.5), 'pixels')
        self.assertEqual(timer.snapshot()['last_stage'], 'post')

    @unittest.skipUnless(shutil.which('node'), 'Node.js unavailable')
    def test_browser_uses_owned_unknown_activity_instead_of_stale_decode(self):
        js = _javascript_with_exports('freshState','readLiveSnapshot')
        script = r"""
const api = globalThis.__statusProReleaseTest;
const ns = {state:api.freshState(), activeRun:{queue_task_id:'task',_stageTimingEpoch:1,settings:{}},
 runTelemetry:{execution_task_known:true,executing_task:{id:'task'},native_progress:{phase:'VAE Decoding',current:15,total:15,unit:'tiles'},
 stage_timing:{task_id:'task',execution_epoch:1,stages:{post:{active:true}},
 enhancement:{stage:'post',phase:'Future processor conditioning',current:null,total:null,unit:null}}}};
const result = api.readLiveSnapshot(ns);
if(result.id!=='post' || result.rawName!=='Future processor conditioning' || result.progress!==null || result.steps.current!==null)
 throw new Error('Unknown enhancement phase inherited Decode or its tile counter');
"""
        result = subprocess.run([shutil.which('node'),'-'], input=js+script, text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

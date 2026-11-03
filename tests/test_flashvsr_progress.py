"""FlashVSR progress describes processor passes, not generation steps or tiles."""
import ast
import shutil
import subprocess
import unittest
from unittest.mock import patch

from test_enhancement_stages import helpers
import test_enhancement_coverage
from test_release_smoke import _javascript_with_exports, _source


class FlashVSRProgressTests(unittest.TestCase):
    def test_bridge_load_and_two_pass_upscale_retain_actual_endpoints(self):
        clock = [0.0]
        class Bridge:
            def load_upsampler(self, spatial_upsampling, progress_callback=None, **kwargs):
                progress_callback('Caching')
                clock[0] += 20
            def upscale(self, sample, spatial_upsampling, progress_callback=None, **kwargs):
                for _ in range(2):
                    progress_callback('Denoising', 0, 2)
                    for chunk in range(2):
                        clock[0] += 2
                        progress_callback('Denoising', chunk + 1, 2)
                        progress_callback('TCDecoder Decoding', chunk, 2)
                        clock[0] += 1
                        progress_callback('TCDecoder Decoding', chunk + 1, 2)
                    progress_callback('Color Correction', 0, 1, 'frames')
                    clock[0] += 1
                    progress_callback('Color Correction', 1, 1, 'frames')
                return sample
        setup = test_enhancement_coverage.EnhancementCoverageTests()
        scope, timer, epoch, owner, modules, imports = setup.setup_owner({
            'postprocessing.flashvsr.wgp_bridge': {'FlashVSRBridge': Bridge}})
        forwarded = []
        def callback(phase, current=None, total=None, unit=None):
            forwarded.append((phase, current, total, unit))
        with patch('time.perf_counter', side_effect=lambda: clock[0]):
            bridge = Bridge()
            bridge.load_upsampler('flashvsr2pass*2', progress_callback=callback)
            self.assertEqual(bridge.upscale('pixels', 'flashvsr2pass*2', progress_callback=callback), 'pixels')
        performance = owner._latest_performance
        activities = performance['activities']
        self.assertEqual(len(activities), 7)  # Loading, then three activities per pass.
        self.assertEqual(activities[0]['label'], 'FlashVSR - Loading models')
        self.assertEqual(activities[0]['elapsed'], 20)
        for part in range(2):
            upscale, decode, colour = activities[1 + part * 3:4 + part * 3]
            self.assertIn(f'Pass {part + 1}', upscale['label'])
            for activity in (upscale, decode):
                self.assertEqual((activity['current'], activity['total'], activity['unit']), (2, 2, 'chunks'))
                self.assertEqual(activity['state'], 'complete')
            self.assertEqual((upscale['elapsed'], decode['elapsed']), (4, 2))
            self.assertEqual((colour['current'], colour['total'], colour['unit']), (1, 1, 'frames'))
        self.assertEqual(performance['steps'], [])
        self.assertEqual(timer.snapshot()['last_stage'], 'post')
        self.assertNotIn('denoise', timer.snapshot()['stages'])
        self.assertTrue(all(label.startswith('FlashVSR') for label, *_ in forwarded))

    def test_active_snapshots_and_early_exit_preserve_reported_work(self):
        scope = helpers()
        snapshot_scope = dict(scope)
        node = next(n for n in ast.walk(ast.parse(_source()))
                    if isinstance(n, ast.FunctionDef) and n.name == '_performance_snapshot')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'snapshot-test', 'exec'), snapshot_scope)
        performance = scope['_new_performance_observer']('task')
        clock = [10.0]
        with patch('time.perf_counter', side_effect=lambda: clock[0]):
            callback = scope['_observe_flashvsr_progress'](performance, None)
            callback('Denoising', 0, 6)
            clock[0] = 14.0
            live = snapshot_scope['_performance_snapshot']({'status_pro_performance': performance})
            self.assertEqual(live['activities'][0]['elapsed'], 4)
            self.assertEqual(performance['activities'][0]['elapsed'], 0)  # Snapshot is read-only.
            callback('Denoising', 2, 6)
            callback.finish()
        activity = performance['activities'][0]
        self.assertEqual((activity['current'], activity['total']), (2, 6))
        self.assertLess(activity['progress'], 100)

    def test_standalone_upscale_plans_only_the_work_it_uses(self):
        scope = {}
        node = next(n for n in ast.walk(ast.parse(_source()))
                    if isinstance(n, ast.FunctionDef) and n.name == 'plan_stages_for_task')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'plan-test', 'exec'), scope)
        plan = scope['plan_stages_for_task']
        task = {'params': {'mode': 'edit_postprocessing', 'model_type': 'minimax_h3_t2v',
                           'spatial_upsampling': 'flashvsr2pass*2', 'image_refs': ['source.jpg']}}
        self.assertEqual(plan(task), ['prepare', 'post', 'save'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js unavailable')
    def test_retained_activities_reach_live_details_and_history_without_false_generate(self):
        exports = ['freshState', 'observePerformanceTelemetry', 'applyServerStageTiming',
                   'stageActivities', 'recoveredPerformanceGroups']
        if 'function stageDurations(state)' in _source(): exports.append('stageDurations')
        script = r"""
const api=globalThis.__statusProReleaseTest;
const run={queue_task_id:'task',started_at:1000000,settings:{mode:'edit_postprocessing'}};
const activity={id:'pass1',label:'FlashVSR - Pass 1 - Upscaling',stage:'post',
 started_at:1001,state:'complete',elapsed:4,current:1,total:1,unit:'chunks',progress:100};
const telemetry={performance:{id:'flash',task_id:'task',activities:[activity],steps:[
 {sequence:1,phase:0,step:1,total_steps:1,stage:'post',label:'Denoising',duration_seconds:4,completed_at:1005}]},
 stage_timing:{task_id:'task',execution_epoch:1,last_stage:'post',stages:{post:{elapsed:5,completed:true}}}};
api.observePerformanceTelemetry(run,telemetry);
api.observePerformanceTelemetry(run,telemetry);
if(run.processor_activities.length!==1) throw new Error('Repeated polls duplicated activity');
if(api.recoveredPerformanceGroups(run)[0].stage!=='post') throw new Error('Processor denoising recovered as Generate');
const ns={activeRun:run,state:api.freshState()};
api.applyServerStageTiming(ns,telemetry);
const rows=api.stageActivities(ns.state,ns.state.records.post);
if(rows.length!==1 || rows[0].current!==1 || rows[0].total!==1 || rows[0].unit!=='chunks')
 throw new Error('Actual chunk endpoint missing from live details');
if(api.stageDurations) {
 const phases=Object.values(api.stageDurations(ns.state)).filter(p=>p.raw_label===activity.label);
 if(phases.length!==1 || phases[0].current!==1 || phases[0].duration_seconds!==4)
  throw new Error('History lost retained completed work');
}
"""
        result=subprocess.run([shutil.which('node'), '-'], input=_javascript_with_exports(*exports)+script,
                              text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

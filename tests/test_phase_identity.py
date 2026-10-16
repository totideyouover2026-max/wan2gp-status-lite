"""Activity identity must describe work, independent of changing native metrics."""
import json
import shutil
import subprocess
import unittest

from test_enhancement_stages import helpers
from test_release_smoke import _javascript_with_exports, _source


@unittest.skipUnless(shutil.which('node'), 'Node.js unavailable')
class PhaseIdentityTests(unittest.TestCase):
    def run_js(self, exports, script):
        result = subprocess.run([shutil.which('node'), '-'], input=_javascript_with_exports(*exports)+script,
                                text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_voice_stream_is_one_advancing_activity_and_retains_observed_frames(self):
        scope = helpers()
        timer = scope['_StageTimingTelemetry']()
        epoch = timer.start_task('task', new_execution=True)
        events = []
        labels = [('Generating Speech', None, None, None)]
        labels += [(f'Sample 1/2 - Denoising | Generating Speech - {count * 0.05:.1f}s | {count/3:.1f}s / 200s',
                    count, 430, 'frames') for count in range(1, 191)]
        labels += [('Decoding Speech', 1, 4, 'tiles')]
        for label, current, total, unit in labels:
            timer.observe_progress('task', epoch, label, current, total, unit, stage_id=scope['_structured_stage_id'](label))
            events.append(timer.snapshot())
        self.run_js(('freshState','readLiveSnapshot','applySnapshot','stageActivities','formatCounter'), r"""
const api=globalThis.__statusProReleaseTest;
const ns={state:api.freshState(),activeRun:{queue_task_id:'task',_stageTimingEpoch:1,settings:{}},
 runTelemetry:{execution_task_known:true,executing_task:{id:'task'},native_progress:{phase:'VAE Decoding',current:15,total:15,unit:'tiles'}}};
let now=1000000;
Date.now=()=>now;
const events="""+json.dumps(events)+r""";
for (let i=0;i<events.length;i++) {
 now+=333;
 ns.runTelemetry.stage_timing=events[i];
 const snapshot=api.readLiveSnapshot(ns);
 api.applySnapshot(ns,snapshot);
 if (i<events.length-1 && api.stageActivities(ns.state,ns.state.records.denoise).length!==1)
  throw new Error('Metric update created another activity at '+i);
}
const rows=api.stageActivities(ns.state,ns.state.records.denoise);
if(rows.length!==1 || rows[0].label!=='Generating Speech') throw new Error('Standalone/generic callback handoff duplicated speech');
if(rows[0].current!==190 || rows[0].total!==430 || rows[0].unit!=='frames' || rows[0].progress>=100)
 throw new Error('Speech completion fabricated its maximum counter');
if(api.formatCounter(rows[0])!=='190/430 frames') throw new Error('Frames displayed as steps');
if(rows[0].elapsed<60 || rows[0].state!=='complete') throw new Error('Activity timing restarted with each update');
if(ns.state.currentId!=='decode') throw new Error('Actual waveform decoding did not reach Decode');
""")

    def test_dynamic_counters_are_metadata_but_real_parts_and_passes_remain_distinct(self):
        self.run_js(('phaseInfo','normalizedPhaseLabel'), r"""
const api=globalThis.__statusProReleaseTest;
const check=(v,m)=>{if(!v)throw new Error(m);};
for(const [base,unit] of [['Denoising | Compute Audio Codes','tokens'],['Denoising','steps'],['Denoising | Audio samples','seconds']]) {
 const keys=[];
 for(let n=1;n<20;n++) {
  const suffix=unit==='seconds'?'s':unit;
  keys.push(api.phaseInfo(`${base} ${n}/430 ${suffix}`,{current:n,total:430,unit}).id);
 }
 check(new Set(keys).size===1,base+' counter became identity');
}
const layerKeys=[];
for(let n=1;n<=35;n++) layerKeys.push(api.phaseInfo(`Upscaling H3 latent between phases (${n}/35)`,{current:n,total:35,unit:'layers'}).id);
check(new Set(layerKeys).size===1,'parenthesised metric became activity identity');
check(api.phaseInfo('Generating Speech').id===api.phaseInfo('Denoising | Generating Speech - 3.1s').id,'generic wrapper changed speech identity');
check(api.phaseInfo('Generating Audio').id===api.phaseInfo('Denoising | Generating Audio | 1m 20s / 2m 30s').id,'formatted native time changed audio identity');
for(const prefix of ['Part','Segment','Window','Chunk','Prompt','Sample','Speaker','Reference','Phase','Pass','Voice','Track','Sentence','Utterance','View','Take','Shot','Scene']) {
 const a=api.phaseInfo(`Generating Speech - ${prefix} 1/2 - 0.2s`,{current:1,total:2,unit:'segments'});
 const b=api.phaseInfo(`Generating Speech - ${prefix} 2/2 - 0.4s`,{current:2,total:2,unit:'segments'});
 check(a.id!==b.id,prefix+' identities collapsed');
}
for(const label of ['Phase 2','Encoding Text Prompt 1/2','Denoising Second Phase | Phase 2/2 High Resolution','Loading model v2.1','Generating Speech - Speaker 2','Resolution 768x1344'])
 check(api.normalizedPhaseLabel(label,{current:1,total:2,unit:'steps'})===label,'Semantic label damaged: '+label);
check(api.phaseInfo('Denoising First Phase | Generating Speech').id!==api.phaseInfo('Denoising Second Phase | Generating Speech').id,'real passes collapsed');
""")

    def test_repeated_real_activity_after_another_phase_stays_separate(self):
        self.run_js(('freshState','applySnapshot','stageActivities','finishPhase'), r"""
const api=globalThis.__statusProReleaseTest;
const ns={state:api.freshState(),activeRun:{settings:{}}};
let now=1000000; Date.now=()=>now;
function event(id,label,current,total,unit) {
 now+=1000;
 api.applySnapshot(ns,{id,rawName:label,rawMessage:label,steps:{current,total,unit},progress:current/total*100,
  progressScope:'phase',transitionEvidence:'structured'});
}
event('denoise','Generating Speech - Part 1/2 - 0.1s',2,430,'frames');
event('denoise','Denoising | Generating Speech - Part 1/2 - 0.2s',4,430,'frames');
event('encode','Encoding Text Prompt 2/2',7,12,'layers');
event('denoise','Generating Speech - Part 2/2',8,430,'frames');
event('denoise','Denoising | Generating Speech - Part 2/2 - 0.4s',10,430,'frames');
api.finishPhase(ns.state);
const rows=api.stageActivities(ns.state,ns.state.records.denoise);
if(rows.length!==2 || rows[0].current!==4 || rows[1].current!==10) throw new Error('Parts duplicated, merged, or fabricated completion');
const encode=api.stageActivities(ns.state,ns.state.records.encode);
if(encode.length!==1 || encode[0].current!==7) throw new Error('Closing an activity fabricated encoder completion');
// Returning to the exact same named activity after another stage is a new occurrence.
event('decode','Decoding Speech',1,4,'tiles');
event('denoise','Generating Speech - Part 2/2 - 1.2s',25,430,'frames');
if(api.stageActivities(ns.state,ns.state.records.denoise).length!==3) throw new Error('Repeated real activity was suppressed');
""")

    def test_pro_history_records_one_semantic_phase_with_real_counters(self):
        if 'class StatusProPlugin' not in _source(): self.skipTest('Pro history only')
        self.run_js(('freshState','applySnapshot','stageDurations','finishPhase'), r"""
const api=globalThis.__statusProReleaseTest;
const ns={state:api.freshState(),activeRun:{settings:{}}};
let now=1000000; Date.now=()=>now;
for(let n=1;n<=100;n++) {
 now+=300;
 api.applySnapshot(ns,{id:'denoise',rawName:`Denoising | Generating Speech - ${n/10}s`,steps:{current:n,total:430,unit:'frames'},
  progress:n/430*100,progressScope:'phase',transitionEvidence:'structured'});
}
api.finishPhase(ns.state);
const phases=Object.values(api.stageDurations(ns.state)).filter(p=>p.stage==='denoise' && !p.authoritative);
if(phases.length!==1 || phases[0].current!==100 || phases[0].total!==430 || phases[0].duration_seconds<29)
 throw new Error('History stored per-update rows or invented counters');
if(/\d+s/.test(phases[0].label)) throw new Error('Changing audio duration persisted in activity label');
""")

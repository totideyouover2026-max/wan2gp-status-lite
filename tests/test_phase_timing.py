"""Phase-local estimates must not reuse cumulative stage durations."""
import shutil
import subprocess
import unittest
from test_release_smoke import _javascript_with_exports


class PhaseTimingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_phase_local_average_eta_and_cumulative_stage_duration(self):
        js = _javascript_with_exports("freshState", "applySnapshot", "applyServerStageTiming")
        script = r"""
const api=globalThis.__statusProReleaseTest;
const check=(v,m)=>{if(!v)throw new Error(m);};
let now=1000000;Date.now=()=>now;
globalThis.performance={now:()=>now};
function ns(){return {state:api.freshState(),activeRun:{queue_task_id:'task',settings:{}}};}
function server(n,id,elapsed,active,runCount=1){
 api.applyServerStageTiming(n,{stage_timing:{task_id:'task',execution_epoch:1,last_stage:id,
  stages:{[id]:{elapsed,active_elapsed:active,active:true,run_count:runCount}}}});
}
function snap(n,id,name,current,total){
 api.applySnapshot(n,{id,rawName:name,rawMessage:name,progress:current/total*100,
  progressScope:'phase',steps:{current,total,unit:'steps'},transitionEvidence:'structured'});
}
// H3 returns to Generate after decode, upscale and encode. A hidden tab can
// first see phase two after several steps, with no counter reset rendered.
const h3=ns();server(h3,'denoise',80,80);snap(h3,'denoise','Denoising First Phase',8,8);
server(h3,'decode',10,10);snap(h3,'decode','VAE Decoding',1,1);
server(h3,'denoise',100,20,2);snap(h3,'denoise','Phase 2/2 Tiled',2,12);
let r=h3.state.records.denoise;
check(r.elapsed===100,'combined Generate time was lost');
check(r.stepSeconds===10,'phase one inflated phase two average: '+r.stepSeconds);
check(r.eta===100,'phase one inflated phase two ETA: '+r.eta);
now+=10000;server(h3,'denoise',110,30,2);snap(h3,'denoise','Phase 2/2 Tiled',3,12);
check(r.stepSeconds===10&&r.eta===90,'phase baseline changed during phase two');
server(h3,'denoise',200,120,2);snap(h3,'denoise','Phase 2/2 Tiled',12,12);
check(r.stepSeconds===10&&r.eta===0,'last step did not update phase average');
// Adjacent denoising passes can share one uninterrupted server stage.
const adjacent=ns();server(adjacent,'denoise',80,80);snap(adjacent,'denoise','Denoising First Phase',8,8);
now+=1000;server(adjacent,'denoise',80,80);snap(adjacent,'denoise','Denoising Second Phase',0,4);
now+=10000;server(adjacent,'denoise',90,90);snap(adjacent,'denoise','Denoising Second Phase',1,4);
r=adjacent.state.records.denoise;
check(r.elapsed===90&&r.stepSeconds===10&&r.eta===30,'adjacent phases reused prior elapsed time');
// Counter-only restarts must also isolate timing, including legacy DOM mode.
const legacy=ns();snap(legacy,'denoise','Denoising',0,4);
now+=40000;snap(legacy,'denoise','Denoising',4,4);
snap(legacy,'denoise','Denoising',0,4);
now+=5000;snap(legacy,'denoise','Denoising',1,4);
r=legacy.state.records.denoise;
check(r.stepSeconds===5&&r.eta===15,'counter reset retained previous phase time');
// Enhancement ETA also uses the current visit, preserving combined Enhance time.
const post=ns();server(post,'post',80,80);snap(post,'post','Upsampling - Denoising',8,8);
server(post,'decode',10,10);snap(post,'decode','VAE Decoding',1,1);
server(post,'post',100,20,2);snap(post,'post','Upsampling - Denoising',2,4);
r=post.state.records.post;
check(r.elapsed===100&&r.eta===20,'enhancement ETA reused previous visit');
"""
        result = subprocess.run([shutil.which("node"), "-"], input=js+script,
                                text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

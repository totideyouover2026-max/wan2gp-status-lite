"""Worker timing remains authoritative when the browser runs no timers."""
import copy
import json
import shutil
import subprocess
import types
import unittest
from unittest.mock import patch

from test_enhancement_stages import helpers
from test_v13_compatibility import python_helpers
from test_release_smoke import _source, _javascript_with_exports


class BackgroundTimingTests(unittest.TestCase):
    def test_window_and_sample_timings_survive_without_polling(self):
        scope = helpers()
        with patch("time.time", return_value=1000):
            timer = scope["_StageTimingTelemetry"]()
            epoch = timer.start_task("task", now=0)
        timer.observe_sample("task", 1, now=0, execution_epoch=epoch)
        timer.observe_window("task", 1, now=0, execution_epoch=epoch)
        for stage, when in (("encode", 2), ("denoise", 7), ("decode", 17), ("save", 22)):
            timer.observe_stage("task", stage, now=when, execution_epoch=epoch)
        timer.observe_window("task", 2, now=25, execution_epoch=epoch)
        timer.observe_stage("task", "denoise", now=28, execution_epoch=epoch)
        timer.observe_stage("task", "save", now=38, execution_epoch=epoch)
        timer.finish_task("task", now=40, completed=True)
        saved = timer.snapshot(now=200)
        self.assertEqual(saved["completed_at"], 1040)
        self.assertEqual([s["started_at"] for s in saved["segments"]], [1000, 1025])
        first = saved["segments"][0]
        self.assertEqual({k:v["elapsed"] for k,v in first["stages"].items()},
                         {"prepare":2, "encode":5, "denoise":10, "decode":5, "save":3})
        timer.finish_task("task", now=300, completed=True)
        self.assertEqual(timer.snapshot(now=300)["completed_at"], 1040)
        timer.start_task("next", now=301)
        self.assertEqual(saved["segments"][0]["stages"]["denoise"]["elapsed"], 10)

    def test_outcomes_keep_timing_by_browser_scope_and_do_not_retag_outputs(self):
        telemetry = python_helpers("_TaskOutcomeTelemetry")["_TaskOutcomeTelemetry"]()
        telemetry.begin("a", 1, task={"id":"a","settings":{}}, scope_id=10)
        telemetry.capture_outputs("a", 1, [{"path":"first.mp4"}], sample_no=1, window_no=1)
        telemetry.capture_outputs("a", 1, [{"path":"first.mp4"},{"path":"second.mp4"}], sample_no=1, window_no=2)
        timing={"task_id":"a","execution_epoch":1,"started_at":1000,"completed_at":1040,"stages":{}}
        telemetry.finish("a",1,True,stage_timing=timing)
        telemetry.begin("b",2,task={"id":"b"},scope_id=20)
        self.assertEqual([s["task_id"] for s in telemetry.snapshot(scope_id=10)], ["a"])
        outputs=telemetry.snapshot(scope_id=10)[0]["output_records"]
        self.assertEqual([s["window_no"] for s in outputs], [1,2])
        snapshot=telemetry.snapshot(scope_id=10)
        snapshot[0]["stage_timing"]["stages"]["prepare"]={"elapsed":999}
        self.assertEqual(telemetry.snapshot(scope_id=10)[0]["stage_timing"]["stages"], {})

    def test_worker_finishes_with_timings_without_any_ui_poll(self):
        scope=helpers()
        scope["_latest_performance_snapshot"]=lambda *args: None
        scope["_task_telemetry"]=lambda task, **kwargs: {"id":task["id"],"settings":task.get("params",{})}
        owner=types.SimpleNamespace(_stage_timing=scope["_StageTimingTelemetry"](),
            _task_outcomes=python_helpers("_TaskOutcomeTelemetry")["_TaskOutcomeTelemetry"](),
            _generation_timing_observer_installed=False)
        clock=[0]
        def body(task,send_cmd,state=None):
            for title,when in (("Encoding Text Prompt",2),("Denoising",7),("VAE Decoding",17),("Saving output",22)):
                clock[0]=when;send_cmd("status",title)
            clock[0]=25
            return True
        owner.generate_media=body
        owner.set_global=lambda name,value:setattr(owner,name,value)
        scope["_install_generation_timing_observer"](owner)
        state={"gen":{}}
        with patch("time.time",side_effect=lambda:1000+clock[0]),patch("time.monotonic",side_effect=lambda:clock[0]):
            owner.generate_media({"id":"a","params":{}},lambda *args:None,state=state)
        saved=owner._task_outcomes.snapshot(scope_id=id(state["gen"]))[0]
        self.assertTrue(saved["known"])
        self.assertEqual(saved["stage_timing"]["stages"]["denoise"]["elapsed"],10)
        self.assertEqual(saved["stage_timing"]["completed_at"],1025)
        self.assertEqual(saved["task"]["id"],"a")

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_browser_resumes_finished_unseen_and_partially_seen_runs(self):
        pro = 'class StatusProPlugin' in _source()
        exports=["freshState","startRun","syncRunTelemetry","recoverBackgroundExecutions","stageTimingForRun"]
        if pro:exports.append("aggregateStageDurations")
        js=_javascript_with_exports(*exports)
        script=r"""
const api=globalThis.__statusProReleaseTest;
const storage={getItem:()=>null,setItem(){},removeItem(){}};
globalThis.window={localStorage:storage,sessionStorage:storage};
globalThis.document={querySelector:()=>null,querySelectorAll:()=>[]};
function check(value,message){if(!value)throw new Error(message);}
function ns(){return {state:api.freshState(),source:{querySelector:()=>null,querySelectorAll:()=>[]},
    container:{querySelector:()=>null},download:{active:false},sessionId:'test',sessionRunIds:new Set(),
    runHistory:[],historyBaseline:[],historyPersistence:'browser',selectedRunIds:new Set(),
    recoverablePrompts:new Map(),openHistoryRuns:new Set(),openHistoryGroups:new Set(),
    visibleHistoryGroups:new Set(),historyRecording:true,lastExecutingTaskKey:'',lastExecutionProgressSignature:''};}
function timing(id,epoch,start=1000){return {task_id:id,execution_epoch:epoch,started_at:start,completed_at:start+25,
    sample_no:1,window_no:1,last_stage:'save',stages:{prepare:{elapsed:2,completed:true,active:false,run_count:1},
    encode:{elapsed:5,completed:true,active:false,run_count:1},denoise:{elapsed:10,completed:true,active:false,run_count:1},
    decode:{elapsed:5,completed:true,active:false,run_count:1},save:{elapsed:3,completed:true,active:false,run_count:1}}};}
function outcome(id,epoch,start=1000){const t=timing(id,epoch,start);return {task_id:id,execution_epoch:epoch,
    task:{id,settings:{model_type:'flux'},planned_stages:['prepare','encode','denoise','decode','save']},
    known:true,success:true,started_at:start,completed_at:start+25,stage_timing:t,
    output_records:[{path:id+'.png',sample_no:1,window_no:1,media_type:'image',settings:{}}]};}
const completed=outcome('a',1);
const telemetry={server_time:2000,in_progress:false,execution_task_known:true,executing_task:null,
    active_task:null,stage_timing:completed.stage_timing,task_outcomes:[completed],output_records:completed.output_records};
// No browser ticks at all during the task.
const unseen=ns();unseen.runTelemetry=telemetry;
api.syncRunTelemetry(unseen);
check(!unseen.activeRun && unseen.state.records.denoise.elapsed===10,'hidden work was charged to Prepare');
check(unseen.state.overallElapsed===25,'focus delay extended completed run');
if(PRO){check(unseen.runHistory.length===1,'unseen completed task missing from history');
    check(api.aggregateStageDurations(unseen.runHistory[0].stages).prepare===2,'Prepare duration fabricated');}
api.syncRunTelemetry(unseen);
if(PRO)check(unseen.runHistory.length===1,'completed task replayed twice');
// Prepare was seen, but none of the later phases were rendered.
const seen=ns();seen.runTelemetry={...telemetry,server_time:1000,in_progress:true,executing_task:completed.task,
    active_task:completed.task,task_outcomes:[],stage_timing:{task_id:'a',execution_epoch:1,started_at:1000,
        stages:{prepare:{elapsed:0,active:true}}}};
api.syncRunTelemetry(seen);
seen.runTelemetry={...telemetry,executing_task:completed.task,active_task:completed.task};
api.syncRunTelemetry(seen);
check(seen.state.records.denoise.elapsed===10 && seen.state.records.prepare.elapsed===2,'terminal outcome skipped recorded stage timings');
check(seen.state.overallElapsed===25,'seen task completion used focus time');
// Archived timing for A survives B replacing the current snapshot.
const next=outcome('b',2,1030);
const queued=ns();queued.runTelemetry={...telemetry,task_outcomes:[completed,next],stage_timing:next.stage_timing};
api.syncRunTelemetry(queued);
if(PRO)check(queued.runHistory.length===2,'hidden queue tasks were lost');
// A newly loaded page must not resurrect older cleared history.
const old=outcome('old',3,900);const cleared=ns();
api.recoverBackgroundExecutions(cleared,{...telemetry,task_outcomes:[old]});
check(cleared.activeRun===undefined && cleared.runHistory.length===0,'old completed task resurrected');
// Two windows retain independent durations and outputs, even if only Prepare was seen.
if(PRO){
 const two=outcome('two',4);const first=timing('two',4,1000);const second=timing('two',4,1025);second.window_no=2;
 two.stage_timing={...second,segments:[first,second]};two.completed_at=1050;
 two.output_records=[{path:'one.mp4',sample_no:1,window_no:1,settings:{}},{path:'two.mp4',sample_no:1,window_no:2,settings:{}}];
 const partial=ns();partial.runTelemetry={...telemetry,server_time:1000,in_progress:true,executing_task:two.task,
     active_task:two.task,task_outcomes:[],sliding_window:true,window_no:1,total_windows:2,
     stage_timing:{task_id:'two',execution_epoch:4,started_at:1000,stages:{prepare:{elapsed:0,active:true}}}};
 api.syncRunTelemetry(partial);
 partial.runTelemetry={...telemetry,executing_task:two.task,active_task:two.task,task_outcomes:[two],
     sliding_window:true,window_no:2,total_windows:2,stage_timing:two.stage_timing,output_records:two.output_records};
 api.syncRunTelemetry(partial);
 check(partial.runHistory.length===2,'partially observed windows missing');
 check(partial.runHistory.every(run=>run.duration_seconds===25),'window timings mixed');
 check(partial.runHistory.find(run=>run.window_no===1).outputs.join()==='one.mp4','second output leaked into first window');
}
""".replace('PRO', 'true' if pro else 'false')
        result=subprocess.run([shutil.which('node'),'-'],input='Date.now=()=>1000000;\n'+js+script,
                              text=True,encoding='utf-8',capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

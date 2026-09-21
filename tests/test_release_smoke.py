import ast
import importlib.util
import inspect
import json
import pathlib
import shutil
import subprocess
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN_PATH = ROOT / "plugin.py"
DOWNLOAD_PATH = ROOT / "download_telemetry.py"


def _download_module():
    spec = importlib.util.spec_from_file_location("status_lite_download_release_test", DOWNLOAD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _source() -> str:
    return PLUGIN_PATH.read_text(encoding="utf-8")


def _returned_string(function_name: str) -> str:
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != function_name:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Constant):
                if isinstance(child.value.value, str):
                    return child.value.value
    raise AssertionError(f"No constant string return found for {function_name}")


def _javascript_with_exports(*names: str) -> str:
    marker = "    boot();\n})();"
    javascript = _returned_string("_javascript")
    if marker not in javascript:
        raise AssertionError("Embedded JavaScript boot marker changed")
    exports = ", ".join(names)
    return javascript.replace(
        marker,
        f"    globalThis.__statusProReleaseTest = {{ {exports} }};\n}})();",
    )


class LiteReleaseSmokeTests(unittest.TestCase):
    def test_manifest_and_python_identity_agree(self):
        source = _source()
        download_source = DOWNLOAD_PATH.read_text(encoding="utf-8")
        compile(source, str(PLUGIN_PATH), "exec")
        compile(download_source, str(DOWNLOAD_PATH), "exec")
        manifest = json.loads((ROOT / "plugin_info.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], "Status Lite")
        self.assertEqual(manifest["version"], "1.1.1")
        self.assertEqual(manifest["type"], "extension")
        self.assertEqual(manifest["wan2gp_version"], "0")
        self.assertIn('class StatusLitePlugin(WAN2GPPlugin):', source)
        self.assertIn('self.name = "Status Lite"', source)
        self.assertIn('self.version = "1.1.1"', source)
        self.assertNotIn("Status Pro", download_source)
        self.assertIn("Status Lite", download_source)

    def test_markup_is_live_only(self):
        markup = _returned_string("_markup")
        for token in (
            "data-status-lite",
            "data-sp-stages",
            "data-sp-downloads",
            "data-sp-detail",
            "data-sp-detail-activities",
            "data-sp-collapse",
        ):
            self.assertIn(token, markup)
        for forbidden in (
            "history",
            "data-sp-export",
            "data-sp-import",
            "data-sp-settings-button",
            "dialog",
        ):
            self.assertNotIn(forbidden, markup.lower())

    def test_javascript_has_no_run_history_path(self):
        javascript = _returned_string("_javascript")
        for forbidden in (
            "runHistory",
            "sessionStorage",
            "data-sp-history",
            "data-sp-export",
            "data-sp-import",
            "persistRunHistory",
            "loadRunHistory",
            "exportHistory",
            "requestGalleryNavigation",
        ):
            self.assertNotIn(forbidden, javascript)
        self.assertEqual(javascript.count("window.localStorage.getItem"), 1)
        self.assertEqual(javascript.count("window.localStorage.setItem"), 1)
        self.assertIn("COLLAPSED_KEY", javascript)

    def test_live_recovery_and_subwindow_features_remain(self):
        javascript = _returned_string("_javascript")
        for token in (
            "function recoverMissedPerformanceStages",
            "function recoveredPerformanceGroups",
            "function observePerformanceTelemetry",
            "function stageModelInfo",
            "function windowDetails",
            "document.addEventListener(\"visibilitychange\"",
            "window.addEventListener(\"focus\"",
            "namespace.activeRun = null",
        ):
            self.assertIn(token, javascript)
        self.assertNotIn("splitMissedSlidingWindows", javascript)
        stage_classifier = javascript[javascript.index("function stageIdFor"):javascript.index("function phaseInfo")]
        self.assertIn("distilled refinement", stage_classifier)
        self.assertIn('const resumableStage = ["input", "post"].includes(snapshot.id)', javascript)

    def test_status_pro_takes_precedence_without_duplicate_observers(self):
        source = _source()
        javascript = _returned_string("_javascript")
        self.assertIn('import builtins', source)
        self.assertIn('_register_status_variant("lite")', source)
        post_setup = source[source.index("    def post_ui_setup"):source.index("    @staticmethod", source.index("    def post_ui_setup"))]
        self.assertLess(post_setup.index("_status_pro_registered()"), post_setup.index("install_download_observer()"))
        self.assertLess(post_setup.index("_status_pro_registered()"), post_setup.index("self._install_step_observer()"))
        self.assertIn('function disableForStatusPro(root)', javascript)
        self.assertIn('root.querySelector("#status-pro-container")', javascript)
        self.assertIn('function nativeStatusSource(root, container)', javascript)
        self.assertIn('root.querySelector("#gen_status")', javascript)
        self.assertNotIn('const source = container.previousElementSibling', javascript)

    def test_backend_has_only_live_bridges(self):
        source = _source()
        self.assertIn('elem_id="status-lite-download-bridge"', source)
        self.assertIn('elem_id="status-lite-run-bridge"', source)
        for forbidden in (
            "gallery_request_bridge",
            "gallery_result_bridge",
            "navigate_to_history_output",
            'self.request_component("gallery_tabs")',
            'self.request_global("get_settings_from_file")',
        ):
            self.assertNotIn(forbidden, source)

    def test_download_wrapper_forwards_current_and_future_arguments(self):
        module = _download_module()
        calls = []
        marker = object()

        def download_file(url, filename, gen=None, show_filename=True, future_option=None):
            calls.append((url, filename, gen, show_filename, future_option))
            if future_option == "fail":
                raise RuntimeError("download failed")
            return filename

        download = types.ModuleType("shared.utils.download")
        download.download_file = download_file
        download.process_files_def = lambda *args, **kwargs: None
        download.create_progress_hook = lambda filename: lambda *args: None
        download.download_def_missing_files = lambda definition: []
        utils = types.ModuleType("shared.utils")
        utils.download = download
        shared = types.ModuleType("shared")
        shared.utils = utils
        names = ("shared", "shared.utils", "shared.utils.download")
        previous = {name: sys.modules.get(name) for name in names}
        sys.modules.update(dict(zip(names, (shared, utils, download))))
        try:
            telemetry = module.DownloadTelemetry()
            observer = module.DownloadObserver(telemetry)
            self.assertTrue(observer._install_shared_download_wrappers())
            wrapped = download.download_file
            self.assertEqual(inspect.signature(wrapped, follow_wrapped=False).parameters["kwargs"].kind,
                             inspect.Parameter.VAR_KEYWORD)
            self.assertEqual(wrapped("https://one", "one.bin"), "one.bin")
            self.assertEqual(wrapped("https://two", "two.bin", gen=marker, show_filename=False), "two.bin")
            self.assertEqual(wrapped(url="https://three", filename="three.bin", gen=marker,
                                     show_filename=False, future_option="future"), "three.bin")
            self.assertEqual(calls[1], ("https://two", "two.bin", marker, False, None))
            self.assertEqual(calls[2], ("https://three", "three.bin", marker, False, "future"))
            self.assertEqual(telemetry.snapshot()["files"][-1]["name"], "three.bin")
            second = module.DownloadObserver(module.DownloadTelemetry())
            self.assertTrue(second._install_shared_download_wrappers())
            self.assertIs(download.download_file, wrapped)
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                wrapped("https://four", "four.bin", future_option="fail")
            failed = telemetry.snapshot()
            self.assertEqual(failed["files"][-1]["name"], "four.bin")
            self.assertEqual(failed["files"][-1]["state"], "failed")
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_stopped_queue_ignores_lingering_abort_and_progress(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for queue-status regression validation")
        javascript = _javascript_with_exports("readLiveSnapshot")
        test_script = r'''
const {readLiveSnapshot} = globalThis.__statusProReleaseTest;
const namespace = {
  state: {currentId: null, overallElapsed: null, steps: {}, records: {}},
  source: {querySelector: selector => selector === "textarea, input" ? {value: "Aborting"} : null, querySelectorAll: () => []},
  download: {active: false, visible: false},
  runTelemetry: {in_progress: true, active_task: {id: 1}, status: "Aborting"}
};
function check(condition, message) { if (!condition) throw new Error(message); }
check(readLiveSnapshot(namespace).aborting, "active abort must remain visible");
namespace.runTelemetry = {in_progress: false, active_task: null, status: "Aborting", queue_length: 0};
for (let tick = 0; tick < 5; tick++) {
  check(readLiveSnapshot(namespace) === null, "stopped queue revived Prepare");
}
namespace.runTelemetry.status = "";
check(readLiveSnapshot(namespace) === null, "stale DOM abort revived Prepare");
namespace.download.visible = true;
namespace.state.currentId = "prepare";
check(readLiveSnapshot(namespace) === null, "completed download revived Prepare");
namespace.runTelemetry.model_lifecycle = {state: "unloaded"};
check(readLiveSnapshot(namespace) === null, "completed unload revived Prepare");
namespace.runTelemetry.model_lifecycle = {state: "unloading"};
check(readLiveSnapshot(namespace).activity === "unload", "live unload was hidden");
namespace.runTelemetry.model_lifecycle = null;
namespace.download.active = true;
namespace.source.querySelector = () => null;
check(readLiveSnapshot(namespace).rawName === "Downloading model files", "live download was hidden");
namespace.download.active = false;
namespace.runTelemetry = {in_progress: true, active_task: {id: 2}, status: "Loading model"};
check(readLiveSnapshot(namespace).rawName === "Loading model", "next queued run was hidden");
namespace.runTelemetry = null;
namespace.source.querySelector = selector => selector === "textarea, input" ? {value: "Aborting"} : null;
check(readLiveSnapshot(namespace).aborting, "missing telemetry disabled DOM fallback");
'''
        result = subprocess.run(
            [node, "-"], input=javascript + "\n" + test_script,
            text=True, encoding="utf-8", capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_telemetry_recovery_and_memory_sample_deduplication(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for live telemetry validation")
        javascript = _javascript_with_exports("syncRunTelemetry", "startRun", "freshState", "observePerformanceTelemetry")
        test_script = r'''
const api = globalThis.__statusProReleaseTest;
const assert = (condition, message) => {if (!condition) throw new Error(message);};
globalThis.window = {localStorage: {getItem: () => null, setItem: () => {}}};
const ns = {
  state: api.freshState(), source: {querySelector: () => null},
  container: {querySelector: () => null}, historyRecording: false,
  runHistory: [], sessionRunIds: new Set(), historyOpen: false,
};
const task = {id: 1, settings: {}};
api.startRun(ns, task, {server_time: 1, in_progress: true, active_task: task});
const originalRun = ns.activeRun;
for (const telemetry of [
  {server_time: 2, error: "temporary snapshot failure"},
  {server_time: 2}, {server_time: 2, in_progress: true, active_task: null},
  {server_time: 2, in_progress: false, error: "snapshot failure"},
]) {
  ns.runTelemetry = telemetry;
  api.syncRunTelemetry(ns);
  assert(ns.activeRun === originalRun, "incomplete telemetry closed the active run");
}
ns.runTelemetry = {server_time: 3, in_progress: true, active_task: task};
api.syncRunTelemetry(ns);
assert(ns.activeRun === originalRun, "telemetry recovery split the run");
ns.runTelemetry = {server_time: 4, in_progress: false, active_task: null};
api.syncRunTelemetry(ns);
assert(ns.activeRun === null, "valid completion did not finish the run");
const run = {queue_task_id: 1, started_at: 0};
const telemetry = {resource_sample: {sampled_at: 2, ram_rss_bytes: 200},
  performance: {id: "p", task_id: 1, steps: [{sequence: 1, completed_at: 1,
    memory: {sampled_at: 1, ram_rss_bytes: 100}}]}};
api.observePerformanceTelemetry(run, telemetry);
api.observePerformanceTelemetry(run, telemetry);
telemetry.resource_sample = {sampled_at: 3, ram_rss_bytes: 400};
api.observePerformanceTelemetry(run, telemetry);
assert(run.resources.sample_count === 2, "periodic sample was counted twice");
assert(run.resources.metrics.ram_rss_bytes.average_bytes === 300, "wrong memory average");
assert(run.step_performance.length === 1, "step observation duplicated");
'''
        result = subprocess.run([node, "-"], input=javascript + "\n" + test_script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_embedded_javascript_syntax(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for embedded JavaScript syntax validation")
        result = subprocess.run(
            [node, "--check", "-"],
            input=_returned_string("_javascript"),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_release_documents_exist(self):
        for name in ("README.md", "USER_GUIDE.md", "CHANGELOG.md", "RELEASE_NOTES.md", "LICENSE"):
            self.assertTrue((ROOT / name).is_file(), name)


    def test_v13_execution_boundary_and_completed_stage_retention(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for execution-boundary validation")
        source = _source()
        for token in ('"status_display":', '"execution_task_known":', '"executing_task": executing_task'):
            self.assertIn(token, source)
        javascript = _javascript_with_exports("syncRunTelemetry", "readLiveSnapshot", "applySnapshot", "freshState")
        script = r"""
const api=globalThis.__statusProReleaseTest,ok=(v,m)=>{if(!v)throw new Error(m)};
globalThis.window={localStorage:{getItem:()=>null,setItem:()=>{}}};
const ns={state:api.freshState(),source:{querySelector:()=>null,querySelectorAll:()=>[]},
container:{querySelector:()=>null},download:{active:false,visible:false},historyRecording:false,
runHistory:[],sessionRunIds:new Set(),sessionId:"test",lastExecutingTaskKey:"",
lastExecutionProgressSignature:"",progressEpochReady:true};
const A={id:"A",settings:{}},B={id:"B",settings:{}};
const t=(task,phase,extra={})=>({server_time:10,in_progress:true,execution_task_known:true,
executing_task:task,active_task:task,queue_length:task?1:0,status_display:Boolean(task),status:phase,
progress_phase:[phase,null],native_progress:{phase,current:null,total:null,unit:null,progress:null},...extra});
const sync=x=>{ns.runTelemetry=x;api.syncRunTelemetry(ns)};
sync(t(A,"Saving"));api.applySnapshot(ns,api.readLiveSnapshot(ns));
ok(ns.activeRun.queue_task_id==="A"&&ns.state.currentId==="save","A did not reach Save");
sync(t(null,"Saved",{queue_length:1,status_display:true,output_records:[{path:"result.mp4",settings:{}}]}));
ok(ns.activeRun===null&&ns.state.records.save.state==="complete","A completion was not retained");
ok(ns.completedStateUntil>Date.now()&&api.readLiveSnapshot(ns)===null,"stale Saved survived transition");
sync(t(B,"Saved"));ok(ns.activeRun.queue_task_id==="B"&&!ns.progressEpochReady&&api.readLiveSnapshot(ns)===null,"B inherited A progress");
sync(t(B,"Loading model"));ok(ns.progressEpochReady&&api.readLiveSnapshot(ns).id==="prepare","fresh B progress missing");
const legacy={...t(A,"Preparing")};delete legacy.execution_task_known;delete legacy.executing_task;
legacy.active_task=A;ns.activeRun=null;sync(legacy);ok(ns.activeRun.queue_task_id==="A","legacy fallback regressed");
"""
        result = subprocess.run([node, "-"], input=javascript + "\n" + script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_repeated_pipeline_passes_preserve_monotonic_stage_completion(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for repeated-pass validation")
        javascript = _javascript_with_exports("freshState", "applySnapshot", "startRun")
        script = r"""
const api=globalThis.__statusProReleaseTest,ok=(v,m)=>{if(!v)throw new Error(m)};
globalThis.window={localStorage:{getItem:()=>null,setItem:()=>{}}};
const make=()=>({state:api.freshState(),activeRun:{settings:{},step_performance:[]},
 source:{querySelector:()=>null,querySelectorAll:()=>[]},sessionId:"test"});
const snap=(id,name,evidence="structured",aborting=false)=>({id,rawName:name,rawMessage:name,
 progress:null,steps:{current:null,total:null,unit:null},stageElapsed:null,overallElapsed:null,
 transitionEvidence:evidence,aborting});
const ns=make();
api.applySnapshot(ns,snap("denoise","Denoising first phase"));
api.applySnapshot(ns,snap("decode","VAE Decoding"));
api.applySnapshot(ns,snap("post","Spatial refinement"));
api.applySnapshot(ns,snap("denoise","Denoising second phase"));
ok(ns.state.records.denoise.hasCompleted&&ns.state.records.denoise.isActive,"Generate cannot be complete and active");
ok(ns.state.records.decode.hasCompleted&&ns.state.records.post.hasCompleted,"later Generate erased downstream completion");
ok(ns.state.records.decode.state==="complete"&&ns.state.records.post.state==="complete","downstream ticks disappeared");
ok(ns.state.phaseOrder.filter(id=>ns.state.phases[id].stage==="denoise").length===2,"new Generate pass not tracked separately");
api.applySnapshot(ns,snap("decode","VAE Decoding second pass"));
api.applySnapshot(ns,snap("save","Saving"));
ok(ns.state.records.denoise.hasCompleted&&ns.state.records.decode.hasCompleted,"repeated stages lost completion");
ok(ns.state.records.save.isActive,"final Save not displayed");
const stale=make();
for(const s of [snap("encode","Encoding Text Prompt"),snap("denoise","Denoising"),snap("decode","VAE Decoding")]) api.applySnapshot(stale,s);
const phaseCount=stale.state.phaseOrder.length;
ok(api.applySnapshot(stale,snap("denoise","Denoising","legacy-dom"))===false,"stale Generate rewind accepted");
ok(stale.state.currentId==="decode"&&stale.state.phaseOrder.length===phaseCount,"stale Generate mutated phases");
ok(api.applySnapshot(stale,snap("encode","Encoding Text Prompt","legacy-dom"))===false,"stale Encode rewind accepted");
const aborting=make();
for(const s of [snap("denoise","Denoising first phase"),snap("decode","VAE Decoding"),
 snap("post","Spatial refinement"),snap("denoise","Denoising second phase")]) api.applySnapshot(aborting,s);
api.applySnapshot(aborting,snap("denoise","Aborting","structured",true));
ok(aborting.state.records.denoise.hasCompleted&&aborting.state.records.denoise.state==="aborting","repeat-pass abort erased completion");
ok(aborting.state.records.decode.hasCompleted&&aborting.state.records.post.hasCompleted,"abort erased earlier stages");
api.startRun(ns,{id:"B",settings:{}},{server_time:2});
ok(Object.values(ns.state.records).every(r=>!r.hasRun&&!r.hasCompleted&&!r.isActive),"new task inherited stage history");
"""
        result = subprocess.run([node, "-"], input=javascript + "\n" + script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


    def test_stable_stage_slots_and_structured_v13_authority(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for stage-presentation validation")
        javascript = _javascript_with_exports(
            "freshState", "applySnapshot", "renderStages", "readLiveSnapshot", "structuredStageId", "STAGE_DEFS",
        )
        script = r"""
const api=globalThis.__statusProReleaseTest,ok=(v,m)=>{if(!v)throw new Error(m)};
globalThis.window={localStorage:{getItem:()=>null,setItem:()=>{}}};
function element(tag="div"){
 const node={tagName:tag,dataset:{},style:{},children:[],attributes:{},className:"",disabled:false,title:"",
  classList:{values:new Set(),toggle(name,on){if(on)this.values.add(name);else this.values.delete(name)},contains(name){return this.values.has(name)}},
  append(...items){items.forEach(item=>this.children.push(item))},appendChild(item){this.children.push(item);return item},
  setAttribute(name,value){this.attributes[name]=String(value)},
  querySelectorAll(selector){return selector==="[data-stage-id]"?this.children.filter(x=>x.dataset.stageId):[]},
  querySelector(selector){const cls=selector.replace(/^\./,"");return this.children.find(x=>x.className===cls)||null}};
 return node;
}
globalThis.document={createElement:element};
const container=element();container.clientWidth=2000;
const ns={state:api.freshState(),activeRun:{settings:{},step_performance:[]},
 panel:{querySelector:s=>s==="[data-sp-stages]"?container:null},
 source:{querySelector:()=>null,querySelectorAll:()=>[]}};
api.renderStages(ns);
ok(container.children.length===api.STAGE_DEFS.length,"canonical stage slots were not reserved");
const ids=container.children.map(x=>x.dataset.stageId),refs=[...container.children];
ok(ids.join(",")==="prepare,input,encode,denoise,decode,post,save","canonical order changed");
ok(container.children[2].dataset.stagePosition==="3"&&
   container.children[2].querySelector(".status-lite__stage-icon").textContent==="3","Encode number was not canonical");
ok(container.children[1].classList.contains("status-pro__stage--reserved")||
   container.children[1].classList.contains("status-lite__stage--reserved"),"unseen Inputs slot is prominent");
const snap=(id,name,evidence="structured")=>({id,rawName:name,rawMessage:name,progress:null,
 steps:{current:null,total:null,unit:null},stageElapsed:null,overallElapsed:null,transitionEvidence:evidence,aborting:false});
for(const s of [snap("input","VAE Encoding"),snap("encode","Encoding Text Prompt"),
 snap("denoise","Denoising"),snap("decode","VAE Decoding"),snap("post","Upsampling")]){
 api.applySnapshot(ns,s);api.renderStages(ns);
 ok(container.children.every((node,index)=>node===refs[index]),"existing stage node was reordered");
 ok(container.children.map(x=>x.dataset.stageId).join(",")===ids.join(","),"optional stage changed order");
}
const before=container.children.length;
api.applySnapshot(ns,snap("denoise","Denoising second phase"));api.renderStages(ns);
ok(container.children.length===before&&ns.state.records.denoise.runCount===2,"repeated Generate duplicated its card");
api.applySnapshot(ns,snap("save","Saving"));api.renderStages(ns);
ok(container.children.every((node,index)=>node===refs[index]),"late Save reordered stage nodes");
ok(api.structuredStageId("VAE Encoding")==="input","VAE Encode structured mapping");
ok(api.structuredStageId("Encoding Text Prompt 1/2")==="encode","prompt Encode structured mapping");
ok(api.structuredStageId("VAE Decoding")==="decode","VAE Decode structured mapping");
const live={state:api.freshState(),activeRun:{settings:{},step_performance:[]},download:{active:false,visible:false},
 source:{querySelector:()=>null,querySelectorAll:()=>[]}};
const phase=(name,status)=>({in_progress:true,execution_task_known:true,executing_task:{id:1},
 active_task:{id:1},native_progress:{phase:name,current:null,total:null,unit:null,progress:null},
 progress_phase:[name,null],status});
live.runTelemetry=phase("VAE Decoding","Generating");
ok(api.readLiveSnapshot(live).id==="decode","stale Generate text overrode structured Decode");
live.runTelemetry=phase("Denoising","Encoding prompt");
ok(api.readLiveSnapshot(live).id==="denoise","stale Encode text overrode structured Generate");
api.applySnapshot(live,snap("decode","VAE Decoding"));
live.runTelemetry=phase("Uncatalogued accelerator phase","Encoding prompt");
const unknown=api.readLiveSnapshot(live);
ok(unknown.id==="decode"&&unknown.transitionEvidence==="structured-unknown","unknown structured phase rewound");
"""
        result = subprocess.run([node, "-"], input=javascript + "\n" + script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


    def test_authoritative_stage_timing_recovery_and_interpolation(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for authoritative timing validation")
        source = _source()
        self.assertIn('"stage_timing": self._stage_timing.snapshot', source)
        javascript = _javascript_with_exports(
            "freshState", "applySnapshot", "applyServerStageTiming", "stageElapsedNow", "startRun",
        )
        script = r"""
const api=globalThis.__statusProReleaseTest,ok=(v,m)=>{if(!v)throw new Error(m)};
let mono=30000;
Object.defineProperty(globalThis,"performance",{value:{now:()=>mono},configurable:true});
globalThis.window={localStorage:{getItem:()=>null,setItem:()=>{}}};
const ns={state:api.freshState(),activeRun:{settings:{},step_performance:[]},sessionId:"timing",
 source:{querySelector:()=>null,querySelectorAll:()=>[]}};
const timing=(stages,last="denoise",revision=1)=>({stage_timing:{task_id:"A",revision,last_stage:last,stages}});
api.applyServerStageTiming(ns,timing({
 input:{elapsed:2,active:false,completed:true,run_count:1},
 encode:{elapsed:7,active:false,completed:true,run_count:1},
 denoise:{elapsed:15,active:true,completed:false,run_count:1}
}));
ok(ns.state.records.input.hasCompleted&&ns.state.records.input.elapsed===2,"missed Inputs not reconstructed");
ok(ns.state.records.encode.hasCompleted&&ns.state.records.encode.elapsed===7,"polling delay inflated Encode");
ok(ns.state.currentId==="denoise"&&ns.state.records.denoise.elapsed===15,"active Generate not recovered");
mono+=2000;
ok(Math.abs(api.stageElapsedNow(ns.state.records.denoise)-17)<0.001,"local monotonic interpolation failed");
api.applyServerStageTiming(ns,timing({denoise:{elapsed:50,active:true,completed:false,run_count:1}},"denoise",2));
ok(Math.abs(api.stageElapsedNow(ns.state.records.denoise)-50)<0.001,"fresh server timing did not reconcile");
mono+=2000;
ok(Math.abs(api.stageElapsedNow(ns.state.records.denoise)-52)<0.001,"interpolation did not resume");
api.applyServerStageTiming(ns,timing({denoise:{elapsed:18,active:false,completed:true,run_count:2}},"denoise",3));
ok(ns.state.records.denoise.elapsed===18&&ns.state.records.denoise.runCount===2,"repeated Generate timing lost");
const before=ns.state.records.denoise.runCount;
api.applySnapshot(ns,{id:"decode",rawName:"VAE Decoding",rawMessage:"",steps:{},progress:null,
 stageElapsed:null,overallElapsed:null,transitionEvidence:"structured",aborting:false});
const phaseCount=ns.state.phaseOrder.length;
ok(api.applySnapshot(ns,{id:"denoise",rawName:"Denoising",rawMessage:"",steps:{},progress:null,
 stageElapsed:null,overallElapsed:null,transitionEvidence:"legacy-dom",aborting:false})===false,"stale rewind accepted");
ok(ns.state.records.denoise.runCount===before&&ns.state.phaseOrder.length===phaseCount,"stale rewind altered timing");
api.applyServerStageTiming(ns,timing({save:{elapsed:3,active:false,completed:true,run_count:1}},"save",4));
mono+=1600;
ok(api.stageElapsedNow(ns.state.records.save)===3,"completion grace inflated Save");
api.applyServerStageTiming(ns,timing({denoise:{elapsed:12,active:false,completed:false,run_count:1}},"denoise",5));
api.applySnapshot(ns,{id:"denoise",rawName:"Aborting",rawMessage:"Aborting",steps:{},progress:null,
 stageElapsed:null,overallElapsed:null,transitionEvidence:"structured",aborting:true});
mono+=5000;
ok(api.stageElapsedNow(ns.state.records.denoise)===12&&ns.state.records.denoise.state==="aborting","abort timing continued");
api.startRun(ns,{id:"B",settings:{}},{server_time:2});
ok(Object.values(ns.state.records).every(r=>!r.serverTimed&&!r.hasRun),"task timing leaked");
const legacy={state:api.freshState(),activeRun:{},source:ns.source};
const oldNow=Date.now;let wall=1000;Date.now=()=>wall;
api.applySnapshot(legacy,{id:"encode",rawName:"Encoding prompt",rawMessage:"",steps:{},progress:null,
 stageElapsed:null,overallElapsed:null,aborting:false});
wall=11000;
api.applySnapshot(legacy,{id:"denoise",rawName:"Denoising",rawMessage:"",steps:{},progress:null,
 stageElapsed:null,overallElapsed:null,aborting:false});
Date.now=oldNow;
ok(Math.abs(legacy.state.records.encode.elapsed-10)<0.001,"legacy frontend timing fallback broke");
"""
        result = subprocess.run([node, "-"], input=javascript + "\n" + script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

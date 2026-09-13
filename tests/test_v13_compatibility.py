"""Shared live V13 regressions, run independently against each edition."""
import ast
import copy
import json
import shutil
import subprocess
import time
import types
import unittest
from functools import wraps

from test_release_smoke import _source, _javascript_with_exports


def python_helpers(*names):
    tree = ast.parse(_source())
    definitions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"time": time, "wraps": wraps}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "v13-test", "exec"), scope)
    return scope


class V13CompatibilityTests(unittest.TestCase):
    def test_normalized_progress_copies_valid_counts_and_rejects_stale_units(self):
        normalize = python_helpers("_native_progress_snapshot")["_native_progress_snapshot"]
        for phase, current, total, unit in [
            ("Encoding Text Prompt 1/2", 17, 32, "layers"),
            ("VAE Encoding", 6, 14, "tiles"),
            ("VAE Decoding", 4, 12, "tiles"),
        ]:
            for field in ["last_progress_args", "phase_progress_units"]:
                gen = {"progress_phase": [phase, current], "num_inference_steps": 50}
                gen[field] = [(current, total), None, None, unit] if field == "last_progress_args" else [total, unit]
                result = normalize(gen)
                self.assertEqual(result, {"phase": phase, "current": current, "total": total,
                                          "unit": unit, "progress": current / total * 100})
                gen["progress_phase"][1] = 0
                self.assertEqual(result["current"], current)
                json.dumps(result, allow_nan=False)
        for current in [None, -1, float("nan")]:
            result = normalize({"progress_phase": ["Preparing Conditioning", current],
                                "phase_progress_units": [32, "layers"],
                                "last_progress_args": [(17, 32), None, None, "layers"]})
            self.assertEqual(result, dict(phase="Preparing Conditioning", current=None, total=None, unit=None, progress=None))
        self.assertIsNone(normalize({})["progress"])
        self.assertIsNone(normalize({"progress_phase": ["VAE Decoding", 4], "num_inference_steps": 30})["total"])
        legacy = normalize({"progress_phase": ["Denoising", 12], "num_inference_steps": 30})
        self.assertEqual((legacy["current"], legacy["total"], legacy["unit"], legacy["progress"]), (12, 30, "steps", 40))
        self.assertEqual(normalize({"progress_phase": ["VAE Decoding", 13], "phase_progress_units": [12, "tiles"]})["progress"], 100)
        self.assertIn('"native_progress": _native_progress_snapshot(gen)', _source())

    def test_callback_layer_tile_progress_never_changes_denoising_observer(self):
        scope = python_helpers("_install_step_observer", "_new_performance_observer", "_telemetry_value")
        reads = []
        scope.update(_skip_count=lambda pipe: reads.append("skip") or 0,
                     _memory_snapshot=lambda torch: reads.append("memory") or {}, MAX_STEP_TELEMETRY=300)
        forwarded = []
        def original(*args, **kwargs):
            forwarded.append((args, kwargs))
            return "original-result"
        owner = types.SimpleNamespace(_step_observer_installed=False, build_callback=lambda *a, **k: original)
        owner.set_global = lambda name, value: setattr(owner, name, value)
        scope["_install_step_observer"](owner)
        state = {"gen": {"queue": [{"id": 1}]}}
        callback = owner.build_callback(state, types.SimpleNamespace(cache=None), num_inference_steps=30)
        performance = owner._latest_performance
        baseline = copy.deepcopy(performance)
        read_count = len(reads)
        for title, unit in [("Encoding Text Prompt", "layers"), ("Encoding Text Prompt 1/2", "layers"),
                            ("VAE Encoding", "tiles"), ("VAE Decoding", "tiles"), ("", "tiles")]:
            for step in [0, -1]:
                kwargs = dict(step_idx=step, progress_title=title, progress_unit=unit, override_num_inference_steps=12)
                self.assertEqual(callback(**kwargs), "original-result")
                self.assertEqual(forwarded[-1], ((), kwargs))
                self.assertEqual(performance, baseline)
                self.assertEqual(len(reads), read_count)
        positional = (3, None, True, None, 14, None, None, None, "tiles", "prefix", "VAE Encoding")
        callback(*positional)
        self.assertEqual(forwarded[-1], (positional, {}))
        self.assertEqual(performance, baseline)
        callback(0)
        self.assertEqual(len(performance["steps"]), 1)
        self.assertEqual(performance["steps"][0]["total_steps"], 30)
        callback(30)  # Pre-V13 Decode sentinel.
        self.assertEqual(len(performance["steps"]), 1)
        callback(-1)
        self.assertEqual(performance["callback_phase"], 1)
        callback(step_idx=1, override_num_inference_steps=40, progress_title=None)
        self.assertEqual(performance["steps"][-1]["total_steps"], 40)

    def test_coexistence_registry_and_lite_observer_guard(self):
        names = ["_register_status_variant"]
        lite = "def _status_pro_registered" in _source()
        if lite:
            names += ["_status_pro_registered", "post_ui_setup"]
        scope = python_helpers(*names)
        for order in [("pro", "lite"), ("lite", "pro")]:
            scope.update(builtins=types.SimpleNamespace(), _STATUS_VARIANT_REGISTRY_KEY="test_registry")
            for variant in order:
                scope["_register_status_variant"](variant)
            self.assertEqual(scope["_register_status_variant"]("pro"), {"pro", "lite"})
            if lite:
                self.assertTrue(scope["_status_pro_registered"]())
                # Any attempted observer installation or panel insertion on this
                # minimal owner would raise; the Pro guard must return first.
                owner = types.SimpleNamespace(_insertion_registered=False)
                scope["post_ui_setup"](owner, {"gen_status": object()})
                self.assertTrue(owner._insertion_registered)

    def test_native_dom_sequences_decode_cancellation_and_qwen(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for V13 frontend validation")
        javascript = _javascript_with_exports(
            "freshState", "readLiveSnapshot", "readSnapshot", "readReportedPhaseStatus", "stageIdFor",
            "applySnapshot", "stageActivities", "finishPhase", "formatCounter", "stageSupportsEta",
            "reinterpretQwenSilentEncode", "renderDetail", "STAGE_DEFS", "recoveredPerformanceGroups",
        )
        script = r"""
const api = globalThis.__statusProReleaseTest;
const assert = (value, message) => { if (!value) throw new Error(message); };
globalThis.window = {localStorage: {getItem: () => null, setItem: () => {}}};
function namespace() {return {state: api.freshState(), activeRun: {settings: {}, step_performance: []},
    source: {querySelector: () => null, querySelectorAll: () => []}, download: {active: false, visible: false}};}
function native(ns, phase, current = null, total = null, unit = null) {
    ns.runTelemetry = {in_progress: true, native_progress: {phase, current, total, unit},
        progress_phase: [phase, current], status: "", active_task: {settings: {}}};
    return api.readLiveSnapshot(ns);
}
assert(api.STAGE_DEFS.map(s => s.id).join() === "prepare,input,encode,denoise,decode,post,save", "top-level timeline changed");
for (const [phase, id] of [["VAE Encoding", "input"], ["Preparing Conditioning", "encode"],
    ["Encoding Text Prompt 1/2", "encode"], ["Denoising", "denoise"], ["VAE Decoding", "decode"]]) {
    assert(api.stageIdFor(phase) === id, `wrong mapping: ${phase}`);
}
for (const sequence of [
    ["Preparing", "Encoding Text Prompt", "Denoising", "VAE Decoding", "Saving"],
    ["Preparing", "VAE Encoding", "Preparing Conditioning", "Encoding Text Prompt", "Denoising", "VAE Decoding", "Saving"],
    ["Encoding Text Prompt 1/2", "Preparing Conditioning", "Encoding Text Prompt 2/2", "Denoising"]
]) {
    const ns = namespace();
    let previous = -1;
    for (const phase of sequence) {
        const s = native(ns, phase, phase === "Preparing Conditioning" ? null : 4, 12,
            phase.startsWith("VAE") ? "tiles" : phase.startsWith("Encoding") ? "layers" : "steps");
        const index = api.STAGE_DEFS.findIndex(stage => stage.id === s.id);
        assert(index >= previous, `rewound at ${phase}`); previous = index;
        api.applySnapshot(ns, s);
        assert(ns.state.currentId === s.id, "wrong active stage");
        if (["input", "encode", "decode"].includes(s.id)) {
            assert(ns.state.records[s.id].eta === null && !api.stageSupportsEta(ns.state.records[s.id]), "invented tile/layer ETA");
        }
    }
    const activities = api.stageActivities(ns.state, ns.state.records.encode);
    assert(activities.length > 0, "Encode phases missing");
    assert(activities.filter(a => a.total).every(a => a.progress === 100 && a.current === a.total), "completed phase not finalized");
}
const repeated = namespace();
for (const phase of ["Preparing Conditioning", "Encoding Text Prompt 1/2", "Preparing Conditioning", "Encoding Text Prompt 2/2"]) {
    api.applySnapshot(repeated, native(repeated, phase));
}
assert(api.stageActivities(repeated.state, repeated.state.records.encode).length === 4, "repeated subphase overwritten");
for (const initial of ["VAE Encoding", "Encoding Text Prompt"]) {
    for (const stop of ["Aborting", "Cancelling", "Interrupting", "Stopping", "Early-stop processing"]) {
        const ns = namespace();
        api.applySnapshot(ns, native(ns, initial, 4, 12, "tiles"));
        const previous = ns.state.currentId, phaseId = ns.state.currentPhaseId;
        api.applySnapshot(ns, native(ns, stop));
        assert(ns.state.currentId === previous && ns.state.records[previous].state === "aborting", "cancel rewound stage");
        assert(ns.state.currentPhaseId === phaseId && ns.state.phases[phaseId].state === "aborting", "cancel completed old phase");
        api.finishPhase(ns.state);
        assert(ns.state.phases[phaseId].state === "aborting", "finish marked abort complete");
    }
}
const decode = namespace();
decode.state.records.denoise.stepTotal = 30;
api.applySnapshot(decode, native(decode, "VAE Decoding", 4, 12, "tiles"));
assert(Math.abs(decode.state.records.decode.progress - 100/3) < 0.001, "V13 Decode percentage lost");
assert(api.formatCounter(decode.state.steps) === "4/12 tiles", "Decode substituted denoising counter");
assert(api.formatCounter({current: 17, total: 32, unit: "layers"}) === "17/32 layers", "layer formatter");
assert(api.formatCounter({current: 12, total: 30}) === "12/30 steps", "legacy counter default");
const legacy = namespace();
legacy.state.records.denoise.stepTotal = 30;
api.applySnapshot(legacy, {id: "decode", rawName: "VAE Decoding", rawMessage: "VAE Decoding", progress: 100,
    steps: {current: 30, total: 30}, stageElapsed: null, overallElapsed: null});
assert(legacy.state.records.decode.progress === null, "legacy Decode bar trusted");
function element(textContent, attrs = {}) {return {textContent, getAttribute: name => attrs[name] ?? null};}
const fixture = {
    ".progress-title": element("VAE Decoding"), ".progress-percent": element("99%"),
    ".progress-track": element("", {"aria-valuenow": "33.3"}),
    ".progress-amount": element("4 / 12 tiles"), ".progress-timing": element("8s / 24s")
};
const tracker = {querySelector: selector => fixture[selector] || null};
const dom = namespace();
dom.source.querySelector = selector => selector === ".wangp-progress" ? tracker : null;
const parsed = api.readSnapshot(dom);
assert(parsed.rawName === "VAE Decoding" && parsed.progress === 33.3 && parsed.progressScope === "phase", "V13 DOM fields");
assert(parsed.steps.current === 4 && parsed.steps.total === 12 && parsed.steps.unit === "tiles" && parsed.stageElapsed === 8, "V13 DOM counter/timing");
const primary = native(dom, "Encoding Text Prompt", 17, 32, "layers");
assert(primary.id === "encode" && primary.steps.current === 17, "DOM overrode native state");
const oldFields = {".progress-level-inner": element("Denoising - 40% | 4s / 10s"), ".progress-text": element("12/30 steps | 4s")};
const old = {querySelector: selector => oldFields[selector] || null};
dom.source = {querySelector: () => null, querySelectorAll: () => [old]}; dom.tracker = null;
const oldParsed = api.readSnapshot(dom);
assert(oldParsed.id === "denoise" && oldParsed.progress === 40 && oldParsed.steps.current === 12, "legacy DOM regression");
dom.runTelemetry = null;
api.applySnapshot(dom, {id: "encode", rawName: "Encoding Text Prompt", steps: {current: null, total: null}, progress: null});
oldFields[".progress-level-inner"].textContent = "Stopping";
const legacyStop = api.readLiveSnapshot(dom);
assert(legacyStop.id === "encode" && legacyStop.aborting, "legacy stop rewound phase");
const qwen = namespace();
qwen.runTelemetry = {active_task: {settings: {model_type: "qwen_image"}}, performance: {callback_phase: 0, steps: []}};
const fallback = {id: "denoise", rawName: "Denoising", steps: {current: 0, total: 30}, progress: 0};
assert(api.reinterpretQwenSilentEncode(qwen, fallback).id === "encode", "Qwen layer-only phase ended Encode");
qwen.runTelemetry.performance.steps = [{sequence: 1}];
assert(api.reinterpretQwenSilentEncode(qwen, fallback).id === "denoise", "real denoising did not end fallback");
qwen.runTelemetry.performance.steps = [];
assert(api.reinterpretQwenSilentEncode(qwen, {...fallback, progressScope: "phase"}).id === "denoise", "trusted native phase rewritten");
assert(api.recoveredPerformanceGroups({step_performance: []}).length === 0, "phase-only data recovered as performance");
// Exercise the detail renderer, including the Decode progress metric and activity text.
const elements = new Map();
function ui() {return {hidden: false, children: [], textContent: "", classList: {toggle: () => {}},
    replaceChildren() {this.children = [];}, appendChild(child) {this.children.push(child);},
    setAttribute() {}, removeAttribute() {}, append() {}, querySelector() {return null;}};}
globalThis.document = {createElement: () => ui()};
decode.panel = {querySelector: selector => {if (!elements.has(selector)) elements.set(selector, ui()); return elements.get(selector);}};
api.renderDetail(decode);
assert(elements.get("[data-sp-progress-metric]").hidden === false, "V13 Decode metric hidden");
assert(elements.get("[data-sp-detail-activities]").children[0].textContent.includes("4/12 tiles"), "activity counter missing");
assert(elements.get("[data-sp-detail-message]").hidden, "duplicated activity text");
legacy.panel = decode.panel;
api.renderDetail(legacy);
assert(elements.get("[data-sp-progress-metric]").hidden === true, "legacy Decode metric visible");
"""
        result = subprocess.run([node, "-"], input=javascript + "\n" + script, text=True,
                                encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

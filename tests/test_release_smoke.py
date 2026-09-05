import ast
import json
import pathlib
import shutil
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN_PATH = ROOT / "plugin.py"
DOWNLOAD_PATH = ROOT / "download_telemetry.py"


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
        self.assertEqual(manifest["version"], "1.0.0")
        self.assertEqual(manifest["type"], "extension")
        self.assertEqual(manifest["wan2gp_version"], "0")
        self.assertIn('class StatusLitePlugin(WAN2GPPlugin):', source)
        self.assertIn('self.name = "Status Lite"', source)
        self.assertIn('self.version = "1.0.0"', source)
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

    def test_stopped_queue_ignores_lingering_abort_and_progress(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for queue-status regression validation")
        javascript = _javascript_with_exports("readLiveSnapshot")
        test_script = r'''
const {readLiveSnapshot} = globalThis.__statusProReleaseTest;
const namespace = {
  state: {currentId: null, overallElapsed: null, steps: {}, records: {}},
  source: {querySelector: () => ({value: "Aborting"}), querySelectorAll: () => []},
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
namespace.source.querySelector = () => ({value: "Aborting"});
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


if __name__ == "__main__":
    unittest.main()

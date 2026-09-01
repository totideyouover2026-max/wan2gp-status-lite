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

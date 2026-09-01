"""Plugin-local download observation for Status Lite.

The wrappers in this module always delegate the actual transfer to Wan2GP and
Hugging Face. They only mirror progress into a thread-safe in-memory snapshot.
No application files are modified.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable


_VISIBLE_AFTER_COMPLETION_SECONDS = 12.0
_NEW_SESSION_AFTER_SECONDS = 8.0
_MAX_RECORDED_FILES = 200
_TRANSFER_CYCLE_QUIET_SECONDS = 3.0
_MAX_TRANSFER_CYCLES = 6


def _clean_name(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1] or "Unknown file"


def _shortened_name_prefix(value: Any) -> str:
    """Return the retained prefix from Hugging Face's shortened tqdm labels."""
    name = _clean_name(value).lower()
    match = re.match(r"^(.+?)(?:\(\s*(?:…|\.\.\.)\s*\))$", name)
    return match.group(1).strip() if match else ""


class DownloadTelemetry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._files: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._counter = 0
        self._revision = 0
        self._active_batches = 0
        self._started_at: float | None = None
        self._updated_at = 0.0
        self._completed_at: float | None = None
        self._session_label = "Model assets"

    def _touch(self, now: float | None = None) -> float:
        timestamp = time.time() if now is None else now
        self._updated_at = timestamp
        self._revision += 1
        return timestamp

    def _has_running_files(self) -> bool:
        return any(record["state"] in {"pending", "downloading", "retrying"} for record in self._files.values())

    def _maybe_start_session(self, label: str | None = None) -> None:
        now = time.time()
        quiet_for = now - self._updated_at if self._updated_at else float("inf")
        if self._files and not self._has_running_files() and self._active_batches == 0 and quiet_for > _NEW_SESSION_AFTER_SECONDS:
            self._files.clear()
            self._counter = 0
            self._started_at = None
            self._completed_at = None
        if self._started_at is None:
            self._started_at = now
            self._completed_at = None
        if label:
            self._session_label = str(label)

    def _new_record(self, name: str, state: str = "pending", source: str = "") -> dict[str, Any]:
        self._counter += 1
        record = {
            "id": f"download-{self._counter}",
            "name": _clean_name(name),
            "path": str(name or ""),
            "state": state,
            "source": source,
            "downloaded": 0.0,
            "total": None,
            "speed": 0.0,
            "eta": None,
            "started_at": None,
            "last_byte_at": None,
            "completed_at": None,
            "error": "",
            "_last_time": None,
            "_last_downloaded": 0.0,
            "_speed_samples": [],
            "_initial_downloaded": 0.0,
            "_transfer_cycles": [],
        }
        self._files[record["id"]] = record
        while len(self._files) > _MAX_RECORDED_FILES:
            self._files.popitem(last=False)
        return record

    def _find_record(self, name: str, create: bool = False, source: str = "") -> dict[str, Any] | None:
        clean = _clean_name(name)
        lower = clean.lower().replace("(…)", "").replace("...", "")
        shortened_prefix = _shortened_name_prefix(name)
        candidates = []
        for record in self._files.values():
            record_name = str(record["name"]).lower()
            if record_name == clean.lower():
                candidates.append(record)
            elif lower and (record_name.endswith(lower) or lower.endswith(record_name)):
                candidates.append(record)
            elif shortened_prefix and record_name.startswith(shortened_prefix):
                candidates.append(record)
        if candidates:
            running = [record for record in candidates if record["state"] in {"pending", "downloading", "retrying"}]
            return running[0] if running else candidates[-1]
        return self._new_record(name, source=source) if create else None

    @staticmethod
    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

    @staticmethod
    def _record_transfer_cycle(
        record: dict[str, Any],
        now: float,
        previous_byte_at: float,
        previous_downloaded: float,
        downloaded: float,
    ) -> None:
        duration = max(0.0, now - previous_byte_at)
        bytes_transferred = max(0.0, downloaded - previous_downloaded)
        if duration >= _TRANSFER_CYCLE_QUIET_SECONDS and bytes_transferred > 0:
            cycles = list(record.get("_transfer_cycles") or [])
            cycles.append({"bytes": bytes_transferred, "duration": duration})
            record["_transfer_cycles"] = cycles[-_MAX_TRANSFER_CYCLES:]

    def begin_batch(self, missing_files: list[str], label: str | None = None, source: str = "") -> None:
        if not missing_files:
            return
        with self._lock:
            self._maybe_start_session(label)
            self._active_batches += 1
            for name in missing_files:
                record = self._find_record(name, create=True, source=source)
                if record and record["state"] in {"complete", "failed"}:
                    record["state"] = "pending"
                    record["error"] = ""
            self._touch()

    def end_batch(self, missing_files: list[str], error: str | None = None) -> None:
        if not missing_files:
            return
        with self._lock:
            now = time.time()
            for name in missing_files:
                record = self._find_record(name)
                if record is None:
                    continue
                if error:
                    record["state"] = "failed"
                    record["error"] = str(error)
                elif record["state"] != "failed":
                    record["state"] = "complete"
                    if record["total"] is not None:
                        record["downloaded"] = record["total"]
                    record["eta"] = 0.0
                    record["completed_at"] = now
            self._active_batches = max(0, self._active_batches - 1)
            if self._active_batches == 0 and not self._has_running_files():
                self._completed_at = now
            self._touch(now)

    def begin_file(
        self,
        name: str,
        total: float | int | None = None,
        initial: float | int = 0,
        source: str = "",
    ) -> str:
        with self._lock:
            self._maybe_start_session()
            record = self._find_record(name, create=True, source=source)
            assert record is not None
            now = time.time()
            record["state"] = "downloading"
            record["source"] = source or record["source"]
            record["started_at"] = record["started_at"] or now
            if record["downloaded"] > 0:
                record["last_byte_at"] = now
            record["completed_at"] = None
            record["error"] = ""
            record["downloaded"] = max(float(initial or 0), float(record["downloaded"] or 0))
            if total is not None and float(total) > 0:
                record["total"] = float(total)
            record["_last_time"] = now
            record["_last_downloaded"] = float(record["downloaded"])
            record["_initial_downloaded"] = float(record["downloaded"])
            record["_transfer_cycles"] = []
            self._completed_at = None
            self._touch(now)
            return str(record["id"])

    def update_file(
        self,
        name_or_id: str,
        downloaded: float | int,
        total: float | int | None = None,
        state: str = "downloading",
    ) -> None:
        with self._lock:
            record = self._files.get(name_or_id) or self._find_record(name_or_id, create=True)
            assert record is not None
            now = time.time()
            downloaded_value = max(0.0, float(downloaded or 0))
            last_time = record["_last_time"]
            last_downloaded = float(record["_last_downloaded"] or 0)
            previous_downloaded = float(record["downloaded"] or 0)
            previous_byte_at = record.get("last_byte_at")
            if last_time is not None and now > last_time and downloaded_value >= last_downloaded:
                delta = downloaded_value - last_downloaded
                if delta > 0:
                    sample = delta / (now - float(last_time))
                    samples = list(record["_speed_samples"])
                    samples.append(sample)
                    record["_speed_samples"] = samples[-6:]
                    record["speed"] = sum(record["_speed_samples"]) / len(record["_speed_samples"])
            record["_last_time"] = now
            record["_last_downloaded"] = downloaded_value
            record["downloaded"] = downloaded_value
            if downloaded_value > previous_downloaded:
                if previous_byte_at is not None and now - float(previous_byte_at) >= _TRANSFER_CYCLE_QUIET_SECONDS:
                    self._record_transfer_cycle(
                        record,
                        now,
                        float(previous_byte_at),
                        previous_downloaded,
                        downloaded_value,
                    )
                record["last_byte_at"] = now
            record["state"] = state
            if total is not None and float(total) > 0:
                record["total"] = float(total)
            if record["total"] is not None and record["speed"] > 0:
                record["eta"] = max(0.0, (float(record["total"]) - downloaded_value) / float(record["speed"]))
            self._touch(now)

    def complete_file(self, name_or_id: str) -> None:
        with self._lock:
            record = self._files.get(name_or_id) or self._find_record(name_or_id)
            if record is None:
                return
            now = time.time()
            record["state"] = "complete"
            if record["total"] is not None:
                record["downloaded"] = record["total"]
            record["last_byte_at"] = now
            record["eta"] = 0.0
            record["completed_at"] = now
            if self._active_batches == 0 and not self._has_running_files():
                self._completed_at = now
            self._touch(now)

    def fail_file(self, name_or_id: str, error: Any) -> None:
        with self._lock:
            record = self._files.get(name_or_id) or self._find_record(name_or_id, create=True)
            assert record is not None
            now = time.time()
            record["state"] = "failed"
            record["error"] = str(error or "Download failed")
            record["eta"] = None
            if self._active_batches == 0 and not self._has_running_files():
                self._completed_at = now
            self._touch(now)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            files = []
            for record in self._files.values():
                item = {key: value for key, value in record.items() if not key.startswith("_")}
                total = item.get("total")
                if total is not None and float(total) > 0:
                    cycles = list(record.get("_transfer_cycles") or [])
                    average_bytes = self._median([float(cycle["bytes"]) for cycle in cycles])
                    average_duration = self._median([float(cycle["duration"]) for cycle in cycles])
                    started_at = record.get("started_at")
                    elapsed = max(0.0, now - float(started_at)) if started_at is not None else 0.0
                    transferred = max(0.0, float(record["downloaded"] or 0) - float(record.get("_initial_downloaded") or 0))
                    effective_rate = transferred / elapsed if elapsed > 0 and transferred > 0 else None
                    item.update({
                        "transfer_cycles": len(cycles),
                        "cycle_average_bytes": average_bytes,
                        "cycle_average_seconds": average_duration,
                        "effective_rate": effective_rate,
                        "effective_eta": (
                            max(0.0, float(total) - float(record["downloaded"] or 0)) / effective_rate
                            if cycles and effective_rate and effective_rate > 0 else None
                        ),
                    })
                files.append(item)
            known_total = sum(float(record["total"]) for record in files if record["total"] is not None)
            known_downloaded = sum(
                min(float(record["downloaded"] or 0), float(record["total"]))
                for record in files
                if record["total"] is not None
            )
            unknown_downloaded = sum(float(record["downloaded"] or 0) for record in files if record["total"] is None)
            active_files = [record for record in files if record["state"] in {"downloading", "retrying"}]
            aggregate_speed = sum(float(record["speed"] or 0) for record in active_files)
            remaining_known = max(0.0, known_total - known_downloaded)
            eta = remaining_known / aggregate_speed if aggregate_speed > 0 and known_total > 0 else None
            completed = sum(1 for record in files if record["state"] == "complete")
            failed = sum(1 for record in files if record["state"] == "failed")
            running = bool(active_files or self._active_batches > 0 or any(record["state"] == "pending" for record in files))
            visible = bool(files) and (running or self._completed_at is None or now - self._completed_at <= _VISIBLE_AFTER_COMPLETION_SECONDS)
            return {
                "revision": self._revision,
                "visible": visible,
                "active": running,
                "label": self._session_label,
                "started_at": self._started_at,
                "updated_at": self._updated_at,
                "completed_at": self._completed_at,
                "files": files,
                "totals": {
                    "file_count": len(files),
                    "completed": completed,
                    "failed": failed,
                    "downloaded": known_downloaded + unknown_downloaded,
                    "known_downloaded": known_downloaded,
                    "known_total": known_total or None,
                    "speed": aggregate_speed,
                    "eta": eta,
                },
            }

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, separators=(",", ":"))

    def reset_for_tests(self) -> None:
        with self._lock:
            self._files.clear()
            self._counter = 0
            self._revision = 0
            self._active_batches = 0
            self._started_at = None
            self._updated_at = 0.0
            self._completed_at = None


DOWNLOAD_TELEMETRY = DownloadTelemetry()


class _ProgressProxy:
    __status_lite_proxy__ = True

    def __init__(self, progress: Any, telemetry: DownloadTelemetry, name: str, total: Any, initial: Any) -> None:
        self._progress = progress
        self._telemetry = telemetry
        self._name = name
        self._total = total
        self._downloaded = float(initial or 0)
        self._record_id = telemetry.begin_file(name, total=total, initial=initial, source="Hugging Face")

    def update(self, amount: float = 1) -> Any:
        result = self._progress.update(amount)
        self._downloaded += float(amount or 0)
        self._telemetry.update_file(self._record_id, self._downloaded, self._total)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._progress, name)


class DownloadObserver:
    def __init__(self, telemetry: DownloadTelemetry) -> None:
        self.telemetry = telemetry
        self.installed = False
        self.shared_download_available = False
        self.huggingface_progress_available = False
        self.errors: list[str] = []

    def _record_error(self, message: str) -> None:
        message = str(message or "Download observer unavailable").strip()
        if message not in self.errors:
            self.errors.append(message)
        print(f"[Status Lite] {message}")

    def status(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "shared_download_available": self.shared_download_available,
            "huggingface_progress_available": self.huggingface_progress_available,
            "errors": list(self.errors),
        }

    @staticmethod
    def _definition_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        names = ("repoId", "sourceFolderList", "fileList", "targetFolderList")
        definition = {name: kwargs.get(name) for name in names}
        for index, value in enumerate(args[: len(names)]):
            definition[names[index]] = value
        return definition

    def _install_shared_download_wrappers(self) -> bool:
        try:
            from shared.utils import download as download_module
        except Exception as exc:
            self._record_error(f"Wan2GP download telemetry disabled: {exc}")
            return False

        if getattr(download_module, "__status_lite_observer__", None) is not None:
            self.shared_download_available = True
            return True

        required = (
            "process_files_def",
            "download_file",
            "create_progress_hook",
            "download_def_missing_files",
        )
        missing = [name for name in required if not callable(getattr(download_module, name, None))]
        if missing:
            self._record_error(
                "Wan2GP download telemetry disabled because this version does not expose: "
                + ", ".join(missing)
            )
            return False

        original_process_files = download_module.process_files_def
        original_download_file = download_module.download_file
        original_create_progress_hook = download_module.create_progress_hook
        telemetry = self.telemetry

        @functools.wraps(original_process_files)
        def observed_process_files(*args: Any, **kwargs: Any) -> Any:
            definition = self._definition_from_call(args, kwargs)
            try:
                missing = list(download_module.download_def_missing_files(definition))
            except Exception:
                missing = []
            label = f"{definition.get('repoId') or 'Model'} assets"
            telemetry.begin_batch(missing, label=label, source="Hugging Face")
            try:
                result = original_process_files(*args, **kwargs)
            except Exception as exc:
                telemetry.end_batch(missing, error=str(exc))
                raise
            telemetry.end_batch(missing)
            return result

        @functools.wraps(original_download_file)
        def observed_download_file(url: str, filename: str) -> Any:
            name = _clean_name(filename or url)
            telemetry.begin_batch([name], label="Model assets")
            telemetry.begin_file(name, source="Download")
            try:
                result = original_download_file(url, filename)
            except Exception as exc:
                telemetry.fail_file(name, exc)
                telemetry.end_batch([name], error=str(exc))
                raise
            telemetry.complete_file(name)
            telemetry.end_batch([name])
            return result

        @functools.wraps(original_create_progress_hook)
        def observed_create_progress_hook(filename: str) -> Callable[[int, int, int], Any]:
            original_hook = original_create_progress_hook(filename)
            name = _clean_name(filename)

            def hook(block_num: int, block_size: int, total_size: int) -> Any:
                result = original_hook(block_num, block_size, total_size)
                downloaded = max(0, block_num * block_size)
                telemetry.update_file(name, downloaded, total_size if total_size > 0 else None)
                if total_size > 0 and downloaded >= total_size:
                    telemetry.complete_file(name)
                return result

            return hook

        download_module.process_files_def = observed_process_files
        download_module.download_file = observed_download_file
        download_module.create_progress_hook = observed_create_progress_hook
        download_module.__status_lite_observer__ = self
        self.shared_download_available = True
        return True

    def _install_huggingface_progress_wrapper(self) -> bool:
        try:
            from huggingface_hub import file_download as hf_file_download
            hf_tqdm_module = importlib.import_module("huggingface_hub.utils.tqdm")
        except Exception as exc:
            self._record_error(f"Hugging Face progress telemetry disabled: {exc}")
            return False

        current = getattr(hf_file_download, "_get_progress_bar_context", None)
        if not callable(current):
            self._record_error(
                "Hugging Face progress telemetry disabled because _get_progress_bar_context is unavailable"
            )
            return False
        if getattr(current, "__status_lite_wrapped__", False):
            self.huggingface_progress_available = True
            return True
        original_context = current
        telemetry = self.telemetry

        def observed_context(*args: Any, **kwargs: Any):
            existing = kwargs.get("_tqdm_bar")
            if getattr(existing, "__status_lite_proxy__", False):
                return contextlib.nullcontext(existing)
            real_context = original_context(*args, **kwargs)
            name = str(kwargs.get("desc") or "Hugging Face file")
            total = kwargs.get("total")
            initial = kwargs.get("initial", 0)

            @contextlib.contextmanager
            def manager():
                proxy = None
                try:
                    with real_context as progress:
                        proxy = _ProgressProxy(progress, telemetry, name, total, initial)
                        yield proxy
                except Exception as exc:
                    telemetry.fail_file(proxy._record_id if proxy is not None else name, exc)
                    raise
                else:
                    telemetry.complete_file(proxy._record_id if proxy is not None else name)

            return manager()

        observed_context.__status_lite_wrapped__ = True
        observed_context.__status_lite_original__ = original_context
        hf_file_download._get_progress_bar_context = observed_context
        # The imported function keeps the globals of this module, but assigning
        # here also covers callers that resolve it directly from the utility.
        hf_tqdm_module._get_progress_bar_context = observed_context
        self.huggingface_progress_available = True
        return True

    def install(self) -> None:
        if self.installed:
            return
        try:
            self._install_shared_download_wrappers()
        except Exception as exc:
            self._record_error(f"Wan2GP download telemetry disabled after an unexpected error: {exc}")
        try:
            self._install_huggingface_progress_wrapper()
        except Exception as exc:
            self._record_error(f"Hugging Face progress telemetry disabled after an unexpected error: {exc}")
        self.installed = True
        capabilities = []
        if self.shared_download_available:
            capabilities.append("Wan2GP downloads")
        if self.huggingface_progress_available:
            capabilities.append("Hugging Face progress")
        if capabilities:
            print(f"[Status Lite] Download telemetry observer installed ({', '.join(capabilities)})")
        else:
            print("[Status Lite] Download telemetry unavailable; Status Lite will continue without download details")


DOWNLOAD_OBSERVER = DownloadObserver(DOWNLOAD_TELEMETRY)


def install_download_observer() -> None:
    DOWNLOAD_OBSERVER.install()

import json
import os
import threading
import time
import uuid
from functools import wraps
from urllib.parse import unquote

import gradio as gr

try:
    import psutil
except Exception:  # Optional: Status Lite still works without process memory telemetry.
    psutil = None

from shared.utils.plugins import WAN2GPPlugin
from shared.utils import prompt_parser
from .download_telemetry import DOWNLOAD_TELEMETRY, install_download_observer


RUN_SETTING_KEYS = (
    "mode",
    "model_type",
    "base_model_type",
    "model_filename",
    "config",
    "skip_steps_cache_type",
    "skip_steps_multiplier",
    "skip_steps_start_step_perc",
    "image_mode",
    "resolution",
    "num_inference_steps",
    "video_length",
    "duration_seconds",
    "num_frames",
    "frame_num",
    "force_fps",
    "fps",
    "seed",
    "guidance_scale",
    "guidance2_scale",
    "guidance3_scale",
    "guidance_phases",
    "flow_shift",
    "sample_solver",
    "attention_mode",
    "override_attention",
    "attention_sparsity",
    "temporal_upsampling",
    "temporal_upsampling_method",
    "temporal_upsampling_multiplier",
    "spatial_upsampling",
    "spatial_upsampling_method",
    "spatial_upsampling_ratio",
    "film_grain_intensity",
    "film_grain_saturation",
    "activated_loras",
    "loras_multipliers",
    "video_prompt_type",
    "audio_prompt_type",
    "multi_prompts_gen_type",
    "window_no",
    "prompt",
    "negative_prompt",
)

MODEL_WEIGHT_EXTENSIONS = (
    ".safetensors",
    ".gguf",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
)
LTX_POSTPROCESSING_MODEL_TYPES = {
    "ltx23": ("ltx2_22B", "LTX-2 2.3 Pixel Spatial Upscaler"),
    "ltx25": ("ltx2_25_22B_distilled", "LTX-2 2.5 Pixel Spatial Upscaler"),
}
LTX_POSTPROCESSING_STEPS = 8
MAX_STEP_TELEMETRY = 300
_PROCESS = psutil.Process(os.getpid()) if psutil is not None else None


def _ltx_component_fallbacks(model_type):
    """Return accurate role labels when WanGP cannot expose exact resolved files."""
    normalized = str(model_type or "").strip().lower()
    if normalized == "ltx2_25_22b_distilled":
        return {
            "prepare": ["LTX-2.5 22B distilled transformer"],
            "input": ["LTX-2.5 video VAE", "LTX-2.5 audio VAE"],
            "encode": ["Gemma 4 12B LTX v1 text encoder"],
            "decode": ["LTX-2.5 video VAE", "LTX-2.5 audio VAE"],
        }
    if normalized == "ltx2_22b":
        return {
            "prepare": ["LTX-2.3 22B transformer"],
            "input": ["LTX-2.3 video VAE", "LTX-2.3 audio VAE"],
            "encode": ["Gemma 3 12B LTX text encoder"],
            "decode": ["LTX-2.3 video VAE", "LTX-2.3 audio VAE"],
        }
    return {}


def _complete_ltx_components(model_type, components):
    fallbacks = _ltx_component_fallbacks(model_type)
    if not fallbacks:
        return components if isinstance(components, dict) else {}
    completed = dict(components) if isinstance(components, dict) else {}
    for stage, values in fallbacks.items():
        if not completed.get(stage):
            completed[stage] = list(values)
    return completed


class _ModelLifecycleTelemetry:
    """Thread-safe, short-lived model release state for the browser bridge."""

    def __init__(self):
        self._lock = threading.RLock()
        self._event = None

    def begin_unload(self, model_type, model_name):
        token = str(time.time_ns())
        with self._lock:
            self._event = {
                "token": token,
                "state": "unloading",
                "model_type": str(model_type or "")[:300],
                "model_name": str(model_name or model_type or "Previously loaded model")[:500],
                "started_at": time.time(),
                "completed_at": None,
                "error": "",
            }
        return token

    def finish_unload(self, token, error=None):
        with self._lock:
            if not self._event or self._event.get("token") != token:
                return
            self._event["state"] = "failed" if error else "unloaded"
            self._event["completed_at"] = time.time()
            self._event["error"] = str(error or "")[:500]

    def snapshot(self):
        with self._lock:
            event = dict(self._event) if self._event else None
        if not event:
            return None
        completed_at = event.get("completed_at")
        if completed_at is not None and time.time() - float(completed_at) > 5:
            return None
        return event


MODEL_LIFECYCLE_TELEMETRY = _ModelLifecycleTelemetry()


def _telemetry_value(value, depth=0):
    """Return a small JSON-safe representation without copying media payloads."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:8000]
    if depth >= 2:
        return str(value)[:500]
    if isinstance(value, (list, tuple)):
        return [_telemetry_value(item, depth + 1) for item in value[:50]]
    if isinstance(value, dict):
        return {
            str(key)[:100]: _telemetry_value(item, depth + 1)
            for key, item in list(value.items())[:50]
        }
    return str(value)[:500]


def _memory_snapshot(torch_module=None):
    """Return non-synchronizing process and active-device memory counters."""
    sample = {"sampled_at": time.time()}
    if _PROCESS is not None:
        try:
            memory = _PROCESS.memory_info()
            sample["ram_rss_bytes"] = int(memory.rss)
            sample["ram_vms_bytes"] = int(memory.vms)
        except Exception:
            pass

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return sample
    try:
        if not cuda.is_available() or not cuda.is_initialized():
            return sample
        device = int(cuda.current_device())
        sample["gpu_device_index"] = device
        sample["gpu_name"] = str(cuda.get_device_name(device))[:200]
        sample["vram_allocated_bytes"] = int(cuda.memory_allocated(device))
        sample["vram_reserved_bytes"] = int(cuda.memory_reserved(device))
        free_bytes, total_bytes = cuda.mem_get_info(device)
        sample["vram_device_free_bytes"] = int(free_bytes)
        sample["vram_device_total_bytes"] = int(total_bytes)
        sample["vram_device_used_bytes"] = int(total_bytes - free_bytes)
    except Exception:
        pass
    return sample


def _skip_count(pipe):
    cache = getattr(pipe, "cache", None)
    value = getattr(cache, "skipped_steps", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _performance_snapshot(gen):
    source = gen.get("status_pro_performance") if isinstance(gen, dict) else None
    if not isinstance(source, dict):
        return None
    steps = []
    for item in list(source.get("steps") or [])[-MAX_STEP_TELEMETRY:]:
        if isinstance(item, dict):
            steps.append({str(key)[:80]: _telemetry_value(value) for key, value in item.items()})
    return {
        "id": str(source.get("id") or "")[:120],
        "task_id": _telemetry_value(source.get("task_id")),
        "started_at": _telemetry_value(source.get("started_at")),
        "callback_phase": _telemetry_value(source.get("callback_phase", 0)),
        "phase_started_at": _telemetry_value(source.get("phase_started_at")),
        "steps": steps,
        "steps_truncated": bool(source.get("steps_truncated")),
    }


def _latest_performance_snapshot(gen, latest_performance=None):
    """Prefer the newest observer even if WanGP has replaced its mutable gen state."""
    gen_source = gen.get("status_pro_performance") if isinstance(gen, dict) else None
    candidates = [source for source in (gen_source, latest_performance) if isinstance(source, dict)]
    if not candidates:
        return None
    source = max(
        candidates,
        key=lambda value: (
            float(value.get("started_at") or 0),
            len(value.get("steps") or []),
        ),
    )
    return _performance_snapshot({"status_pro_performance": source})


def _new_performance_observer(task_id=None):
    observer = {
        "id": f"{time.time_ns()}",
        "started_at": time.time(),
        "callback_phase": 0,
        "phase_started_at": None,
        "steps": [],
        "steps_truncated": False,
        "_last_step_at": time.perf_counter(),
        "_phase_label": "",
        "_last_step": None,
        "_next_sequence": 0,
    }
    if task_id is not None:
        observer["task_id"] = _telemetry_value(task_id)
    return observer


def _observe_postprocessing_progress(performance, progress_callback, torch_module=None):
    """Capture registered postprocessor progress callbacks without inventing steps."""
    def observed(phase, current_step=None, total_steps=None):
        try:
            current = int(current_step) if current_step is not None else None
            total = int(total_steps) if total_steps is not None else None
        except (TypeError, ValueError):
            current = total = None
        label = str(phase or "").strip()[:300]
        if current is not None and total is not None and current > 0 and total > 0:
            previous_step = performance.get("_last_step")
            previous_label = str(performance.get("_phase_label") or "")
            if label != previous_label or (previous_step is not None and current <= previous_step):
                if previous_label:
                    performance["callback_phase"] = int(performance.get("callback_phase") or 0) + 1
                performance["phase_started_at"] = time.time()
                performance["_phase_label"] = label
                performance["_last_step"] = None
            if current != performance.get("_last_step"):
                now = time.perf_counter()
                performance["_next_sequence"] = int(performance.get("_next_sequence") or 0) + 1
                performance["steps"].append({
                    "sequence": performance["_next_sequence"],
                    "phase": int(performance.get("callback_phase") or 0),
                    "step": current,
                    "total_steps": total,
                    "pass_no": -1,
                    "duration_seconds": round(max(0.0, now - float(performance.get("_last_step_at") or now)), 4),
                    "skip_method": None,
                    "skipped": None,
                    "skipped_delta": None,
                    "skipped_total": None,
                    "completed_at": time.time(),
                    "memory": _memory_snapshot(torch_module),
                    "label": label,
                })
                if len(performance["steps"]) > MAX_STEP_TELEMETRY:
                    performance["steps"] = performance["steps"][-MAX_STEP_TELEMETRY:]
                    performance["steps_truncated"] = True
                performance["_last_step_at"] = now
                performance["_last_step"] = current
        return progress_callback(phase, current_step, total_steps)

    return observed


def _component_filename(value):
    """Reduce a local path or download URL to its display-safe filename."""
    if not isinstance(value, str) or not value.strip():
        return ""
    clean = value.strip().split("|", 1)[0].split("?", 1)[0].split("#", 1)[0]
    clean = clean.replace("\\", "/").rstrip("/")
    return unquote(clean.rsplit("/", 1)[-1])[:500]


def _component_filenames(values):
    names = []
    seen = set()

    def visit(value, depth=0):
        if depth > 5:
            return
        if isinstance(value, str):
            name = _component_filename(value)
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, depth + 1)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, depth + 1)

    visit(values)
    return names


def _vae_weight_names(values, include_upsamplers=False):
    names = []
    for name in _component_filenames(values):
        lowered = name.lower()
        if not lowered.endswith(MODEL_WEIGHT_EXTENSIONS):
            continue
        if "vae" not in lowered and "autoencoder" not in lowered:
            continue
        if not include_upsamplers and ("upscale" in lowered or "upsampler" in lowered):
            continue
        names.append(name)
    return names


def _model_components(
    settings,
    get_model_def=None,
    get_base_model_type=None,
    get_model_handler=None,
    get_model_config_groups=None,
    model_config_groups=None,
    get_model_recursive_prop=None,
    get_model_filename=None,
    transformer_quantization="",
    transformer_dtype_policy="",
    text_encoder_quantization="",
):
    """Resolve stage component filenames without exposing local paths or URLs."""
    if not isinstance(settings, dict):
        return {}
    model_type = settings.get("model_type") or settings.get("base_model_type")
    if not model_type:
        return {}

    model_def = None
    if callable(get_model_def):
        try:
            model_def = get_model_def(model_type)
        except Exception:
            pass
    model_def = model_def if isinstance(model_def, dict) else {}
    config_id = settings.get("config")
    if config_id and callable(get_model_config_groups) and model_config_groups is not None:
        select_configs = getattr(model_config_groups, "selected_model_configs", None)
        if callable(select_configs):
            try:
                resolved_def = model_def.copy()
                groups = get_model_config_groups(model_type, model_def)
                for _, _, selected_config in select_configs(groups, config_id):
                    if isinstance(selected_config, dict):
                        resolved_def.update(selected_config)
                model_def = resolved_def
            except Exception:
                pass

    prepare = _component_filenames(settings.get("model_filename"))
    if not prepare and callable(get_model_filename):
        try:
            prepare = _component_filenames(get_model_filename(
                model_type=model_type,
                quantization=str(transformer_quantization or ""),
                dtype_policy=transformer_dtype_policy or "",
                model_def=model_def,
            ))
        except Exception:
            pass

    encode = []
    if callable(get_model_recursive_prop) and callable(get_model_filename):
        try:
            encoder_urls = get_model_recursive_prop(
                model_type,
                "text_encoder_URLs",
                return_list=True,
                model_def=model_def,
            )
            if encoder_urls:
                encode = _component_filenames(get_model_filename(
                    model_type=model_type,
                    quantization=str(text_encoder_quantization or ""),
                    dtype_policy=transformer_dtype_policy or "",
                    URLs=encoder_urls,
                ))
        except Exception:
            pass

    spatial_upsampling = str(settings.get("spatial_upsampling") or "").strip()
    include_upsamplers = bool(spatial_upsampling)
    generic_vae_overrides = []
    for key in ("VAE_URLs", "vae_URLs", "vae_URL"):
        if model_def.get(key):
            generic_vae_overrides.extend(_vae_weight_names(
                model_def.get(key),
                include_upsamplers=include_upsamplers,
            ))

    decode = []
    if generic_vae_overrides:
        decode.extend(generic_vae_overrides)
    else:
        handler = None
        base_model_type = settings.get("base_model_type")
        if not base_model_type and callable(get_base_model_type):
            try:
                base_model_type = get_base_model_type(model_type)
            except Exception:
                pass
        if callable(get_model_handler):
            try:
                handler = get_model_handler(model_type)
            except Exception:
                pass
        query_files = getattr(handler, "query_model_files", None)
        if callable(query_files) and base_model_type:
            try:
                download_defs = query_files([], base_model_type, model_def=model_def)
            except TypeError:
                try:
                    download_defs = query_files([], base_model_type)
                except Exception:
                    download_defs = []
            except Exception:
                download_defs = []
            for download_def in download_defs or []:
                if isinstance(download_def, dict):
                    decode.extend(_vae_weight_names(
                        download_def.get("fileList"),
                        include_upsamplers=include_upsamplers,
                    ))

        for key in (
            "video_vae_file",
            "audio_vae_file",
            "ltx2_video_vae_file",
            "vae_file",
            "vae_filename",
        ):
            decode.extend(_vae_weight_names(
                model_def.get(key),
                include_upsamplers=include_upsamplers,
            ))

    components = {}
    for stage, values in (
        ("prepare", prepare),
        ("input", decode),
        ("encode", encode),
        ("decode", decode),
    ):
        unique = []
        seen = set()
        for value in values:
            key = value.lower()
            if key not in seen:
                seen.add(key)
                unique.append(value)
        if unique:
            components[stage] = unique
    return _complete_ltx_components(model_type, components)


def _normalize_late_postprocessing_settings(settings):
    """Replace generation-form state with the model a standalone edit actually uses."""
    if not isinstance(settings, dict):
        return settings
    mode = str(settings.get("mode") or "").strip().lower()
    if mode not in {"edit_postprocessing", "edit_remux", "edit_audio"}:
        return settings

    # WanGP's late-edit forms can retain whichever generation model is selected in
    # the main UI. It is not used by the edit task and must not become the model
    # reported by the live panel. Model-backed processors map to what they load.
    for key in (
        "model_type",
        "base_model_type",
        "model_filename",
        "config",
        "model_name",
        "model_family",
        "component_models",
        "activated_loras",
        "loras_multipliers",
        "attention_mode",
        "override_attention",
        "attention_sparsity",
    ):
        settings.pop(key, None)

    effective_model_type = None
    if mode == "edit_postprocessing":
        spatial_upsampling = str(settings.get("spatial_upsampling") or "").strip().lower()
        for method, (model_type, model_name) in LTX_POSTPROCESSING_MODEL_TYPES.items():
            if spatial_upsampling.startswith(method):
                effective_model_type = model_type
                settings["model_name"] = model_name
                settings["model_family"] = "LTX-2"
                break

    if effective_model_type:
        settings["model_type"] = effective_model_type
        settings["num_inference_steps"] = LTX_POSTPROCESSING_STEPS
    return settings


def _fallback_postprocessing_label(value, methods):
    text = str(value or "").strip()
    lowered = text.lower()
    for method, label in sorted(methods.items(), key=lambda item: len(item[0]), reverse=True):
        if not lowered.startswith(method):
            continue
        multiplier = lowered[len(method):]
        try:
            scale = float(multiplier)
        except (TypeError, ValueError):
            scale = None
        if scale is not None:
            scale_text = str(int(scale)) if scale.is_integer() else f"{scale:g}"
            return f"{label} x{scale_text}"
        return label
    return text


def _postprocessing_metadata(settings):
    """Describe post-processing independently from the generation model."""
    if not isinstance(settings, dict):
        return None
    operations = []
    temporal = str(settings.get("temporal_upsampling") or "").strip()
    if temporal:
        temporal_details = {}
        try:
            from postprocessing import temporal_upsamplers as temporal_api
            temporal_label = temporal_api.format_temporal_upsampling_label(temporal)
            temporal_handler = temporal_api.find_temporal_upsampler(temporal)
            temporal_split = temporal_handler.split_value(temporal) if temporal_handler is not None else None
            if temporal_handler is not None:
                temporal_details["processor"] = str(
                    temporal_handler.query_temporal_upsampler_def().get("name") or temporal_handler.__class__.__name__
                )[:300]
            if temporal_split is not None:
                temporal_details["method"] = str(temporal_split[0])[:200]
                temporal_details["scale"] = float(temporal_split[1])
        except Exception:
            temporal_label = _fallback_postprocessing_label(temporal, {"rife": "RIFE"})
        operation = {
            "kind": "temporal_upsampling",
            "label": str(temporal_label or temporal)[:500],
            "value": temporal[:300],
        }
        operation.update(temporal_details)
        operations.append(operation)

    spatial = str(settings.get("spatial_upsampling") or "").strip()
    if spatial:
        spatial_details = {}
        try:
            from postprocessing import spatial_upsamplers as spatial_api
            spatial_label = spatial_api.format_upsampling_label(spatial)
            spatial_handler = spatial_api.find_upsampler(spatial)
            spatial_split = spatial_handler.split_value(spatial) if spatial_handler is not None else None
            if spatial_handler is not None:
                spatial_details["processor"] = str(
                    spatial_handler.query_upsampler_def().get("name") or spatial_handler.__class__.__name__
                )[:300]
            if spatial_split is not None:
                spatial_details["method"] = str(spatial_split[0])[:200]
                spatial_details["scale"] = float(spatial_split[1])
        except Exception:
            spatial_label = _fallback_postprocessing_label(spatial, {
                "flashvsr2pass": "FlashVSR Two Pass",
                "flashvsr": "FlashVSR",
                "seedvr2": "SeedVR2",
                "lanczos": "Lanczos",
                "ltx25": "LTX 2.5 Pixel Spatial Upscaler",
                "ltx23": "LTX 2.3 Pixel Spatial Upscaler",
                "coz": "Chain of Zoom",
            })
        operation = {
            "kind": "spatial_upsampling",
            "label": str(spatial_label or spatial)[:500],
            "value": spatial[:300],
        }
        operation.update(spatial_details)
        lowered_spatial = spatial.lower()
        for method, (model_type, model_name) in LTX_POSTPROCESSING_MODEL_TYPES.items():
            if lowered_spatial.startswith(method):
                operation["model"] = {"model_type": model_type, "model_name": model_name}
                operation["steps"] = LTX_POSTPROCESSING_STEPS
                break
        operations.append(operation)

    intensity = settings.get("film_grain_intensity")
    try:
        intensity_value = float(intensity)
    except (TypeError, ValueError):
        intensity_value = 0
    if intensity_value > 0:
        saturation = settings.get("film_grain_saturation")
        try:
            saturation_value = float(saturation)
        except (TypeError, ValueError):
            saturation_value = None
        intensity_label = f"{intensity_value:g}"
        label = f"Film grain (intensity {intensity_label}"
        if saturation_value is not None:
            label += f", saturation {saturation_value:g}"
        label += ")"
        operation = {
            "kind": "film_grain",
            "label": label,
            "intensity": intensity_value,
        }
        if saturation_value is not None:
            operation["saturation"] = saturation_value
        operations.append(operation)

    if not operations:
        return None
    mode = str(settings.get("mode") or "").strip().lower()
    return {
        "application": "late" if mode == "edit_postprocessing" else "inline",
        "summary": " · ".join(operation["label"] for operation in operations)[:1500],
        "operations": operations,
    }


def _task_telemetry(
    task,
    get_model_name=None,
    get_model_family=None,
    families_infos=None,
    component_resolver=None,
    attention_mode=None,
    get_overridden_attention=None,
    get_auto_attention=None,
):
    if not isinstance(task, dict):
        return None
    params = task.get("params") if isinstance(task.get("params"), dict) else {}
    settings = {
        key: _telemetry_value(params.get(key))
        for key in RUN_SETTING_KEYS
        if key in params and params.get(key) is not None
    }
    settings.setdefault("num_inference_steps", _telemetry_value(task.get("steps")))
    settings.setdefault("video_length", _telemetry_value(task.get("length")))
    if "prompt" not in settings and task.get("prompt") is not None:
        settings["prompt"] = _telemetry_value(task.get("prompt"))
    _normalize_late_postprocessing_settings(settings)
    postprocessing = _postprocessing_metadata(settings)
    if postprocessing:
        settings["postprocessing"] = postprocessing
    model_type = settings.get("model_type") or settings.get("base_model_type")
    effective_attention = settings.get("override_attention") or settings.get("attention_mode")
    if not effective_attention and model_type and callable(get_overridden_attention):
        try:
            effective_attention = get_overridden_attention(model_type)
        except Exception:
            pass
    if not effective_attention and model_type:
        effective_attention = attention_mode
    if str(effective_attention or "").strip().lower() == "auto" and callable(get_auto_attention):
        try:
            effective_attention = get_auto_attention()
        except Exception:
            pass
    if effective_attention:
        settings["attention_mode"] = _telemetry_value(effective_attention)
    if str(effective_attention or "").strip().lower() != "sol":
        settings.pop("attention_sparsity", None)
    settings.pop("override_attention", None)
    if model_type and not settings.get("model_name") and callable(get_model_name):
        try:
            settings["model_name"] = _telemetry_value(get_model_name(model_type))
        except Exception:
            pass
    if model_type and not settings.get("model_family") and callable(get_model_family) and isinstance(families_infos, dict):
        try:
            family_key = get_model_family(model_type, for_ui=True)
            family_info = families_infos.get(family_key)
            if isinstance(family_info, (list, tuple)) and len(family_info) > 1:
                settings["model_family"] = _telemetry_value(family_info[1])
        except Exception:
            pass
    resolved_component_models = {}
    if callable(component_resolver):
        try:
            resolved_component_models = component_resolver(settings) or {}
            resolved_component_models = _complete_ltx_components(model_type, resolved_component_models)
            if resolved_component_models:
                settings["component_models"] = _telemetry_value(resolved_component_models)
        except Exception:
            pass
    if postprocessing:
        for operation in postprocessing.get("operations", []):
            backing_model = operation.get("model") if isinstance(operation, dict) else None
            backing_model_type = backing_model.get("model_type") if isinstance(backing_model, dict) else None
            if not backing_model_type:
                continue
            backing_components = resolved_component_models if backing_model_type == model_type else {}
            if not backing_components and callable(component_resolver):
                try:
                    backing_components = component_resolver({"model_type": backing_model_type}) or {}
                except Exception:
                    backing_components = {}
            backing_components = _complete_ltx_components(backing_model_type, backing_components)
            if backing_components:
                backing_model["component_models"] = _telemetry_value(backing_components)
    return {
        "id": _telemetry_value(task.get("id")),
        "client_id": str(params.get("client_id") or "")[:200],
        "repeats": _telemetry_value(task.get("repeats", 1)),
        "settings": settings,
    }


def _window_prompt(prompt, multi_prompts_gen_type, window_no):
    if not isinstance(prompt, str) or not isinstance(window_no, int) or window_no < 1:
        return None
    mode = prompt_parser.normalize_multi_prompts_mode(multi_prompts_gen_type)
    if "W" not in mode:
        return None
    prompts = prompt_parser.split_prompt_units(prompt, mode)
    if not prompts:
        return None
    return prompts[min(window_no - 1, len(prompts) - 1)]


def _window_prompts(prompt, multi_prompts_gen_type):
    if not isinstance(prompt, str):
        return []
    mode = prompt_parser.normalize_multi_prompts_mode(multi_prompts_gen_type)
    if "W" not in mode:
        return []
    return [_telemetry_value(value) for value in prompt_parser.split_prompt_units(prompt, mode)]


class StatusLitePlugin(WAN2GPPlugin):
    """A live, browser-side presentation for Wan2GP generation progress."""

    def __init__(self):
        super().__init__()
        self.name = "Status Lite"
        self.version = "1.0.0"
        self.description = (
            "History-free pipeline timeline with stage timings and live ETA estimates."
        )
        self._runtime_id = str(uuid.uuid4())
        self._insertion_registered = False
        self._step_observer_installed = False
        self._postprocessing_step_observer_installed = False
        self._latest_performance = None
        self._active_task_id = None
        self._model_lifecycle_observer_installed = False

    def setup_ui(self):
        try:
            install_download_observer()
        except Exception as exc:
            # Download detail is optional. A changed Wan2GP/Hugging Face API
            # must not prevent the core live status UI from loading.
            print(f"[Status Lite] Download telemetry unavailable: {exc}")
        self.request_component("gen_status")
        self.request_component("state")
        self.request_global("get_model_name")
        self.request_global("get_model_family")
        self.request_global("families_infos")
        self.request_global("get_model_def")
        self.request_global("get_base_model_type")
        self.request_global("get_model_handler")
        self.request_global("get_model_config_groups")
        self.request_global("model_config_groups")
        self.request_global("get_model_recursive_prop")
        self.request_global("get_model_filename")
        self.request_global("transformer_quantization")
        self.request_global("transformer_dtype_policy")
        self.request_global("text_encoder_quantization")
        self.request_global("attention_mode")
        self.request_global("get_overridden_attention")
        self.request_global("get_auto_attention")
        self.request_global("build_callback")
        self.request_global("perform_spatial_upsampling")
        self.request_global("release_model")
        self.request_global("torch")
        self.add_custom_js(self._javascript())

    def _install_model_lifecycle_observer(self):
        if self._model_lifecycle_observer_installed:
            return
        original_release = getattr(self, "release_model", None)
        if not callable(original_release):
            return
        if getattr(original_release, "_status_lite_model_lifecycle_observer", False):
            self._model_lifecycle_observer_installed = True
            return

        @wraps(original_release)
        def observed_release(*args, **kwargs):
            global_values = getattr(original_release, "__globals__", {})
            model_type = global_values.get("transformer_type")
            has_loaded_model = (
                global_values.get("wan_model") is not None
                or global_values.get("offloadobj") is not None
            )
            if not has_loaded_model:
                return original_release(*args, **kwargs)

            model_name = model_type
            resolver = getattr(self, "get_model_name", None)
            if model_type and callable(resolver):
                try:
                    model_name = resolver(model_type)
                except Exception:
                    pass
            token = MODEL_LIFECYCLE_TELEMETRY.begin_unload(model_type, model_name)
            try:
                result = original_release(*args, **kwargs)
            except Exception as exc:
                MODEL_LIFECYCLE_TELEMETRY.finish_unload(token, error=exc)
                raise
            MODEL_LIFECYCLE_TELEMETRY.finish_unload(token)
            return result

        observed_release._status_lite_model_lifecycle_observer = True
        self.set_global("release_model", observed_release)
        self.release_model = observed_release
        self._model_lifecycle_observer_installed = True

    def _install_step_observer(self):
        if self._step_observer_installed:
            return
        original_builder = getattr(self, "build_callback", None)
        if not callable(original_builder):
            return
        if getattr(original_builder, "_status_lite_step_observer", False):
            self._step_observer_installed = True
            return
        torch_module = getattr(self, "torch", None)

        @wraps(original_builder)
        def observed_builder(state, pipe, *args, **kwargs):
            callback = original_builder(state, pipe, *args, **kwargs)
            state = state if isinstance(state, dict) else {}
            gen = state.get("gen") if isinstance(state.get("gen"), dict) else {}
            queue = gen.get("queue") if isinstance(gen.get("queue"), list) else []
            task_id = queue[0].get("id") if queue and isinstance(queue[0], dict) else None
            observer_id = f"{time.time_ns()}"
            performance = _new_performance_observer(task_id)
            performance["id"] = observer_id
            gen["status_pro_performance"] = performance
            self._latest_performance = performance
            last_step_at = time.perf_counter()
            last_skip_count = _skip_count(pipe)
            phase_index = 0
            next_sequence = 0
            default_total = kwargs.get("num_inference_steps")
            if default_total is None and len(args) >= 3:
                default_total = args[2]
            try:
                current_total = int(default_total) if default_total is not None and int(default_total) > 0 else None
            except (TypeError, ValueError):
                current_total = None

            @wraps(callback)
            def observed_callback(*callback_args, **callback_kwargs):
                nonlocal last_step_at, last_skip_count, phase_index, next_sequence, current_total
                step_idx = callback_kwargs.get("step_idx", callback_args[0] if callback_args else -1)
                force_refresh = callback_kwargs.get(
                    "force_refresh",
                    callback_args[2] if len(callback_args) > 2 else True,
                )
                try:
                    step_idx = int(step_idx)
                except (TypeError, ValueError):
                    step_idx = -1

                override_total = callback_kwargs.get(
                    "override_num_inference_steps",
                    callback_args[4] if len(callback_args) > 4 else None,
                )
                try:
                    override_total = int(override_total) if override_total is not None and int(override_total) > 0 else None
                except (TypeError, ValueError):
                    override_total = None
                if override_total is not None:
                    current_total = override_total

                now = time.perf_counter()
                if step_idx >= 0:
                    current_skip_count = _skip_count(pipe)
                    skip_delta = None
                    if current_skip_count is not None and last_skip_count is not None:
                        skip_delta = max(0, current_skip_count - last_skip_count)
                    cache = getattr(pipe, "cache", None)
                    next_sequence += 1
                    sample = {
                        "sequence": next_sequence,
                        "phase": phase_index,
                        "step": step_idx + 1,
                        "total_steps": current_total,
                        "pass_no": _telemetry_value(callback_kwargs.get("pass_no", -1)),
                        "duration_seconds": round(max(0.0, now - last_step_at), 4),
                        "skip_method": _telemetry_value(getattr(cache, "cache_type", None)),
                        "skipped": bool(skip_delta) if skip_delta is not None else None,
                        "skipped_delta": skip_delta,
                        "skipped_total": current_skip_count,
                        "completed_at": time.time(),
                        "memory": _memory_snapshot(torch_module),
                    }
                    extra = callback_kwargs.get("denoising_extra")
                    if extra:
                        sample["label"] = str(extra)[:300]
                    performance["steps"].append(sample)
                    if len(performance["steps"]) > MAX_STEP_TELEMETRY:
                        performance["steps"] = performance["steps"][-MAX_STEP_TELEMETRY:]
                        performance["steps_truncated"] = True
                    last_step_at = now
                    last_skip_count = current_skip_count
                elif force_refresh:
                    phase_index += 1
                    performance["callback_phase"] = phase_index
                    performance["phase_started_at"] = time.time()
                    last_step_at = now
                    last_skip_count = _skip_count(pipe)
                return callback(*callback_args, **callback_kwargs)

            return observed_callback

        observed_builder._status_lite_step_observer = True
        self.set_global("build_callback", observed_builder)
        self._step_observer_installed = True

    def _install_postprocessing_step_observer(self):
        if self._postprocessing_step_observer_installed:
            return
        original = getattr(self, "perform_spatial_upsampling", None)
        if not callable(original):
            return
        if getattr(original, "_status_lite_postprocessing_step_observer", False):
            self._postprocessing_step_observer_installed = True
            return
        torch_module = getattr(self, "torch", None)

        @wraps(original)
        def observed(*args, **kwargs):
            spatial_upsampling = kwargs.get(
                "spatial_upsampling",
                args[1] if len(args) > 1 else "",
            )
            progress_callback = kwargs.get("progress_callback")
            lowered = str(spatial_upsampling or "").strip().lower()
            if callable(progress_callback) and lowered.startswith(tuple(LTX_POSTPROCESSING_MODEL_TYPES)):
                performance = _new_performance_observer(self._active_task_id)
                self._latest_performance = performance
                kwargs["progress_callback"] = _observe_postprocessing_progress(
                    performance,
                    progress_callback,
                    torch_module,
                )
            return original(*args, **kwargs)

        observed._status_lite_postprocessing_step_observer = True
        self.set_global("perform_spatial_upsampling", observed)
        self.perform_spatial_upsampling = observed
        self._postprocessing_step_observer_installed = True

    def _run_snapshot_json(self, state):
        try:
            state = state if isinstance(state, dict) else {}
            gen = state.get("gen") if isinstance(state.get("gen"), dict) else {}
            queue = gen.get("queue") if isinstance(gen.get("queue"), list) else []
            if gen.get("in_progress") and queue and isinstance(queue[0], dict):
                self._active_task_id = queue[0].get("id")
            active_task = _task_telemetry(
                queue[0],
                get_model_name=getattr(self, "get_model_name", None),
                get_model_family=getattr(self, "get_model_family", None),
                families_infos=getattr(self, "families_infos", None),
                attention_mode=getattr(self, "attention_mode", None),
                get_overridden_attention=getattr(self, "get_overridden_attention", None),
                get_auto_attention=getattr(self, "get_auto_attention", None),
                component_resolver=lambda settings: _model_components(
                    settings,
                    get_model_def=getattr(self, "get_model_def", None),
                    get_base_model_type=getattr(self, "get_base_model_type", None),
                    get_model_handler=getattr(self, "get_model_handler", None),
                    get_model_config_groups=getattr(self, "get_model_config_groups", None),
                    model_config_groups=getattr(self, "model_config_groups", None),
                    get_model_recursive_prop=getattr(self, "get_model_recursive_prop", None),
                    get_model_filename=getattr(self, "get_model_filename", None),
                    transformer_quantization=getattr(self, "transformer_quantization", ""),
                    transformer_dtype_policy=getattr(self, "transformer_dtype_policy", ""),
                    text_encoder_quantization=getattr(self, "text_encoder_quantization", ""),
                ),
            ) if gen.get("in_progress") and queue else None
            if active_task and gen.get("sliding_window"):
                window_prompts = _window_prompts(
                    active_task["settings"].get("prompt"),
                    active_task["settings"].get("multi_prompts_gen_type"),
                )
                if window_prompts:
                    active_task["window_prompts"] = window_prompts
                window_prompt = _window_prompt(
                    active_task["settings"].get("prompt"),
                    active_task["settings"].get("multi_prompts_gen_type"),
                    gen.get("window_no"),
                )
                if window_prompt:
                    active_task["window_prompt"] = _telemetry_value(window_prompt)
            payload = {
                "server_time": time.time(),
                "runtime_id": self._runtime_id,
                "in_progress": bool(gen.get("in_progress")),
                "queue_length": len(queue),
                "queue_task_ids": [_telemetry_value(task.get("id")) for task in queue if isinstance(task, dict)],
                "active_task": active_task,
                "sliding_window": bool(gen.get("sliding_window")),
                "window_no": _telemetry_value(gen.get("window_no")),
                "total_windows": _telemetry_value(gen.get("total_windows")),
                "status": str(gen.get("status") or "")[:2000],
                "progress_phase": _telemetry_value(gen.get("progress_phase")),
                "queue_errors": _telemetry_value(gen.get("queue_errors") or {}),
                "resource_sample": _memory_snapshot(getattr(self, "torch", None)) if gen.get("in_progress") else None,
                "performance": _latest_performance_snapshot(gen, self._latest_performance),
                "model_lifecycle": MODEL_LIFECYCLE_TELEMETRY.snapshot(),
            }
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            return json.dumps(
                {"server_time": time.time(), "runtime_id": self._runtime_id, "error": str(exc)[:500]},
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def post_ui_setup(self, components: dict):
        if self._insertion_registered or components.get("gen_status") is None:
            return

        self._install_step_observer()
        self._install_postprocessing_step_observer()
        self._install_model_lifecycle_observer()

        state_component = components.get("state")

        def create_status_lite_host():
            with gr.Column(elem_id="status-lite-container") as container:
                gr.HTML(
                    value=self._markup(),
                    elem_id="status-lite-host",
                    show_label=False,
                )
                download_bridge = gr.Textbox(
                    value="{}",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="status-lite-download-bridge",
                    elem_classes=["status-lite-download-bridge"],
                )
                run_bridge = gr.Textbox(
                    value=json.dumps(
                        {"server_time": time.time(), "runtime_id": self._runtime_id},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="status-lite-run-bridge",
                    elem_classes=["status-lite-run-bridge"],
                )
                download_timer = gr.Timer(value=0.5, active=True)
                download_timer.tick(
                    fn=DOWNLOAD_TELEMETRY.snapshot_json,
                    inputs=None,
                    outputs=[download_bridge],
                    queue=False,
                    show_progress="hidden",
                    api_name=False,
                    show_api=False,
                    trigger_mode="always_last",
                )
                if state_component is not None:
                    run_timer = gr.Timer(value=0.5, active=True)
                    run_timer.tick(
                        fn=self._run_snapshot_json,
                        inputs=[state_component],
                        outputs=[run_bridge],
                        queue=False,
                        show_progress="hidden",
                        api_name=False,
                        show_api=False,
                        trigger_mode="always_last",
                    )
            return container

        self.insert_after(
            target_component_id="gen_status",
            new_component_constructor=create_status_lite_host,
        )
        self._insertion_registered = True

    @staticmethod
    def _markup() -> str:
        return """
<section class="status-lite" data-status-lite hidden aria-label="Generation status">
  <header class="status-lite__header">
    <div class="status-lite__heading">
      <span class="status-lite__badge">Status Lite</span>
      <span class="status-lite__live" data-sp-live>Waiting for progress</span>
    </div>
    <div class="status-lite__header-actions">
      <div class="status-lite__summary" aria-live="polite">
        <span data-sp-steps></span>
        <span data-sp-overall></span>
        <span data-sp-eta></span>
      </div>
      <button class="status-lite__collapse" data-sp-collapse type="button" aria-expanded="true" title="Collapse Status Lite">▼</button>
    </div>
  </header>
  <div class="status-lite__body" data-sp-body>
    <section class="status-lite__idle" data-sp-idle hidden aria-live="polite">
      <div class="status-lite__idle-copy">
        <strong data-sp-idle-title>Ready</strong>
        <span data-sp-idle-message>Live generation timing will appear here.</span>
      </div>
    </section>
    <div data-sp-running>
    <div class="status-lite__stages" data-sp-stages role="tablist" aria-label="Generation stages"></div>
    <section class="status-lite__downloads" data-sp-downloads hidden aria-label="Model downloads">
    <div class="status-lite__downloads-header">
      <div class="status-lite__downloads-title">
        <span class="status-lite__download-indicator" aria-hidden="true">↓</span>
        <div>
          <strong data-sp-download-title>Downloading model files</strong>
          <span data-sp-download-summary>Preparing download information…</span>
        </div>
      </div>
      <span class="status-lite__download-total" data-sp-download-total></span>
    </div>
    <div class="status-lite__download-overall" aria-hidden="true">
      <div data-sp-download-overall-fill></div>
    </div>
    <div class="status-lite__download-files" data-sp-download-files></div>
    </section>
    <div class="status-lite__detail" data-sp-detail role="tabpanel" aria-live="polite">
      <div class="status-lite__detail-copy">
        <strong data-sp-detail-name>Preparing</strong>
        <span class="status-lite__detail-activities" data-sp-detail-activities hidden></span>
        <span class="status-lite__detail-model" data-sp-detail-model hidden></span>
        <span data-sp-detail-message>Waiting for generation progress.</span>
      </div>
      <dl class="status-lite__metrics">
        <div><dt>Status</dt><dd data-sp-detail-state>Pending</dd></div>
        <div><dt>Elapsed</dt><dd data-sp-detail-elapsed>—</dd></div>
        <div data-sp-eta-metric><dt>Expected left</dt><dd data-sp-detail-eta>—</dd></div>
        <div data-sp-progress-metric><dt>Progress</dt><dd data-sp-detail-progress>—</dd></div>
        <div data-sp-step-metric hidden><dt>Avg step time</dt><dd data-sp-detail-step-time>—</dd></div>
      </dl>
    </div>
    <div class="status-lite__overall-track" aria-hidden="true">
      <div class="status-lite__overall-fill" data-sp-overall-fill></div>
    </div>
    </div>
  </div>
</section>
"""

    @staticmethod
    def _javascript() -> str:
        return r"""
(function () {
    const NAMESPACE = "__wangpStatusLite";
    const COLLAPSED_KEY = "wangp.status-lite.collapsed.v1";
    const MAX_STEP_RECORDS = 300;
    const TICK_MS = 250;
    const IDLE_GRACE_MS = 1600;
    const RESET_AFTER_MS = 3000;

    const STAGE_DEFS = [
        { id: "prepare", label: "Prepare" },
        { id: "input", label: "Inputs", optional: true },
        { id: "encode", label: "Encode" },
        { id: "denoise", label: "Generate" },
        { id: "decode", label: "Decode" },
        { id: "post", label: "Enhance", optional: true },
        { id: "save", label: "Save", optional: true }
    ];

    const STYLE_TEXT = `
#status-lite-container {
    display: none;
    min-width: 0;
}
#status-lite-container.status-lite-container--active {
    display: flex;
}
#status-lite-host {
    display: none;
    min-width: 0;
}
#status-lite-host.status-lite-host--active {
    display: block;
}
.status-lite-source--active {
    display: none !important;
}
#status-lite-download-bridge,
.status-lite-download-bridge,
#status-lite-run-bridge,
.status-lite-run-bridge {
    display: none !important;
}
.status-lite {
    --sp-accent: var(--color-accent, var(--primary-500, #0ea5e9));
    --sp-accent-strong: var(--primary-600, #0284c7);
    --sp-good: #22c55e;
    --sp-muted: var(--body-text-color-subdued, #94a3b8);
    --sp-text: var(--body-text-color, #f8fafc);
    --sp-panel: var(--block-background-fill, #1e293b);
    --sp-panel-soft: var(--background-fill-secondary, #0f172a);
    --sp-border: var(--border-color-primary, #334155);
    box-sizing: border-box;
    container-name: status-lite;
    container-type: inline-size;
    width: 100%;
    padding: 12px;
    border: 1px solid var(--sp-border);
    border-radius: var(--block-radius, 8px);
    background: var(--sp-panel);
    color: var(--body-text-color, #f8fafc);
    box-shadow: var(--block-shadow, none);
}
.status-lite *, .status-lite *::before, .status-lite *::after { box-sizing: border-box; }
.status-lite__header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
}
.status-lite__heading, .status-lite__summary, .status-lite__header-actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.status-lite__header-actions {
    justify-content: flex-end;
    flex-wrap: nowrap;
    margin-left: auto;
}
.status-lite__collapse {
    display: inline-grid;
    place-items: center;
    width: 28px;
    height: 28px;
    flex: 0 0 auto;
    padding: 0;
    border: 1px solid var(--sp-border);
    border-radius: 7px;
    background: var(--sp-panel-soft);
    color: var(--sp-muted);
    font-size: 1rem;
    line-height: 1;
    cursor: pointer;
    transition: border-color 160ms ease, color 160ms ease, background-color 160ms ease;
}
.status-lite__collapse:hover {
    border-color: var(--sp-accent);
    color: var(--body-text-color, #f8fafc);
}
.status-lite__collapse:focus-visible { outline: 2px solid var(--sp-accent); outline-offset: 2px; }
.status-lite__body[hidden] { display: none !important; }
.status-lite__body [hidden] { display: none !important; }
.status-lite--collapsed .status-lite__header { margin-bottom: 0; }
.status-lite__idle {
    display: grid;
    gap: 10px;
}
.status-lite__idle-copy {
    display: grid;
    gap: 2px;
    padding: 2px 1px;
}
.status-lite__idle-copy strong { font-size: .9rem; }
.status-lite__idle-copy span {
    color: var(--sp-muted);
    font-size: .75rem;
}
.status-lite__badge {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 3px 9px;
    border-radius: 7px;
    background: var(--sp-accent-strong);
    color: white;
    font-weight: 700;
    letter-spacing: .01em;
}
.status-lite__live { font-weight: 650; }
.status-lite__summary {
    justify-content: flex-end;
    color: var(--sp-muted);
    font-size: .82rem;
    font-variant-numeric: tabular-nums;
}
.status-lite__summary span:not(:empty) + span:not(:empty)::before {
    content: "•";
    margin-right: 8px;
    opacity: .55;
}
.status-lite__stages {
    display: flex;
    align-items: stretch;
    justify-content: center;
    justify-content: safe center;
    gap: 7px;
    width: 100%;
    overflow-x: auto;
    padding: 1px 1px 5px;
    scrollbar-width: thin;
}
.status-lite__stage {
    position: relative;
    display: grid;
    grid-template-columns: 24px minmax(0, 1fr);
    grid-template-areas: "icon name" "icon time";
    grid-template-rows: auto auto;
    column-gap: 9px;
    row-gap: 3px;
    align-content: center;
    align-items: center;
    flex: 1 1 150px;
    max-width: 260px;
    min-width: 108px;
    min-height: 52px;
    padding: 7px 10px;
    overflow: hidden;
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    background: var(--sp-panel-soft);
    color: inherit;
    text-align: left;
    cursor: pointer;
    opacity: .68;
    transition: flex-grow 280ms ease, min-width 280ms ease, opacity 180ms ease,
                border-color 180ms ease, background-color 180ms ease;
}
.status-lite__stages--inline .status-lite__stage {
    grid-template-columns: 24px minmax(0, auto) minmax(0, auto);
    grid-template-areas: "icon name time";
    grid-template-rows: auto;
    justify-content: center;
    min-height: 44px;
}
.status-lite__stage:hover { opacity: .92; }
.status-lite__stage:focus-visible { outline: 2px solid var(--sp-accent); outline-offset: 2px; }
.status-lite__stage--complete { opacity: .82; }
.status-lite__stage--current {
    flex-grow: 1.7;
    flex-basis: 230px;
    max-width: 420px;
    min-width: 178px;
    opacity: 1;
    border-color: var(--sp-accent);
    background: color-mix(in srgb, var(--sp-accent) 16%, var(--sp-panel-soft));
}
.status-lite__stage--selected:not(.status-lite__stage--current) {
    flex-grow: 1.35;
    max-width: 320px;
    min-width: 150px;
    opacity: 1;
    border-color: color-mix(in srgb, var(--sp-accent) 68%, var(--sp-border));
}
.status-lite__stage--selected::after {
    content: "";
    position: absolute;
    left: 9px;
    right: 9px;
    bottom: 0;
    height: 2px;
    border-radius: 2px;
    background: var(--sp-accent);
}
.status-lite__stage-icon {
    grid-area: icon;
    display: inline-grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border: 1px solid var(--sp-border);
    border-radius: 999px;
    color: var(--sp-muted);
    font-size: .7rem;
    font-weight: 750;
}
.status-lite__stage--complete .status-lite__stage-icon {
    border-color: color-mix(in srgb, var(--sp-good) 70%, var(--sp-border));
    background: color-mix(in srgb, var(--sp-good) 18%, transparent);
    color: var(--sp-good);
}
.status-lite__stage--current .status-lite__stage-icon {
    border-color: var(--sp-accent);
    background: var(--sp-accent);
    color: white;
    animation: status-lite-pulse 1.8s ease-in-out infinite;
}
.status-lite__stage-name {
    grid-area: name;
    overflow: hidden;
    font-size: .82rem;
    font-weight: 700;
    line-height: 1.2;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-lite__stage-time {
    grid-area: time;
    overflow: hidden;
    color: var(--sp-muted);
    font-size: .72rem;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-lite__downloads {
    margin-top: 7px;
    padding: 10px;
    border: 1px solid color-mix(in srgb, var(--sp-accent) 48%, var(--sp-border));
    border-radius: 8px;
    background: color-mix(in srgb, var(--sp-accent) 7%, var(--sp-panel-soft));
}
.status-lite__downloads-header,
.status-lite__downloads-title {
    display: flex;
    align-items: center;
    gap: 10px;
}
.status-lite__downloads-header {
    justify-content: space-between;
}
.status-lite__downloads-title > div {
    display: grid;
    gap: 2px;
}
.status-lite__downloads-title strong { font-size: .84rem; }
.status-lite__downloads-title span,
.status-lite__download-total {
    color: var(--sp-muted);
    font-size: .72rem;
    font-variant-numeric: tabular-nums;
}
.status-lite__download-indicator {
    display: inline-grid;
    place-items: center;
    width: 27px;
    height: 27px;
    flex: 0 0 auto;
    border-radius: 999px;
    background: var(--sp-accent);
    color: white !important;
    font-size: 1rem !important;
    font-weight: 800;
}
.status-lite__downloads[data-active="true"] .status-lite__download-indicator {
    animation: status-lite-download-pulse 1.4s ease-in-out infinite;
}
.status-lite__download-overall {
    height: 4px;
    margin: 9px 0;
    overflow: hidden;
    border-radius: 99px;
    background: color-mix(in srgb, var(--sp-border) 78%, transparent);
}
.status-lite__download-overall > div {
    width: 0;
    height: 100%;
    border-radius: inherit;
    background: var(--sp-accent);
    transition: width 220ms linear;
}
.status-lite__download-files {
    display: grid;
    gap: 5px;
    max-height: 224px;
    overflow-y: auto;
    padding-right: 2px;
    scrollbar-width: thin;
}
.status-lite__download-file {
    display: grid;
    grid-template-columns: 20px minmax(150px, .8fr) minmax(280px, 1.2fr);
    grid-template-areas: "icon name stats" "icon freshness cycles" "icon bar bar";
    gap: 4px 9px;
    align-items: center;
    min-height: 34px;
    padding: 5px 7px;
    border-radius: 6px;
    background: color-mix(in srgb, var(--sp-panel) 58%, transparent);
}
.status-lite__download-file-icon {
    grid-area: icon;
    display: inline-grid;
    place-items: center;
    width: 20px;
    height: 20px;
    border: 1px solid var(--sp-border);
    border-radius: 999px;
    color: var(--sp-muted);
    font-size: .68rem;
    font-weight: 750;
}
.status-lite__download-file[data-state="downloading"] .status-lite__download-file-icon,
.status-lite__download-file[data-state="retrying"] .status-lite__download-file-icon {
    border-color: var(--sp-accent);
    color: var(--sp-accent);
}
.status-lite__download-file[data-state="complete"] .status-lite__download-file-icon {
    border-color: color-mix(in srgb, var(--sp-good) 72%, var(--sp-border));
    color: var(--sp-good);
}
.status-lite__download-file[data-state="failed"] .status-lite__download-file-icon {
    border-color: #ef4444;
    color: #ef4444;
}
.status-lite__download-file-name {
    grid-area: name;
    overflow: hidden;
    font-size: .74rem;
    font-weight: 650;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-lite__download-file-stats {
    grid-area: stats;
    color: var(--sp-muted);
    font-size: .69rem;
    font-variant-numeric: tabular-nums;
    text-align: right;
    white-space: nowrap;
}
.status-lite__download-file-freshness {
    grid-area: freshness;
    overflow: hidden;
    color: var(--sp-muted);
    font-size: .64rem;
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-lite__download-file-cycles {
    grid-area: cycles;
    overflow: hidden;
    color: color-mix(in srgb, var(--sp-accent) 74%, var(--sp-muted));
    font-size: .64rem;
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-lite__download-file-bar {
    grid-area: bar;
    height: 2px;
    overflow: hidden;
    border-radius: 99px;
    background: color-mix(in srgb, var(--sp-border) 70%, transparent);
}
.status-lite__download-file-bar > span {
    display: block;
    width: 0;
    height: 100%;
    border-radius: inherit;
    background: var(--sp-accent);
    transition: width 180ms linear;
}
.status-lite__download-file[data-state="complete"] .status-lite__download-file-bar > span {
    background: var(--sp-good);
}
.status-lite__detail {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    min-height: 56px;
    margin-top: 7px;
    padding: 9px 10px;
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    background: color-mix(in srgb, var(--sp-panel-soft) 82%, transparent);
}
.status-lite__detail-copy {
    display: grid;
    flex: 1 1 auto;
    gap: 2px;
    min-width: 0;
}
.status-lite__detail-copy strong { font-size: .86rem; }
.status-lite__detail-copy span {
    max-width: 100%;
    overflow: hidden;
    color: var(--sp-muted);
    font-size: .74rem;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-lite__detail-copy .status-lite__detail-model {
    color: color-mix(in srgb, var(--sp-accent) 76%, var(--sp-muted));
    font-weight: 650;
}
.status-lite__detail-copy .status-lite__detail-model--list {
    display: grid;
    gap: 2px;
    max-width: none;
    overflow: visible;
    text-overflow: clip;
    white-space: normal;
}
.status-lite__detail-copy .status-lite__detail-activities {
    display: grid;
    gap: 2px;
    max-width: none;
    overflow: visible;
    color: var(--sp-text);
    text-overflow: clip;
    white-space: normal;
}
.status-lite__detail-activity-line {
    display: block;
    min-width: 0;
    overflow-wrap: anywhere;
}
.status-lite__detail-model-line {
    display: block;
    min-width: 0;
    overflow-wrap: anywhere;
}
.status-lite__metrics {
    display: grid;
    flex: 0 0 auto;
    grid-template-columns: repeat(5, minmax(72px, auto));
    gap: 9px 16px;
    margin: 0;
}
.status-lite__metrics div { display: grid; gap: 1px; }
.status-lite__metrics dt {
    color: var(--sp-muted);
    font-size: .64rem;
    line-height: 1.1;
    text-transform: uppercase;
}
.status-lite__metrics dd {
    margin: 0;
    font-size: .76rem;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.status-lite__overall-track {
    height: 3px;
    margin-top: 8px;
    overflow: hidden;
    border-radius: 99px;
    background: color-mix(in srgb, var(--sp-border) 75%, transparent);
}
.status-lite__overall-fill {
    width: 0;
    height: 100%;
    border-radius: inherit;
    background: var(--sp-accent);
    transition: width 180ms linear;
}
.status-lite__overall-fill--indeterminate {
    width: 28% !important;
    animation: status-lite-indeterminate 1.35s ease-in-out infinite;
}
@keyframes status-lite-indeterminate {
    0% { transform: translateX(-110%); }
    50% { transform: translateX(135%); }
    100% { transform: translateX(360%); }
}
@keyframes status-lite-pulse {
    0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--sp-accent) 35%, transparent); }
    50% { box-shadow: 0 0 0 4px transparent; }
}
@keyframes status-lite-download-pulse {
    0%, 100% { transform: translateY(0); opacity: 1; }
    50% { transform: translateY(2px); opacity: .72; }
}
@container status-lite (max-width: 760px) {
    .status-lite__stage {
        flex-basis: 128px;
        max-width: none;
        min-width: 118px;
    }
    .status-lite__stage--current {
        flex-basis: 210px;
        max-width: none;
    }
    .status-lite__detail {
        align-items: stretch;
        flex-direction: column;
    }
    .status-lite__detail-copy { width: 100%; }
    .status-lite__detail-copy span {
        max-width: none;
        white-space: normal;
    }
    .status-lite__metrics {
        grid-template-columns: repeat(auto-fit, minmax(84px, 1fr));
        width: 100%;
    }
}
@container status-lite (max-width: 650px) {
    .status-lite__header { align-items: flex-start; flex-direction: column; }
    .status-lite__header-actions { justify-content: space-between; width: 100%; }
    .status-lite__summary { justify-content: flex-start; }
    .status-lite__stage { flex-basis: 104px; min-width: 104px; }
    .status-lite__stage--current { min-width: 174px; }
}
@container status-lite (max-width: 520px) {
    .status-lite__metrics { grid-template-columns: repeat(2, minmax(90px, 1fr)); }
    .status-lite__download-file {
        grid-template-columns: 20px minmax(0, 1fr);
        grid-template-areas: "icon name" "icon stats" "icon freshness" "icon cycles" "icon bar";
    }
    .status-lite__download-file-stats { text-align: left; }
}
@container status-lite (max-width: 900px) {
    .status-lite { padding: 10px; }
    .status-lite__header {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        grid-template-rows: auto auto;
        gap: 8px 10px;
        align-items: center;
        margin-bottom: 9px;
    }
    .status-lite__heading {
        grid-column: 1;
        grid-row: 1;
        min-width: 0;
        flex-wrap: nowrap;
    }
    .status-lite__live {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .status-lite__header-actions { display: contents; }
    .status-lite__summary {
        grid-column: 1 / -1;
        grid-row: 2;
        justify-content: center;
        min-width: 0;
        text-align: center;
    }
    .status-lite__collapse {
        grid-column: 2;
        grid-row: 1;
    }
    .status-lite__stages { gap: 6px; }
    .status-lite__stages .status-lite__stage {
        grid-template-columns: 24px;
        grid-template-areas: "icon";
        grid-template-rows: auto;
        justify-content: center;
        flex: 0 0 44px;
        width: 44px;
        min-width: 44px;
        max-width: 44px;
        min-height: 58px;
        padding: 7px 9px;
        column-gap: 0;
    }
    .status-lite__stages .status-lite__stage:not(.status-lite__stage--selected) .status-lite__stage-name,
    .status-lite__stages .status-lite__stage:not(.status-lite__stage--selected) .status-lite__stage-time {
        display: none;
    }
    .status-lite__stages .status-lite__stage--selected {
        grid-template-columns: 24px minmax(0, auto);
        grid-template-areas: "icon name" "icon time";
        grid-template-rows: auto auto;
        justify-content: center;
        flex: 1 1 220px;
        width: auto;
        min-width: 170px;
        max-width: none;
        column-gap: 9px;
    }
    .status-lite__stages .status-lite__stage--selected .status-lite__stage-name,
    .status-lite__stages .status-lite__stage--selected .status-lite__stage-time {
        display: block;
    }
}
@media (prefers-reduced-motion: reduce) {
    .status-lite__stage, .status-lite__overall-fill { transition: none; }
    .status-lite__stage--current .status-lite__stage-icon { animation: none; }
    .status-lite__overall-fill--indeterminate { width: 100% !important; animation: none; opacity: .45; }
    .status-lite__downloads[data-active="true"] .status-lite__download-indicator { animation: none; }
}
`;

    function appRoot() {
        if (window.gradioApp) return window.gradioApp();
        const app = document.querySelector("gradio-app");
        return app ? (app.shadowRoot || app) : document;
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function optionalNumber(value) {
        if (value === null || value === undefined || value === "") return null;
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function formatDuration(seconds, approximate = false) {
        if (!Number.isFinite(seconds) || seconds < 0) return "—";
        const rounded = Math.max(0, Math.round(seconds));
        const hours = Math.floor(rounded / 3600);
        const minutes = Math.floor((rounded % 3600) / 60);
        const secs = rounded % 60;
        let value;
        if (hours > 0) value = `${hours}h ${minutes}m`;
        else if (minutes > 0) value = `${minutes}m ${secs}s`;
        else value = `${secs}s`;
        return approximate ? `~${value}` : value;
    }

    function formatStepDuration(seconds) {
        if (!Number.isFinite(seconds) || seconds < 0) return "—";
        if (seconds < 10) return `${seconds.toFixed(2)}s`;
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        return formatDuration(seconds);
    }

    function formatLiveStepDuration(seconds) {
        if (!Number.isFinite(seconds) || seconds < 0) return "—";
        if (seconds < 0.1) return `${seconds.toFixed(2)}s`;
        if (seconds < 1) return `${seconds.toFixed(1)}s`;
        return formatDuration(seconds);
    }

    function formatBytes(bytes) {
        if (!Number.isFinite(bytes) || bytes < 0) return "—";
        const units = ["B", "KB", "MB", "GB", "TB"];
        let value = bytes;
        let index = 0;
        while (value >= 1024 && index < units.length - 1) {
            value /= 1024;
            index += 1;
        }
        const digits = value >= 100 || index === 0 ? 0 : (value >= 10 ? 1 : 2);
        return `${value.toFixed(digits)} ${units[index]}`;
    }

    function formatRate(bytesPerSecond) {
        return Number.isFinite(bytesPerSecond) && bytesPerSecond > 0
            ? `${formatBytes(bytesPerSecond)}/s`
            : "";
    }

    const RESOURCE_FIELDS = [
        "ram_rss_bytes",
        "ram_vms_bytes",
        "vram_allocated_bytes",
        "vram_reserved_bytes",
        "vram_device_used_bytes",
        "vram_device_free_bytes"
    ];

    function observeResourceSample(run, sample, includeInAverage = true) {
        if (!run || !sample || typeof sample !== "object") return;
        const sampledAt = optionalNumber(sample.sampled_at);
        if (Number.isFinite(sampledAt) && sampledAt === run._last_resource_sample_at) return;
        if (Number.isFinite(sampledAt)) run._last_resource_sample_at = sampledAt;
        const resources = run.resources || {
            sample_count: 0,
            observation_count: 0,
            sampling_interval_seconds: 0.5,
            scope: "Wan2GP process and active CUDA device",
            gpu_device_index: null,
            gpu_name: null,
            vram_device_total_bytes: null,
            metrics: {}
        };
        resources.observation_count += 1;
        if (includeInAverage) resources.sample_count += 1;
        if (sample.gpu_device_index !== null && sample.gpu_device_index !== undefined) {
            resources.gpu_device_index = sample.gpu_device_index;
        }
        if (sample.gpu_name) resources.gpu_name = String(sample.gpu_name);
        const gpuTotal = optionalNumber(sample.vram_device_total_bytes);
        if (Number.isFinite(gpuTotal)) resources.vram_device_total_bytes = gpuTotal;
        RESOURCE_FIELDS.forEach(field => {
            const value = optionalNumber(sample[field]);
            if (!Number.isFinite(value)) return;
            const metric = resources.metrics[field] || {
                start_bytes: value,
                end_bytes: value,
                peak_bytes: value,
                total_bytes: 0,
                sample_count: 0
            };
            metric.end_bytes = value;
            metric.peak_bytes = Math.max(metric.peak_bytes, value);
            if (includeInAverage) {
                metric.total_bytes += value;
                metric.sample_count += 1;
                metric.average_bytes = Math.round(metric.total_bytes / metric.sample_count);
            }
            resources.metrics[field] = metric;
        });
        run.resources = resources;
    }

    function observePerformanceTelemetry(run, telemetry) {
        if (!run || !telemetry) return;
        observeResourceSample(run, telemetry.resource_sample);
        const performance = telemetry.performance;
        if (!performance || typeof performance !== "object") return;
        const observerTaskId = performance.task_id;
        if (
            observerTaskId !== null && observerTaskId !== undefined &&
            run.queue_task_id !== null && run.queue_task_id !== undefined &&
            String(observerTaskId) !== String(run.queue_task_id)
        ) return;
        const observerId = String(performance.id || "observer");
        run.step_performance = Array.isArray(run.step_performance) ? run.step_performance : [];
        run._performance_step_keys = run._performance_step_keys || {};
        (Array.isArray(performance.steps) ? performance.steps : []).forEach(step => {
            if (!step || typeof step !== "object") return;
            const completedAt = optionalNumber(step.completed_at);
            const completedMs = Number.isFinite(completedAt) ? completedAt * 1000 : null;
            if (Number.isFinite(completedMs) && Number.isFinite(run.started_at) && completedMs < run.started_at - 1000) return;
            if (Number.isFinite(completedMs) && Number.isFinite(run.completed_at) && completedMs > run.completed_at + 1000) return;
            const key = `${observerId}:${step.sequence}`;
            if (run._performance_step_keys[key]) return;
            run._performance_step_keys[key] = true;
            const copy = cloneJson(step, {});
            copy.observer_id = observerId;
            if (observerTaskId !== null && observerTaskId !== undefined) copy.observer_task_id = observerTaskId;
            run.step_performance.push(copy);
            if (run.step_performance.length > MAX_STEP_RECORDS) {
                run.step_performance = run.step_performance.slice(-MAX_STEP_RECORDS);
                run._performance_step_keys = Object.fromEntries(run.step_performance.map(record => [`${record.observer_id}:${record.sequence}`, true]));
                run.step_performance_source_truncated = true;
            }
            observeResourceSample(run, copy.memory, false);
        });
        if (performance.steps_truncated) run.step_performance_source_truncated = true;
    }

    function parseDuration(text) {
        const source = String(text || "");
        let seconds = 0;
        let matched = false;
        const hours = source.match(/(\d+(?:\.\d+)?)\s*h(?:ours?)?\b/i);
        const minutes = source.match(/(\d+(?:\.\d+)?)\s*m(?:in(?:utes?)?)?\b/i);
        const secs = source.match(/(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?\b/i);
        if (hours) { seconds += Number(hours[1]) * 3600; matched = true; }
        if (minutes) { seconds += Number(minutes[1]) * 60; matched = true; }
        if (secs) { seconds += Number(secs[1]); matched = true; }
        if (matched) return seconds;
        const clock = source.match(/\b(?:(\d+):)?(\d{1,2}):(\d{2})\b/);
        if (!clock) return null;
        return Number(clock[1] || 0) * 3600 + Number(clock[2]) * 60 + Number(clock[3]);
    }

    function parsePercent(levelText, progressBar) {
        const match = String(levelText || "").match(/(\d+(?:\.\d+)?)\s*%/);
        if (match) return clamp(Number(match[1]), 0, 100);
        if (progressBar) {
            const width = String(progressBar.style.width || "").match(/(\d+(?:\.\d+)?)\s*%/);
            if (width) return clamp(Number(width[1]), 0, 100);
        }
        return null;
    }

    function parseSteps(metaText) {
        const match = String(metaText || "").match(/(\d+)\s*\/\s*(\d+)(?:\s*steps?)?/i);
        if (!match) return { current: null, total: null };
        return { current: Number(match[1]), total: Number(match[2]) };
    }

    function parseProgressTiming(levelText) {
        const timing = String(levelText || "").split("|").pop().trim();
        const values = timing.split(/\s*\/\s*/).map(parseDuration).filter(Number.isFinite);
        return {
            elapsed: values[0] ?? null,
            total: values.length >= 2 ? values[1] : null
        };
    }

    function stageIdFor(rawName) {
        const name = String(rawName || "").toLowerCase();
        const modelLifecycle = /\b(?:prepar|load|loading|loaded|unload|unloading|unloaded|releas|download|queue|cache|compil|warm.?up|initializ|abort|cancel|interrupt)\w*\b/.test(name) &&
            /\b(?:model|weight|checkpoint|transformer|encoder|vae|whisper|vocoder|lora|file|asset|prompt enhancer)\w*\b/.test(name);
        if (modelLifecycle || /\b(?:initializ|abort|cancel|interrupt)\w*\b/.test(name)) return "prepare";
        if (/\b(?:sav(?:e|ing|ed)?|export\w*|writ(?:e|ing|ten)?|mux\w*|remux\w*|finaliz\w*)\b/.test(name)) return "save";

        // Semantic prompt/text work belongs to Encode even when it uses words such as
        // "enhancing" or mentions references. Check it before media preprocessing.
        if (/(?:enhanc\w*\s+prompt|prompt\s+enhanc\w*|(?:prompt|text|caption)\s+(?:and\s+references?\s+)?encod\w*|encod\w*\s+(?:(?:h3\s+)?prompt|speaker\s+\d*\s*reference)|condition\w*|embed\w*|token\w*|text\s+feature)/.test(name)) return "encode";

        const inputContext = /\b(?:input|control|source|guide|mask|pose|depth|canny|face|movement|reference|background)\w*\b/.test(name);
        const inputWork = /\b(?:prepar|pre.?process|load|extract|remov|resiz|crop|trim|normaliz|separat|align|encod|decod)\w*\b/.test(name);
        const mediaPreprocessing = /\b(?:prepar|pre.?process|load|extract|remov|resiz|crop|trim|normaliz|separat|align)\w*\b.*\b(?:frame|image|video|audio)\w*\b|\b(?:frame|image|video|audio)\w*\b.*\b(?:prepar|pre.?process|load|extract|remov|resiz|crop|trim|normaliz|separat|align)\w*\b/.test(name);
        if (/\bvae\s+encod\w*\b/.test(name) ||
            /\b(?:pre.?process|extract)\w*\b/.test(name) ||
            (inputContext && inputWork) || mediaPreprocessing ||
            /\b(?:extracting\s+(?:pose|depth|face)|removing\s+(?:image\s+)?references?\s+background)\b/.test(name)) return "input";

        if (/\b(?:denois|diffus|sampl|synthesis|synthes|generating\s+(?:audio|waveform|speech)|spectrum\s+smoothing\s+replay)/.test(name)) return "denoise";
        if (/(?:vae\s*decod|decod|reconstruct)/.test(name)) return "decode";
        if (/(?:post.?process|upscal|upsampl|interpol|color correction|film grain|tcdecoder|seedvc|voice replacement|audio post|soundtrack|enhanc)/.test(name)) return "post";
        if (/(?:encod|prompt|condition|embed|feature|token)/.test(name)) return "encode";
        if (/(?:prepar|initial|load|download|queue|cache|start)/.test(name)) return "prepare";
        return "prepare";
    }

    function phaseInfo(rawName) {
        const label = String(rawName || "Preparing")
            .replace(/^(?:(?:prompt|sample|sliding window)\s+\d+\s*\/\s*\d+\s*,?\s*)+/i, "")
            .replace(/^\s*-\s*/, "")
            .trim() || "Preparing";
        const kind = stageIdFor(label);
        const phaseMatch = label.match(/\bdenoising\s+(first|second|third|\d+(?:st|nd|rd|th)?)\s+phase\b/i);
        const phaseNumber = phaseMatch
            ? ({ first: 1, second: 2, third: 3 }[phaseMatch[1].toLowerCase()] || Number.parseInt(phaseMatch[1], 10))
            : null;
        const key = label.toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-+|-+$/g, "") || kind;
        return { id: `${kind}:${key}`, kind, label, phaseNumber };
    }

    function stageNameFrom(levelText) {
        const first = String(levelText || "").split("|")[0].trim();
        return first.replace(/\s+-\s+\d+(?:\.\d+)?\s*%\s*$/, "").trim() || "Preparing";
    }

    function loadCollapsedPreference() {
        try {
            return window.localStorage.getItem(COLLAPSED_KEY) === "1";
        } catch (_) {
            return false;
        }
    }

    function saveCollapsedPreference(collapsed) {
        try {
            window.localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
        } catch (_) {
            // Compact mode still works when local storage is unavailable.
        }
    }

    function createStageRecords() {
        return Object.fromEntries(STAGE_DEFS.map(def => [def.id, {
            id: def.id,
            label: def.label,
            visible: !def.optional,
            state: "pending",
            preloaded: false,
            unreported: false,
            activity: null,
            activityModel: "",
            rawName: "",
            rawMessage: "",
            startedAt: null,
            elapsed: null,
            elapsedBase: 0,
            reportedElapsed: null,
            reportedAt: null,
            nativeEta: null,
            progress: null,
            eta: null,
            samples: [],
            stepCurrent: null,
            stepTotal: null,
            lastStepCurrent: null,
            lastStepElapsed: null,
            lastStepAt: null,
            stepSamples: [],
            stepSeconds: null,
            recovered: false
        }]));
    }

    function freshState() {
        return {
            records: createStageRecords(),
            phases: {},
            phaseOrder: [],
            currentPhaseId: null,
            activeDenoisePhase: null,
            lastDenoisePhase: null,
            currentId: null,
            selectedId: null,
            selectionIsManual: false,
            overallElapsed: null,
            steps: { current: null, total: null },
            lastSeenAt: 0,
            inactiveSince: 0,
            jobStartedAt: Date.now()
        };
    }

    function resetJob(namespace) {
        namespace.state = freshState();
    }

    function cloneJson(value, fallback = {}) {
        try {
            return JSON.parse(JSON.stringify(value));
        } catch (_) {
            return fallback;
        }
    }

    function runtimeIdFromTelemetry(telemetry) {
        return String(telemetry && telemetry.runtime_id || "").trim();
    }

    function parseRunBridge(container) {
        const field = container && container.querySelector(
            "#status-lite-run-bridge textarea, #status-lite-run-bridge input"
        );
        const raw = String(field ? field.value : "").trim();
        if (!raw) return { raw: "", telemetry: null };
        try {
            const parsed = JSON.parse(raw);
            const telemetry = parsed && typeof parsed === "object" && Number.isFinite(Number(parsed.server_time))
                ? parsed
                : null;
            return { raw, telemetry };
        } catch (_) {
            return { raw, telemetry: null };
        }
    }

    function readRunSnapshot(namespace) {
        const field = namespace.container.querySelector(
            "#status-lite-run-bridge textarea, #status-lite-run-bridge input"
        );
        const raw = String(field ? field.value : "").trim();
        if (!raw || raw === namespace.runRaw) return namespace.runTelemetry;
        namespace.runRaw = raw;
        try {
            const parsed = JSON.parse(raw);
            namespace.runTelemetry = parsed && typeof parsed === "object" && Number.isFinite(Number(parsed.server_time))
                ? parsed
                : null;
            if (namespace.runTelemetry) {
                const runtimeId = runtimeIdFromTelemetry(namespace.runTelemetry);
                if (runtimeId) namespace.runtimeId = runtimeId;
            }
        } catch (_) {
            namespace.runTelemetry = null;
        }
        return namespace.runTelemetry;
    }

    function runTaskKey(task) {
        return task && task.id !== null && task.id !== undefined ? String(task.id) : "";
    }

    function windowDetails(telemetry) {
        if (!telemetry) return { number: null, total: null };
        let number = optionalNumber(telemetry.window_no);
        let total = optionalNumber(telemetry.total_windows);
        const status = String(telemetry.status || "");
        const match = status.match(/sliding\s+window\s+(\d+)\s*\/\s*(\d+)/i);
        if (match) {
            number = number ?? Number(match[1]);
            total = total ?? Number(match[2]);
        }
        if (!telemetry.sliding_window && !match) return { number: null, total: null };
        return {
            number: Number.isFinite(number) && number > 0 ? Math.floor(number) : null,
            total: Number.isFinite(total) && total > 0 ? Math.floor(total) : null
        };
    }

    function windowPromptFor(task, windowNo) {
        const prompts = task && Array.isArray(task.window_prompts) ? task.window_prompts : [];
        if (Number.isFinite(windowNo) && prompts.length) {
            const prompt = prompts[Math.min(Math.max(0, Math.floor(windowNo) - 1), prompts.length - 1)];
            if (prompt !== null && prompt !== undefined && String(prompt)) return String(prompt);
        }
        return task && task.window_prompt ? String(task.window_prompt) : null;
    }

    function relevantQueueError(telemetry, run) {
        const errors = telemetry && telemetry.queue_errors;
        const clientId = String(run && run.client_id || "");
        return clientId && errors && typeof errors === "object" ? errors[clientId] : "";
    }

    function finishPhase(state) {
        const record = state.phases[state.currentPhaseId];
        if (!record || record.state !== "current") return;
        record.elapsed = (Date.now() - record.startedAt) / 1000;
        record.state = "complete";
    }

    function resetStageForNextDenoisePhase(state, id) {
        const record = state.records[id];
        if (!record) return;
        const definition = STAGE_DEFS.find(def => def.id === id);
        record.visible = !definition || !definition.optional;
        record.state = "pending";
        record.preloaded = false;
        record.unreported = false;
        record.activity = null;
        record.activityModel = "";
        record.rawName = "";
        record.rawMessage = "";
        record.startedAt = null;
        record.elapsed = null;
        record.elapsedBase = 0;
        record.reportedElapsed = null;
        record.reportedAt = null;
        record.progress = null;
        record.eta = null;
        record.samples = [];
        record.stepCurrent = null;
        record.stepTotal = null;
        record.lastStepCurrent = null;
        record.lastStepElapsed = null;
        record.lastStepAt = null;
        record.stepSamples = [];
        record.stepSeconds = null;
        record.recovered = false;
    }

    function resetDownstreamStagesForNextDenoisePhase(state) {
        ["decode", "post", "save"].forEach(id => resetStageForNextDenoisePhase(state, id));
        if (!state.selectionIsManual) state.selectedId = "denoise";
    }

    function applyPhase(state, snapshot, now) {
        const phase = phaseInfo(snapshot.rawName);
        if (state.currentPhaseId !== phase.id) {
            finishPhase(state);
            const isLaterDenoisePhase = phase.kind === "denoise" &&
                Number.isFinite(phase.phaseNumber) &&
                Number.isFinite(state.lastDenoisePhase) &&
                phase.phaseNumber > state.lastDenoisePhase;
            if (isLaterDenoisePhase) resetDownstreamStagesForNextDenoisePhase(state);
            state.currentPhaseId = phase.id;
            state.phases[phase.id] = {
                id: phase.id,
                label: phase.label,
                stage: snapshot.id,
                startedAt: now,
                elapsed: null,
                state: snapshot.aborting ? "aborting" : "current"
            };
            state.phaseOrder.push(phase.id);
        }
        if (phase.kind === "denoise" && Number.isFinite(phase.phaseNumber)) {
            state.activeDenoisePhase = phase.phaseNumber;
            state.lastDenoisePhase = phase.phaseNumber;
        }
        const record = state.phases[state.currentPhaseId];
        record.state = snapshot.aborting ? "aborting" : "current";
        record.elapsed = (now - record.startedAt) / 1000;
    }

    function runStatusFrom(namespace, telemetry) {
        const field = namespace.source.querySelector("textarea, input");
        const message = `${String(telemetry && telemetry.status || "")} ${String(field && field.value || "")}`;
        if (/\b(abort(?:ed|ing)?|cancel(?:led|ling)?|interrupt(?:ed|ing)?)\b/i.test(message)) return "aborted";
        if (/\b(error|failed|failure|exception)\b/i.test(message)) return "failed";
        return "completed";
    }

    function observeRunOutcome(namespace, ...messages) {
        const run = namespace.activeRun;
        if (!run) return;
        const message = messages
            .filter(value => value !== null && value !== undefined && value !== "")
            .map(value => typeof value === "string" ? value : JSON.stringify(value))
            .join(" ")
            .replace(/\s+/g, " ")
            .trim();
        if (!message) return;
        if (run.notice_baseline && message === run.notice_baseline) return;

        const aborted = /\b(abort(?:ed|ing)?|cancel(?:led|ling)?|interrupt(?:ed|ing)?)\b/i.test(message);
        const failed = /\b(error|failed|failure|exception|traceback|out of memory|oom|insufficient|unsufficient|tried to allocate)\b/i.test(message);
        if (failed) {
            run.outcome_status = "failed";
            if (/\b(cuda out of memory|out of memory|oom|tried to allocate|insufficient vram|unsufficient vram)\b/i.test(message)) {
                run.status_reason = "Out of GPU memory (VRAM). Reduce resolution, frame count, or memory use.";
            } else if (/\b(insufficient ram|unsufficient ram|reserved ram)\b/i.test(message)) {
                run.status_reason = "Insufficient system or reserved RAM.";
            } else {
                run.status_reason = message.slice(0, 500);
            }
            return;
        }
        if (aborted && run.outcome_status !== "failed") {
            run.outcome_status = "aborted";
            run.status_reason = "Cancelled before completion.";
        }
    }

    function visibleFailureNotice(namespace) {
        if (!namespace.root || !namespace.root.querySelectorAll) return "";
        const notices = Array.from(namespace.root.querySelectorAll('[role="alert"], .toast, .error'));
        for (let index = notices.length - 1; index >= Math.max(0, notices.length - 12); index -= 1) {
            const message = String(notices[index].textContent || "").trim();
            if (/\b(error|failed|failure|out of memory|oom|insufficient|unsufficient|abort|cancel)\b/i.test(message)) {
                return message;
            }
        }
        return "";
    }

    function normalizeLatePostprocessingModel(settings) {
        if (!settings || typeof settings !== "object") return false;
        const mode = String(settings.mode || "").trim().toLowerCase();
        if (!["edit_postprocessing", "edit_remux", "edit_audio"].includes(mode)) return false;

        const spatial = String(settings.spatial_upsampling || "").trim().toLowerCase();
        const variants = {
            ltx23: {modelType: "ltx2_22B", modelName: "LTX-2 2.3 Pixel Spatial Upscaler"},
            ltx25: {modelType: "ltx2_25_22B_distilled", modelName: "LTX-2 2.5 Pixel Spatial Upscaler"}
        };
        const variantKey = mode === "edit_postprocessing"
            ? Object.keys(variants).find(method => spatial.startsWith(method))
            : null;
        const variant = variantKey ? variants[variantKey] : null;
        const previousModelType = String(settings.model_type || settings.base_model_type || "");
        const previousModelName = String(settings.model_name || "");
        const hadPreviousModelIdentity = Boolean(
            previousModelType || settings.model_filename || previousModelName || settings.model_family || settings.component_models
        );

        [
            "model_type", "base_model_type", "model_filename", "config", "model_name", "model_family",
            "activated_loras", "loras_multipliers"
        ].forEach(key => {
            delete settings[key];
        });
        if (variant) {
            settings.model_type = variant.modelType;
            settings.model_name = variant.modelName;
            settings.model_family = "LTX-2";
            settings.num_inference_steps = 8;
            if (previousModelType !== variant.modelType) {
                delete settings.component_models;
                delete settings.attention_mode;
                delete settings.override_attention;
                delete settings.attention_sparsity;
            }
        } else {
            delete settings.component_models;
            delete settings.attention_mode;
            delete settings.override_attention;
            delete settings.attention_sparsity;
        }
        const alreadyMatched = Boolean(
            variant && previousModelType === variant.modelType && (!previousModelName || previousModelName === variant.modelName)
        );
        return hadPreviousModelIdentity && !alreadyMatched;
    }

    function fallbackPostprocessingLabel(value, kind) {
        const text = String(value || "").trim();
        const lowered = text.toLowerCase();
        const methods = kind === "temporal_upsampling"
            ? {rife: "RIFE"}
            : {
                flashvsr2pass: "FlashVSR Two Pass",
                flashvsr: "FlashVSR",
                seedvr2: "SeedVR2",
                lanczos: "Lanczos",
                ltx25: "LTX 2.5 Pixel Spatial Upscaler",
                ltx23: "LTX 2.3 Pixel Spatial Upscaler",
                coz: "Chain of Zoom"
            };
        const method = Object.keys(methods).sort((left, right) => right.length - left.length)
            .find(candidate => lowered.startsWith(candidate));
        if (!method) return text;
        const scale = optionalNumber(lowered.slice(method.length));
        return Number.isFinite(scale) ? `${methods[method]} x${scale}` : methods[method];
    }

    function normalizePostprocessingMetadata(settings) {
        if (!settings || typeof settings !== "object") return null;
        const source = settings.postprocessing && typeof settings.postprocessing === "object" && !Array.isArray(settings.postprocessing)
            ? settings.postprocessing
            : null;
        let operations = source && Array.isArray(source.operations)
            ? source.operations.filter(operation => operation && typeof operation === "object" && !Array.isArray(operation)).map(operation => {
                const normalized = {
                    kind: String(operation.kind || "").slice(0, 80),
                    label: String(operation.label || "").slice(0, 500)
                };
                if (operation.value !== null && operation.value !== undefined) normalized.value = String(operation.value).slice(0, 300);
                if (operation.processor) normalized.processor = String(operation.processor).slice(0, 300);
                if (operation.method) normalized.method = String(operation.method).slice(0, 200);
                for (const key of ["scale", "intensity", "saturation", "steps"]) {
                    const value = optionalNumber(operation[key]);
                    if (Number.isFinite(value)) normalized[key] = value;
                }
                if (operation.model && typeof operation.model === "object" && !Array.isArray(operation.model)) {
                    normalized.model = {
                        model_type: String(operation.model.model_type || "").slice(0, 300),
                        model_name: String(operation.model.model_name || "").slice(0, 500)
                    };
                    if (operation.model.component_models && typeof operation.model.component_models === "object" && !Array.isArray(operation.model.component_models)) {
                        const components = {};
                        Object.entries(operation.model.component_models).slice(0, 10).forEach(([stage, values]) => {
                            const names = (Array.isArray(values) ? values : [values])
                                .slice(0, 20)
                                .map(value => String(value || "").slice(0, 500))
                                .filter(Boolean);
                            if (names.length) components[String(stage).slice(0, 80)] = names;
                        });
                        if (Object.keys(components).length) normalized.model.component_models = components;
                    }
                    if (!normalized.model.model_type && !normalized.model.model_name) delete normalized.model;
                }
                return normalized;
            }).filter(operation => operation.kind && operation.label)
            : [];
        if (!operations.length) {
            const temporal = String(settings.temporal_upsampling || "").trim();
            if (temporal) operations.push({
                kind: "temporal_upsampling",
                label: fallbackPostprocessingLabel(temporal, "temporal_upsampling"),
                value: temporal
            });
            const spatial = String(settings.spatial_upsampling || "").trim();
            if (spatial) {
                const operation = {
                    kind: "spatial_upsampling",
                    label: fallbackPostprocessingLabel(spatial, "spatial_upsampling"),
                    value: spatial
                };
                const loweredSpatial = spatial.toLowerCase();
                if (loweredSpatial.startsWith("ltx25")) operation.model = {
                    model_type: "ltx2_25_22B_distilled",
                    model_name: "LTX-2 2.5 Pixel Spatial Upscaler"
                };
                else if (loweredSpatial.startsWith("ltx23")) operation.model = {
                    model_type: "ltx2_22B",
                    model_name: "LTX-2 2.3 Pixel Spatial Upscaler"
                };
                if (operation.model) operation.steps = 8;
                operations.push(operation);
            }
            const intensity = optionalNumber(settings.film_grain_intensity);
            if (intensity > 0) {
                const saturation = optionalNumber(settings.film_grain_saturation);
                const operation = {
                    kind: "film_grain",
                    label: `Film grain (intensity ${intensity}${Number.isFinite(saturation) ? `, saturation ${saturation}` : ""})`,
                    intensity
                };
                if (Number.isFinite(saturation)) operation.saturation = saturation;
                operations.push(operation);
            }
        }
        if (!operations.length) {
            delete settings.postprocessing;
            return null;
        }
        const mode = String(settings.mode || "").trim().toLowerCase();
        const application = mode
            ? (mode === "edit_postprocessing" ? "late" : "inline")
            : (source && ["late", "inline"].includes(source.application) ? source.application : "inline");
        const metadata = {
            application,
            summary: operations.map(operation => operation.label).join(" · ").slice(0, 1500),
            operations
        };
        settings.postprocessing = metadata;
        return metadata;
    }

    function componentModelsMatchModelType(componentModels, modelType) {
        if (!componentModels || typeof componentModels !== "object" || Array.isArray(componentModels)) return false;
        const names = Object.values(componentModels)
            .flatMap(values => Array.isArray(values) ? values : [values])
            .map(value => String(value || "").trim())
            .filter(Boolean);
        if (!names.length) return false;
        const normalizedModelType = String(modelType || "").trim().toLowerCase();
        if (normalizedModelType.startsWith("ltx2")) {
            if (names.some(name => /(?:minimax|qwen|hunyuan|wan2(?:\.|_)?)/i.test(name))) return false;
            return names.some(name => /(?:ltx|gemma)/i.test(name));
        }
        return true;
    }

    function ltxComponentFallbacks(modelType) {
        const normalized = String(modelType || "").trim().toLowerCase();
        if (normalized === "ltx2_25_22b_distilled") return {
            prepare: ["LTX-2.5 22B distilled transformer"],
            input: ["LTX-2.5 video VAE", "LTX-2.5 audio VAE"],
            encode: ["Gemma 4 12B LTX v1 text encoder"],
            decode: ["LTX-2.5 video VAE", "LTX-2.5 audio VAE"]
        };
        if (normalized === "ltx2_22b") return {
            prepare: ["LTX-2.3 22B transformer"],
            input: ["LTX-2.3 video VAE", "LTX-2.3 audio VAE"],
            encode: ["Gemma 3 12B LTX text encoder"],
            decode: ["LTX-2.3 video VAE", "LTX-2.3 audio VAE"]
        };
        return null;
    }

    function completedPostprocessingComponents(model) {
        if (!model || typeof model !== "object") return null;
        const modelType = String(model.model_type || "");
        const exact = componentModelsMatchModelType(model.component_models, modelType)
            ? cloneJson(model.component_models, {})
            : {};
        const fallbacks = ltxComponentFallbacks(modelType);
        if (!fallbacks) return Object.keys(exact).length ? exact : null;
        Object.entries(fallbacks).forEach(([stage, values]) => {
            if (!exact[stage] || !(Array.isArray(exact[stage]) ? exact[stage] : [exact[stage]]).filter(Boolean).length) {
                exact[stage] = values.slice();
            }
        });
        return exact;
    }

    function normalizeRunSettings(settings) {
        if (!settings || typeof settings !== "object") return false;
        const repairedModel = normalizeLatePostprocessingModel(settings);
        const metadata = normalizePostprocessingMetadata(settings);
        if (!metadata || metadata.application !== "late") return repairedModel;

        const modelType = String(settings.model_type || "").trim();
        if (!modelType.toLowerCase().startsWith("ltx2")) return repairedModel;
        const backingOperation = metadata.operations.find(operation =>
            operation && operation.model && String(operation.model.model_type || "") === modelType
        );
        const backingComponents = backingOperation && completedPostprocessingComponents(backingOperation.model);
        if (componentModelsMatchModelType(backingComponents, modelType)) {
            settings.component_models = cloneJson(backingComponents, {});
            backingOperation.model.component_models = cloneJson(backingComponents, {});
        } else if (!componentModelsMatchModelType(settings.component_models, modelType)) {
            delete settings.component_models;
        }
        return repairedModel;
    }

    function startRun(namespace, task, telemetry, options = {}) {
        resetJob(namespace);
        const observedNow = Number(telemetry && telemetry.server_time) * 1000 || Date.now();
        const now = Number.isFinite(options.startedAt) ? options.startedAt : observedNow;
        const settings = cloneJson(task && task.settings, {});
        normalizeRunSettings(settings);
        const observedWindow = windowDetails(telemetry);
        const window = {
            number: Number.isFinite(options.windowNo) ? options.windowNo : observedWindow.number,
            total: Number.isFinite(options.totalWindows) ? options.totalWindows : observedWindow.total
        };
        const windowPrompt = windowPromptFor(task, window.number);
        if (windowPrompt) settings.prompt = windowPrompt;
        namespace.activeRun = {
            id: `${namespace.sessionId}-${runTaskKey(task) || "observed"}-${Math.round(now)}`,
            session_id: namespace.sessionId,
            queue_task_id: task && task.id !== undefined ? task.id : null,
            client_id: String(task && task.client_id || ""),
            status: "running",
            started_at: now,
            completed_at: null,
            duration_seconds: null,
            settings,
            stages: {},
            step_performance: [],
            resources: null,
            step_summary: null,
            _performance_step_keys: {},
            outputs: [],
            repeats: optionalNumber(task && task.repeats) || 1,
            window_no: window.number,
            total_windows: window.total,
            window_prompt: windowPrompt,
            window_prompts: cloneJson(task && task.window_prompts, []),
            outcome_status: null,
            status_reason: null,
            notice_baseline: visibleFailureNotice(namespace)
        };
        observePerformanceTelemetry(namespace.activeRun, telemetry);
    }

    function updateActiveRun(namespace, task, telemetry) {
        if (!namespace.activeRun || !task) return;
        const settings = {
            ...namespace.activeRun.settings,
            ...cloneJson(task.settings, {})
        };
        normalizeRunSettings(settings);
        namespace.activeRun.settings = settings;
        const window = windowDetails(telemetry);
        if (Number.isFinite(window.number)) namespace.activeRun.window_no = window.number;
        if (Number.isFinite(window.total)) namespace.activeRun.total_windows = window.total;
        const windowPrompt = windowPromptFor(task, namespace.activeRun.window_no);
        if (windowPrompt) {
            namespace.activeRun.window_prompt = windowPrompt;
            namespace.activeRun.settings.prompt = namespace.activeRun.window_prompt;
        }
        namespace.activeRun.repeats = optionalNumber(task.repeats) || namespace.activeRun.repeats || 1;
        observePerformanceTelemetry(namespace.activeRun, telemetry);
    }

    function finishRun(namespace, status, completedAt, telemetry, outputEnd) {
        const run = namespace.activeRun;
        if (!run) return;
        observePerformanceTelemetry(run, telemetry || namespace.runTelemetry);
        recoverMissedPerformanceStages(namespace);
        finishStage(namespace.state, namespace.state.currentId);
        finishPhase(namespace.state);
        const ended = Number.isFinite(completedAt) ? completedAt : Date.now();
        namespace.lastCompletedAt = ended;
        namespace.activeRun = null;
        resetJob(namespace);
    }

    function syncRunTelemetry(namespace) {
        const telemetry = readRunSnapshot(namespace);
        if (!telemetry) return;
        const task = telemetry.active_task && typeof telemetry.active_task === "object"
            ? telemetry.active_task
            : null;
        const nextKey = runTaskKey(task);
        const activeKey = namespace.activeRun && namespace.activeRun.queue_task_id !== null
            ? String(namespace.activeRun.queue_task_id)
            : "";
        const now = Number(telemetry.server_time) * 1000 || Date.now();
        if (namespace.activeRun) observePerformanceTelemetry(namespace.activeRun, telemetry);

        if (task) {
            if (namespace.activeRun && activeKey && activeKey !== nextKey) {
                finishRun(namespace, "completed", now, telemetry);
            }
            if (!namespace.activeRun) startRun(namespace, task, telemetry);
            updateActiveRun(namespace, task, telemetry);
            observeRunOutcome(namespace, telemetry.status, relevantQueueError(telemetry, namespace.activeRun));
            return;
        }

        if (namespace.activeRun) {
            observeRunOutcome(namespace, telemetry.status, relevantQueueError(telemetry, namespace.activeRun));
            finishRun(namespace, runStatusFrom(namespace, telemetry), now, telemetry);
        }
    }

    function findTracker(namespace) {
        if (namespace.tracker && namespace.tracker.isConnected && namespace.tracker.querySelector(".progress-level-inner")) {
            return namespace.tracker;
        }
        const wrappers = Array.from(namespace.source.querySelectorAll(".wrap.default, .wrap"));
        namespace.tracker = wrappers.find(wrapper =>
            wrapper.querySelector(".progress-level-inner") && wrapper.querySelector(".progress-text")
        ) || null;
        return namespace.tracker;
    }

    function statusSnapshot(namespace, message) {
        message = String(message || "").trim();
        if (!message) return null;
        const aborting = /\b(abort(?:ing|ed)?|cancel(?:ling|ed)?|interrupt(?:ing|ed)?)\b/i.test(message);
        const loadWord = "(?:load(?:ing|ed)?|download(?:ing|ed)?)";
        const assetWord = "(?:models?|weights?|files?|assets?)";
        const modelActivity = new RegExp(
            `\\b${loadWord}\\b.*\\b${assetWord}\\b|\\b${assetWord}\\b.*\\b${loadWord}\\b`,
            "i"
        ).test(message);
        if (!aborting && !modelActivity) return null;
        const loadComplete = /\b(?:loaded|downloaded)\b/i.test(message);
        const id = aborting && namespace.state.currentId
            ? namespace.state.currentId
            : stageIdFor(message);
        return {
            id,
            rawName: aborting ? "Aborting" : (loadComplete ? "Model loaded" : "Loading model"),
            rawMessage: message,
            metaText: "",
            stageElapsed: null,
            overallElapsed: namespace.state.overallElapsed,
            progress: loadComplete ? 100 : null,
            steps: namespace.state.steps,
            aborting,
            activity: loadComplete ? "load-complete" : "load",
            activityModel: "",
            textOnly: true
        };
    }

    function modelLifecycleSnapshot(namespace) {
        const telemetry = namespace.runTelemetry;
        const lifecycle = telemetry && telemetry.model_lifecycle;
        if (!lifecycle || typeof lifecycle !== "object") return null;
        const lifecycleState = String(lifecycle.state || "");
        if (!/^(?:unloading|unloaded|failed)$/.test(lifecycleState)) return null;
        const serverTime = optionalNumber(telemetry.server_time);
        const startedAt = optionalNumber(lifecycle.started_at);
        const completedAt = optionalNumber(lifecycle.completed_at);
        const stageElapsed = Number.isFinite(startedAt) && Number.isFinite(serverTime)
            ? Math.max(0, (Number.isFinite(completedAt) ? completedAt : serverTime) - startedAt)
            : null;
        const modelName = String(lifecycle.model_name || lifecycle.model_type || "Previously loaded model");
        const failed = lifecycleState === "failed";
        const complete = lifecycleState === "unloaded";
        return {
            id: "prepare",
            rawName: failed ? "Model unload failed" : (complete ? "Model unloaded" : `Unloading ${modelName}`),
            rawMessage: failed
                ? (String(lifecycle.error || "The previous model could not be unloaded."))
                : (complete
                    ? `${modelName} was released from RAM and VRAM.`
                    : `Releasing ${modelName} from RAM and VRAM before the next model loads.`),
            metaText: "",
            stageElapsed,
            overallElapsed: namespace.state.overallElapsed,
            progress: null,
            steps: namespace.state.steps,
            aborting: false,
            activity: failed ? "unload-failed" : (complete ? "unload-complete" : "unload"),
            activityModel: modelName,
            textOnly: true
        };
    }

    function readStatusField(namespace) {
        const field = namespace.source.querySelector("textarea, input");
        return statusSnapshot(namespace, field ? field.value : "");
    }

    function readPrepareStatus(namespace) {
        const telemetry = namespace.runTelemetry;
        const lifecycleSnapshot = modelLifecycleSnapshot(namespace);
        if (lifecycleSnapshot && lifecycleSnapshot.activity === "unload") return lifecycleSnapshot;
        const phase = Array.isArray(telemetry && telemetry.progress_phase)
            ? String(telemetry.progress_phase[0] || "")
            : "";
        if (phase && stageIdFor(phase) !== "prepare") {
            return null;
        }
        const nativeSnapshot = statusSnapshot(namespace, telemetry && telemetry.status);
        if (nativeSnapshot) return nativeSnapshot;
        return lifecycleSnapshot;
    }

    function readReportedPreGenerationStatus(namespace) {
        if (!namespace.activeRun) return null;
        const telemetry = namespace.runTelemetry;
        const reported = telemetry && telemetry.progress_phase;
        const phase = Array.isArray(reported) ? String(reported[0] || "").trim() : "";
        const id = stageIdFor(phase);
        if (!phase || (id !== "input" && id !== "encode")) return null;
        const currentId = namespace.state && namespace.state.currentId;
        if (id === "encode" && currentId && !["prepare", "input", "encode"].includes(currentId)) return null;
        if (id === "input" && currentId && ["decode", "post", "save"].includes(currentId)) return null;
        return {
            id,
            rawName: phase,
            rawMessage: phase,
            metaText: "",
            stageElapsed: null,
            nativeEta: null,
            overallElapsed: namespace.state.overallElapsed,
            progress: null,
            steps: { current: null, total: null },
            aborting: false,
            textOnly: true
        };
    }

    function readSnapshot(namespace) {
        const tracker = findTracker(namespace);
        if (!tracker) return readStatusField(namespace);
        const level = tracker.querySelector(".progress-level-inner");
        const meta = tracker.querySelector(".progress-text");
        if (!level || !meta) return readStatusField(namespace);
        const levelText = String(level.textContent || "").trim();
        const metaText = String(meta.textContent || "").trim();
        if (!levelText && !metaText) return readStatusField(namespace);
        const rawName = stageNameFrom(levelText);
        const timing = parseProgressTiming(levelText);
        const stageElapsed = timing.elapsed;
        const overallElapsed = parseDuration(metaText.split("|").slice(-1)[0]);
        return {
            id: stageIdFor(rawName),
            rawName,
            rawMessage: levelText,
            metaText,
            stageElapsed,
            nativeEta: Number.isFinite(timing.elapsed) && Number.isFinite(timing.total) && timing.total >= timing.elapsed
                ? timing.total - timing.elapsed
                : null,
            overallElapsed,
            progress: parsePercent(levelText, tracker.querySelector(".progress-bar")),
            steps: parseSteps(metaText),
            aborting: false,
            textOnly: false
        };
    }

    function reinterpretQwenSilentEncode(namespace, snapshot) {
        if (!snapshot || snapshot.id !== "denoise") return snapshot;
        const telemetry = namespace.runTelemetry;
        const task = telemetry && telemetry.active_task;
        const settings = task && task.settings;
        const modelType = String(settings && (settings.base_model_type || settings.model_type) || "").toLowerCase();
        if (!modelType.startsWith("qwen_image")) return snapshot;

        const performance = telemetry && telemetry.performance;
        const hasPerformance = performance && typeof performance === "object";
        if (!hasPerformance) {
            if (!namespace.qwenEncodeFallbackStartedAt) namespace.qwenEncodeFallbackStartedAt = Date.now();
            if (Date.now() - namespace.qwenEncodeFallbackStartedAt > 3000) return snapshot;
        } else {
            namespace.qwenEncodeFallbackStartedAt = 0;
        }
        const callbackPhase = optionalNumber(performance && performance.callback_phase);
        const observedSteps = Array.isArray(performance && performance.steps) ? performance.steps : [];
        if ((Number.isFinite(callbackPhase) && callbackPhase >= 2) || observedSteps.length > 0) return snapshot;

        const serverTime = optionalNumber(telemetry && telemetry.server_time);
        const phaseStartedAt = optionalNumber(performance && performance.phase_started_at);
        const encodingElapsed = Number.isFinite(serverTime) && Number.isFinite(phaseStartedAt)
            ? Math.max(0, serverTime - phaseStartedAt)
            : null;
        return {
            ...snapshot,
            id: "encode",
            rawName: "Encoding prompt and references",
            rawMessage: "Qwen is encoding prompt and reference-image conditioning.",
            stageElapsed: encodingElapsed,
            nativeEta: null,
            progress: null,
            steps: { current: null, total: null },
            textOnly: true
        };
    }

    function performanceStageId(label) {
        const text = String(label || "");
        return /(?:upsampl|spatial refin|distilled refinement|ltx\s*2)/i.test(text) ? "post" : "denoise";
    }

    function recoveredPerformanceGroups(run) {
        const groups = new Map();
        (Array.isArray(run && run.step_performance) ? run.step_performance : []).forEach(step => {
            if (!step || typeof step !== "object") return;
            const observerId = String(step.observer_id || "observer");
            const passNo = optionalNumber(step.pass_no);
            const phaseNo = optionalNumber(step.phase);
            const identity = Number.isFinite(passNo) && passNo > 0
                ? `pass-${Math.floor(passNo)}`
                : `phase-${Number.isFinite(phaseNo) ? Math.floor(phaseNo) : 0}`;
            const key = `${observerId}:${identity}`;
            let group = groups.get(key);
            if (!group) {
                group = {
                    key,
                    observerId,
                    identity,
                    passNo: Number.isFinite(passNo) && passNo > 0 ? Math.floor(passNo) : null,
                    phaseNo: Number.isFinite(phaseNo) ? Math.floor(phaseNo) : 0,
                    label: "",
                    stage: "denoise",
                    duration: 0,
                    durations: [],
                    observedSteps: 0,
                    configuredSteps: null,
                    maxStep: 0,
                    firstCompletedAt: null,
                    lastCompletedAt: null
                };
                groups.set(key, group);
            }
            if (step.label) group.label = String(step.label).slice(0, 300);
            group.stage = performanceStageId(group.label);
            const duration = optionalNumber(step.duration_seconds);
            if (Number.isFinite(duration) && duration >= 0 && duration <= 86400) {
                group.duration += duration;
                group.durations.push(duration);
            }
            group.observedSteps += 1;
            const configured = optionalNumber(step.total_steps);
            if (Number.isFinite(configured) && configured > 0) {
                group.configuredSteps = Math.max(group.configuredSteps || 0, Math.floor(configured));
            }
            const stepNo = optionalNumber(step.step);
            if (Number.isFinite(stepNo) && stepNo > 0) group.maxStep = Math.max(group.maxStep, Math.floor(stepNo));
            const completedAt = optionalNumber(step.completed_at);
            if (Number.isFinite(completedAt)) {
                const completedMs = completedAt * 1000;
                group.firstCompletedAt = Number.isFinite(group.firstCompletedAt)
                    ? Math.min(group.firstCompletedAt, completedMs)
                    : completedMs;
                group.lastCompletedAt = Number.isFinite(group.lastCompletedAt)
                    ? Math.max(group.lastCompletedAt, completedMs)
                    : completedMs;
            }
        });
        return Array.from(groups.values())
            .map(group => ({
                ...group,
                complete: Number.isFinite(group.configuredSteps) && group.maxStep >= group.configuredSteps,
                label: group.label || (group.stage === "post"
                    ? "Post-processing"
                    : (Number.isFinite(group.passNo)
                        ? `Denoising pass ${group.passNo}`
                        : (group.phaseNo > 0 ? `Denoising phase ${group.phaseNo + 1}` : "Denoising")))
            }))
            .sort((left, right) => (left.firstCompletedAt || 0) - (right.firstCompletedAt || 0));
    }

    function recoverMissedPerformanceStages(namespace, snapshot = null) {
        const run = namespace && namespace.activeRun;
        const state = namespace && namespace.state;
        if (!run || !state || !state.records) return false;
        const groups = recoveredPerformanceGroups(run);
        if (!groups.length) return false;
        const currentStage = snapshot && snapshot.id;
        const currentRank = STAGE_DEFS.findIndex(def => def.id === currentStage);
        let recovered = false;

        for (const stageId of ["denoise", "post"]) {
            const record = state.records[stageId];
            if (!record || record.state !== "pending") continue;
            const stageRank = STAGE_DEFS.findIndex(def => def.id === stageId);
            const stageWasPassed = currentRank >= 0 && stageRank >= 0 && currentRank > stageRank;
            const stageIsCurrent = currentStage === stageId;
            const eligible = groups.filter(group =>
                group.stage === stageId && (group.complete || stageWasPassed || stageIsCurrent)
            );
            if (!eligible.length) continue;

            const elapsed = eligible.reduce((total, group) => total + group.duration, 0);
            const configuredSteps = eligible.reduce(
                (total, group) => total + (Number.isFinite(group.configuredSteps) ? group.configuredSteps : 0),
                0
            );
            const observedSteps = eligible.reduce(
                (total, group) => total + (Number.isFinite(group.maxStep) ? group.maxStep : group.observedSteps),
                0
            );
            const durations = eligible.flatMap(group => group.durations).filter(value => value > 0).sort((a, b) => a - b);
            const middle = Math.floor(durations.length / 2);

            record.visible = true;
            record.state = "complete";
            record.preloaded = false;
            record.unreported = false;
            record.recovered = true;
            record.rawName = stageId === "denoise" ? "Denoising (recovered)" : "Post-processing (recovered)";
            record.rawMessage = `Recovered from ${eligible.reduce((total, group) => total + group.observedSteps, 0)} retained WanGP callback observations after browser activity resumed.`;
            record.startedAt = null;
            record.elapsed = elapsed;
            record.elapsedBase = elapsed;
            record.reportedElapsed = elapsed;
            record.reportedAt = Date.now();
            record.progress = (stageWasPassed || eligible.every(group => group.complete)) ? 100 : null;
            record.eta = record.progress === 100 ? 0 : null;
            record.stepCurrent = observedSteps || null;
            record.stepTotal = configuredSteps || null;
            record.lastStepCurrent = record.stepCurrent;
            record.lastStepElapsed = elapsed;
            record.lastStepAt = null;
            record.stepSamples = durations.slice(-6);
            record.stepSeconds = durations.length
                ? (durations.length % 2 ? durations[middle] : (durations[middle - 1] + durations[middle]) / 2)
                : null;

            eligible.forEach(group => {
                if (!(group.complete || stageWasPassed)) return;
                const safeObserver = group.observerId.replace(/[^a-z0-9_-]+/gi, "-").slice(0, 80);
                const phaseId = `${stageId}:recovered-${safeObserver}-${group.identity}`;
                if (state.phases[phaseId]) return;
                state.phases[phaseId] = {
                    id: phaseId,
                    label: group.label,
                    stage: stageId,
                    startedAt: Number.isFinite(group.firstCompletedAt)
                        ? group.firstCompletedAt - group.duration * 1000
                        : null,
                    elapsed: group.duration,
                    state: "complete",
                    recovered: true
                };
                state.phaseOrder.push(phaseId);
            });
            recovered = true;
        }

        if (recovered && state.records.denoise.recovered) {
            markStageUnreported(
                state,
                "prepare",
                "Prepare completed while backgrounded",
                "The browser did not observe Prepare while the WanGP window was minimized; its duration is unavailable."
            );
            markEncodeUnreported(state);
        }
        if (recovered) state.inactiveSince = 0;
        return recovered;
    }

    function finishStage(state, id) {
        if (!id) return;
        const record = state.records[id];
        if (!record || record.state !== "current") return;
        if (record.startedAt) {
            record.elapsed = (Number.isFinite(record.elapsedBase) ? record.elapsedBase : 0) +
                (Date.now() - record.startedAt) / 1000;
        }
        record.state = "complete";
        record.progress = 100;
        record.eta = 0;
        if (id === "denoise" && Number.isFinite(record.stepTotal)) {
            record.stepCurrent = record.stepTotal;
        }
    }

    function markPreparePreloaded(state) {
        const record = state.records.prepare;
        if (!record || record.state !== "pending") return;
        record.visible = true;
        record.state = "complete";
        record.preloaded = true;
        record.unreported = false;
        record.rawName = "Model preloaded";
        record.rawMessage = "The required model was already loaded and ready for this run.";
        record.startedAt = null;
        record.elapsed = 0;
        record.elapsedBase = 0;
        record.reportedElapsed = 0;
        record.progress = 100;
        record.eta = 0;
    }

    function markStageUnreported(state, id, rawName, rawMessage) {
        const record = state.records[id];
        if (!record || record.state !== "pending") return;
        record.visible = true;
        record.state = "complete";
        record.preloaded = false;
        record.unreported = true;
        record.recovered = true;
        record.rawName = rawName;
        record.rawMessage = rawMessage;
        record.startedAt = null;
        record.elapsed = null;
        record.elapsedBase = 0;
        record.reportedElapsed = null;
        record.reportedAt = null;
        record.progress = 100;
        record.eta = 0;
    }

    function markEncodeUnreported(state) {
        markStageUnreported(
            state,
            "encode",
            "Encode status not reported",
            "Wan2GP did not expose a separately measurable Encode phase for this run. Any required prompt, reference, or input conditioning completed before generation began."
        );
    }

    function updateEta(record) {
        if (!Number.isFinite(record.progress) || record.progress <= 0 || record.progress >= 100 || !Number.isFinite(record.elapsed) || record.elapsed <= 0) {
            record.eta = record.progress >= 100 ? 0 : null;
            return;
        }
        const overallEstimate = record.elapsed * (100 - record.progress) / record.progress;
        let estimate = overallEstimate;
        const useful = record.samples.filter(sample => sample.progress <= record.progress);
        if (useful.length >= 2) {
            const latest = useful[useful.length - 1];
            let earlier = useful[0];
            for (let index = useful.length - 2; index >= 0; index -= 1) {
                if (latest.elapsed - useful[index].elapsed >= 2 && latest.progress - useful[index].progress >= 0.2) {
                    earlier = useful[index];
                    break;
                }
            }
            const elapsedDelta = latest.elapsed - earlier.elapsed;
            const progressDelta = latest.progress - earlier.progress;
            if (elapsedDelta > 0 && progressDelta > 0) {
                const recentEstimate = (100 - record.progress) / (progressDelta / elapsedDelta);
                estimate = recentEstimate * 0.65 + overallEstimate * 0.35;
            }
        }
        estimate = clamp(estimate, 0, 86400);
        record.eta = Number.isFinite(record.eta) ? record.eta * 0.72 + estimate * 0.28 : estimate;
    }

    function updateDenoiseEta(record) {
        if (!Number.isFinite(record.stepCurrent) || !Number.isFinite(record.stepTotal) ||
            !Number.isFinite(record.elapsed) || record.stepCurrent <= 0 || record.elapsed <= 0) {
            record.eta = null;
            return;
        }
        if (record.stepCurrent >= record.stepTotal) {
            record.eta = 0;
            return;
        }
        const fallbackStepSeconds = record.elapsed / record.stepCurrent;
        const fallbackEta = fallbackStepSeconds * (record.stepTotal - record.stepCurrent);
        const nativeEta = optionalNumber(record.nativeEta);
        record.eta = clamp(Number.isFinite(nativeEta) ? nativeEta : fallbackEta, 0, 86400);
        record.stepSeconds = (record.elapsed + record.eta) / record.stepTotal;
    }

    function updateStepTiming(record, steps, now = Date.now()) {
        if (record.id !== "denoise" && record.id !== "post") return;
        const current = optionalNumber(steps && steps.current);
        const total = optionalNumber(steps && steps.total);
        record.stepCurrent = current;
        record.stepTotal = total;
        if (!Number.isFinite(current) || !Number.isFinite(record.elapsed)) return;

        if (!Number.isFinite(record.lastStepCurrent) || !Number.isFinite(record.lastStepElapsed)) {
            record.lastStepCurrent = current;
            record.lastStepElapsed = record.elapsed;
            record.lastStepAt = now;
            record.stepSeconds = null;
            return;
        }

        if (current < record.lastStepCurrent) {
            record.lastStepCurrent = current;
            record.lastStepElapsed = record.elapsed;
            record.lastStepAt = now;
            record.stepSamples = [];
            record.stepSeconds = null;
            return;
        }

        if (current === record.lastStepCurrent) return;
        const elapsedDelta = Number.isFinite(record.lastStepAt)
            ? (now - record.lastStepAt) / 1000
            : record.elapsed - record.lastStepElapsed;
        const stepDelta = current - record.lastStepCurrent;
        record.lastStepCurrent = current;
        record.lastStepElapsed = record.elapsed;
        record.lastStepAt = now;
        if (elapsedDelta <= 0 || stepDelta <= 0) return;

        record.stepSamples.push(elapsedDelta / stepDelta);
        record.stepSamples = record.stepSamples.slice(-6);
        const sorted = record.stepSamples.slice().sort((left, right) => left - right);
        const middle = Math.floor(sorted.length / 2);
        record.stepSeconds = sorted.length % 2
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2;
    }

    function applySnapshot(namespace, snapshot) {
        const state = namespace.state;
        const now = Date.now();
        if (!namespace.activeRun && state.inactiveSince && now - state.inactiveSince > RESET_AFTER_MS) {
            finishStage(state, state.currentId);
            resetJob(namespace);
        }
        const activeState = namespace.state;
        activeState.inactiveSince = 0;
        activeState.lastSeenAt = now;
        if (namespace.activeRun && activeState.currentId === null &&
            (snapshot.id === "input" || snapshot.id === "encode" || snapshot.id === "denoise")) {
            markPreparePreloaded(activeState);
        }
        if (namespace.activeRun && snapshot.id === "denoise") markEncodeUnreported(activeState);
        applyPhase(activeState, snapshot, now);

        if (activeState.currentId !== snapshot.id) {
            finishStage(activeState, activeState.currentId);
            activeState.currentId = snapshot.id;
            const next = activeState.records[snapshot.id];
            const recoveredBaseline = next.recovered && next.state === "complete" && Number.isFinite(next.elapsed);
            const elapsedBase = (recoveredBaseline || (snapshot.id === "input" && next.state === "complete" && Number.isFinite(next.elapsed)))
                ? next.elapsed
                : 0;
            next.state = snapshot.aborting ? "aborting" : "current";
            next.preloaded = false;
            next.unreported = false;
            next.activity = null;
            next.activityModel = "";
            next.visible = true;
            next.startedAt = now;
            next.elapsed = elapsedBase || null;
            next.elapsedBase = elapsedBase;
            next.reportedElapsed = null;
            next.reportedAt = null;
            next.nativeEta = null;
            next.progress = null;
            next.eta = null;
            next.samples = [];
            next.stepCurrent = null;
            next.stepTotal = null;
            next.lastStepCurrent = null;
            next.lastStepElapsed = null;
            next.lastStepAt = null;
            next.stepSamples = [];
            next.stepSeconds = null;
            next.recovered = Boolean(recoveredBaseline);
            if (!activeState.selectionIsManual || !activeState.selectedId) activeState.selectedId = snapshot.id;
        }

        const record = activeState.records[snapshot.id];
        record.visible = true;
        record.state = snapshot.aborting ? "aborting" : "current";
        record.rawName = snapshot.rawName;
        record.rawMessage = snapshot.rawMessage;
        record.activity = snapshot.activity || null;
        record.activityModel = String(snapshot.activityModel || "");
        // WanGP exposes Decode as a blocking VAE operation. Its progress value is
        // either a static 0% or the completed denoising bar, not decoder progress.
        record.progress = snapshot.id === "decode" ? null : snapshot.progress;
        record.elapsed = (Number.isFinite(record.elapsedBase) ? record.elapsedBase : 0) +
            (now - record.startedAt) / 1000;
        record.reportedElapsed = Number.isFinite(snapshot.stageElapsed) ? snapshot.stageElapsed : null;
        record.reportedAt = now;
        record.nativeEta = optionalNumber(snapshot.nativeEta);
        const lastSample = record.samples[record.samples.length - 1];
        if (Number.isFinite(record.progress) && Number.isFinite(record.elapsed) &&
            (!lastSample || lastSample.progress !== record.progress || Math.abs(lastSample.elapsed - record.elapsed) >= 1)) {
            record.samples.push({ progress: record.progress, elapsed: record.elapsed });
            record.samples = record.samples.slice(-30);
        }
        updateStepTiming(record, snapshot.steps, now);
        if (snapshot.aborting) record.eta = null;
        else if (record.id === "denoise") updateDenoiseEta(record);
        else updateEta(record);
        activeState.overallElapsed = snapshot.overallElapsed;
        const denoise = activeState.records.denoise;
        activeState.steps = snapshot.id === "decode" && Number.isFinite(denoise.stepTotal)
            ? { current: denoise.stepTotal, total: denoise.stepTotal }
            : snapshot.steps;
    }

    function stageTimeText(state, record) {
        if (record.preloaded) return "Preloaded";
        if (record.unreported) return "Not reported";
        if (record.state === "complete") return formatDuration(record.elapsed);
        if (record.state === "aborting") return Number.isFinite(record.elapsed) ? `${formatDuration(record.elapsed)} elapsed` : "Stopping…";
        if (record.state === "current") {
            const remaining = remainingEstimate(state, record);
            if (Number.isFinite(remaining)) return `${formatDuration(remaining, true)} left`;
            return Number.isFinite(record.elapsed) ? `${formatDuration(record.elapsed)} elapsed` : "Calculating…";
        }
        return "—";
    }

    function stageDisplayLabel(namespace, record) {
        if (!record || record.id !== "denoise" || !Number.isFinite(namespace.state.activeDenoisePhase)) {
            return record ? record.label : "";
        }
        const total = optionalNumber(namespace.activeRun && namespace.activeRun.settings && namespace.activeRun.settings.guidance_phases);
        const suffix = Number.isFinite(total) && total >= 2
            ? `Phase ${namespace.state.activeDenoisePhase} of ${Math.floor(total)}`
            : `Phase ${namespace.state.activeDenoisePhase}`;
        return `${record.label} · ${suffix}`;
    }

    function statusLabel(record) {
        if (record.preloaded) return "Preloaded";
        if (record.unreported) return "Not reported";
        if (record.activity === "unload") return "Unloading";
        if (record.activity === "unload-complete") return "Unloaded";
        if (record.activity === "unload-failed") return "Failed";
        if (record.state === "complete") return "Completed";
        if (record.state === "aborting") return "Aborting";
        if (record.state === "current") return "Running";
        return "Pending";
    }

    function stageSupportsEta(record) {
        if (!record) return false;
        if (record.id === "denoise") return true;
        return record.id === "post" &&
            ((Number.isFinite(record.progress) && record.progress > 0 && record.progress < 100) ||
                (Number.isFinite(record.eta) && record.eta > 0));
    }

    function remainingEstimate(state, record) {
        if (!stageSupportsEta(record)) return null;
        return Number.isFinite(record.eta) && record.eta > 0 ? record.eta : null;
    }

    function totalEta(state) {
        const current = state.records[state.currentId];
        return current ? remainingEstimate(state, current) : null;
    }

    function text(root, selector, value) {
        const element = root.querySelector(selector);
        if (element) element.textContent = value == null ? "" : String(value);
    }

    function setting(settings, ...keys) {
        for (const key of keys) {
            if (settings && settings[key] !== null && settings[key] !== undefined && settings[key] !== "") {
                return settings[key];
            }
        }
        return null;
    }

    function settingText(value) {
        if (value === null || value === undefined || value === "") return "—";
        if (Array.isArray(value)) return value.map(item => settingText(item)).join(", ");
        if (typeof value === "object") return JSON.stringify(value);
        return String(value);
    }

    function attentionModeLabel(settings) {
        const value = setting(settings, "attention_mode", "override_attention");
        if (value === null) return null;
        const raw = String(value).trim();
        const labels = {
            auto: "Auto",
            sdpa: "SDPA",
            flash: "Flash Attention",
            xformers: "xFormers",
            sage: "Sage Attention",
            sage2: "Sage Attention 2",
            sage3: "Sage Attention 3",
            radial: "Radial Attention",
            sol: "Sol Attention"
        };
        const label = labels[raw.toLowerCase()] || raw;
        const tau = setting(settings, "attention_sparsity");
        return raw.toLowerCase() === "sol" && tau !== null ? `${label} · tau ${tau}` : label;
    }

    function compactModelName(value) {
        if (Array.isArray(value)) return value.map(item => compactModelName(item)).join(", ");
        const raw = settingText(value);
        if (raw === "—") return raw;
        const clean = raw.split(/[?#]/, 1)[0].replace(/\\/g, "/").replace(/\/+$/, "");
        const filename = clean.slice(clean.lastIndexOf("/") + 1);
        if (!filename) return raw;
        try {
            return decodeURIComponent(filename);
        } catch (_) {
            return filename;
        }
    }

    function compactLoraNames(value) {
        const values = Array.isArray(value) ? value : [value];
        return values
            .map(item => compactModelName(item).replace(/\.safetensors$/i, ""))
            .filter(item => item && item !== "—")
            .join("\n");
    }

    function stageModelInfo(namespace, record) {
        if (record && /^unload/.test(String(record.activity || "")) && record.activityModel) {
            const name = compactModelName(record.activityModel);
            return {
                role: "Outgoing model",
                itemRole: "Outgoing model",
                names: [name],
                text: `Outgoing model: ${name}`
            };
        }
        const settings = namespace.activeRun && namespace.activeRun.settings;
        let components = settings && settings.component_models;
        const activityText = `${record && record.rawName || ""} ${record && record.rawMessage || ""}`;
        if (settings && record && ["prepare", "input", "encode", "decode"].includes(record.id)) {
            const metadata = normalizePostprocessingMetadata(settings);
            const isPostprocessingStage = metadata && (
                metadata.application === "late" || /upsampl|spatial refin/i.test(activityText)
            );
            const backingOperation = isPostprocessingStage && metadata.operations.find(operation =>
                operation && operation.kind === "spatial_upsampling" && operation.model
            );
            if (backingOperation) {
                // Never fall through to the generation model during an LTX stage.
                // Exact WanGP filenames win; role-accurate LTX labels fill any
                // components the model definition does not expose.
                components = completedPostprocessingComponents(backingOperation.model);
            }
        }
        if (record && record.id === "input" &&
            !/(?:vae|latent|auto.?encod|encod|decod)/i.test(`${record.rawName || ""} ${record.rawMessage || ""}`)) {
            return null;
        }
        const raw = components && record ? components[record.id] : null;
        const values = Array.isArray(raw) ? raw : (raw ? [raw] : []);
        const names = values
            .map(value => compactModelName(value))
            .filter(value => value && value !== "—");
        if (!names.length) return null;
        const roles = {
            prepare: ["Transformer", "Transformers"],
            input: ["Input VAE", "Input VAEs"],
            encode: ["Text encoder", "Text encoders"],
            decode: ["VAE", "VAEs"]
        };
        const roleNames = roles[record.id] || ["Model", "Models"];
        const itemRole = roleNames[0];
        const role = names.length > 1 ? roleNames[1] : itemRole;
        return {
            role,
            itemRole,
            names,
            text: `${role}: ${names.join(" + ")}`
        };
    }

    function renderIdle(namespace) {
        const idle = namespace.panel.querySelector("[data-sp-idle]");
        const running = namespace.panel.querySelector("[data-sp-running]");
        if (!idle || !running) return;
        idle.hidden = false;
        running.hidden = true;
        text(namespace.panel, "[data-sp-live]", "Ready");
        text(namespace.panel, "[data-sp-steps]", "");
        text(namespace.panel, "[data-sp-overall]", "");
        text(namespace.panel, "[data-sp-eta]", "");
        text(namespace.panel, "[data-sp-idle-title]", "Ready to generate");
        text(namespace.panel, "[data-sp-idle-message]", "Live stage timing and telemetry will appear here during the next run.");
    }

    function readDownloadSnapshot(namespace) {
        const field = namespace.container.querySelector(
            "#status-lite-download-bridge textarea, #status-lite-download-bridge input"
        );
        const raw = String(field ? field.value : "").trim();
        if (!raw || raw === namespace.downloadRaw) return namespace.download;
        namespace.downloadRaw = raw;
        try {
            const parsed = JSON.parse(raw);
            namespace.download = parsed && typeof parsed === "object" ? parsed : null;
        } catch (_) {
            namespace.download = null;
        }
        return namespace.download;
    }

    function downloadPanelVisible(namespace) {
        const download = namespace.download;
        if (!download || !download.visible || !Array.isArray(download.files) || !download.files.length) return false;
        return Boolean(download.active || namespace.state.currentId === "prepare");
    }

    function downloadFileStats(file) {
        const state = String(file.state || "pending");
        const downloaded = optionalNumber(file.downloaded);
        const total = optionalNumber(file.total);
        if (state === "failed") return String(file.error || "Download failed");
        if (state === "complete") return Number.isFinite(total) && total > 0 ? formatBytes(total) : "Complete";
        if (state === "pending") return Number.isFinite(total) && total > 0 ? `${formatBytes(total)} · Pending` : "Pending";
        const parts = [];
        if (Number.isFinite(downloaded)) {
            parts.push(Number.isFinite(total) && total > 0
                ? `${formatBytes(downloaded)} / ${formatBytes(total)}`
                : formatBytes(downloaded));
        }
        return parts.join(" · ") || "Starting…";
    }

    function downloadFreshnessText(file, nowSeconds) {
        const state = String(file.state || "pending");
        const startedAt = optionalNumber(file.started_at);
        const lastByteAt = optionalNumber(file.last_byte_at);
        const elapsed = Number.isFinite(startedAt) ? Math.max(0, nowSeconds - startedAt) : null;
        if (state === "pending") return "Waiting to start";
        if (state === "complete" || state === "failed") return "";
        const elapsedText = Number.isFinite(elapsed) ? `${formatDuration(elapsed)} elapsed` : "Starting…";
        if (!Number.isFinite(lastByteAt)) return `${elapsedText} · Connecting`;
        const quietFor = Math.max(0, nowSeconds - lastByteAt);
        if (quietFor < 3) return `${elapsedText} · Receiving data`;
        if (quietFor < 8) return `${elapsedText} · Last byte ${formatDuration(quietFor, true)} ago`;
        return `${elapsedText} · Waiting for next transfer update (${formatDuration(quietFor, true)} since last byte)`;
    }

    function downloadCycleText(file) {
        const state = String(file.state || "pending");
        if (state === "complete" || state === "failed" || state === "pending") return "";
        const cycles = optionalNumber(file.transfer_cycles) || 0;
        if (cycles < 1) return "ETA learning transfer pattern…";
        const parts = [`${cycles} transfer ${cycles === 1 ? "cycle" : "cycles"} observed`];
        const averageBytes = optionalNumber(file.cycle_average_bytes);
        const averageSeconds = optionalNumber(file.cycle_average_seconds);
        if (Number.isFinite(averageBytes) && Number.isFinite(averageSeconds)) {
            parts.push(`Avg cycle: ${formatBytes(averageBytes)} / ${formatDuration(averageSeconds, true)}`);
        }
        const effectiveRate = optionalNumber(file.effective_rate);
        if (Number.isFinite(effectiveRate) && effectiveRate > 0) parts.push(`Effective rate: ${formatRate(effectiveRate)}`);
        const eta = optionalNumber(file.effective_eta);
        if (Number.isFinite(eta) && eta > 0) parts.push(`Estimated remaining: ${formatDuration(eta, true)}`);
        return parts.join(" · ");
    }

    function downloadHeaderEta(download) {
        const files = Array.isArray(download && download.files) ? download.files : [];
        const remaining = files.filter(file => ["pending", "downloading", "retrying"].includes(String(file && file.state || "")));
        if (!remaining.length) return null;
        const estimates = remaining.map(file => optionalNumber(file && file.effective_eta));
        if (estimates.some(estimate => !Number.isFinite(estimate) || estimate < 0)) return null;
        return Math.max(...estimates);
    }

    function renderDownloads(namespace) {
        const panel = namespace.panel.querySelector("[data-sp-downloads]");
        const detail = namespace.panel.querySelector("[data-sp-detail]");
        if (!panel) return;
        const visible = downloadPanelVisible(namespace);
        panel.hidden = !visible;
        if (detail) detail.hidden = visible;
        if (!visible) return;

        const download = namespace.download;
        const totals = download.totals || {};
        const failed = Number(totals.failed || 0);
        const title = download.active
            ? "Downloading model files"
            : (failed > 0 ? "Download completed with errors" : "Downloads complete");
        text(panel, "[data-sp-download-title]", title);
        panel.dataset.active = download.active ? "true" : "false";

        const count = Number(totals.file_count || download.files.length || 0);
        const completed = Number(totals.completed || 0);
        const summary = [`${completed}/${count} files`];
        const knownDownloaded = optionalNumber(totals.known_downloaded);
        const knownTotal = optionalNumber(totals.known_total);
        if (Number.isFinite(knownDownloaded) && Number.isFinite(knownTotal) && knownTotal > 0) {
            summary.push(`${formatBytes(knownDownloaded)} / ${formatBytes(knownTotal)}`);
        } else if (optionalNumber(totals.downloaded) > 0) {
            summary.push(`${formatBytes(optionalNumber(totals.downloaded))} received`);
        }
        text(panel, "[data-sp-download-summary]", summary.join(" · "));

        const totalDetails = [];
        const transferStartedAt = optionalNumber(download.started_at);
        if (download.active && Number.isFinite(transferStartedAt)) {
            totalDetails.push(`${formatDuration(Math.max(0, Date.now() / 1000 - transferStartedAt))} elapsed`);
        }
        text(panel, "[data-sp-download-total]", totalDetails.join(" · "));

        const overallFill = panel.querySelector("[data-sp-download-overall-fill]");
        if (overallFill) {
            const percent = Number.isFinite(knownDownloaded) && Number.isFinite(knownTotal) && knownTotal > 0
                ? clamp(knownDownloaded / knownTotal * 100, 0, 100)
                : (download.active ? 0 : 100);
            overallFill.style.width = `${percent}%`;
        }

        const stateOrder = { downloading: 0, retrying: 1, failed: 2, pending: 3, complete: 4 };
        const files = download.files.slice().sort((left, right) =>
            (stateOrder[left.state] ?? 9) - (stateOrder[right.state] ?? 9)
        );
        const nowSeconds = Date.now() / 1000;
        const fileContainer = panel.querySelector("[data-sp-download-files]");
        const fragment = document.createDocumentFragment();
        files.forEach(file => {
            const row = document.createElement("div");
            const state = String(file.state || "pending");
            row.className = "status-lite__download-file";
            row.dataset.state = state;
            const icon = document.createElement("span");
            icon.className = "status-lite__download-file-icon";
            icon.setAttribute("aria-hidden", "true");
            icon.textContent = state === "complete" ? "✓" : (state === "failed" ? "!" : (state === "downloading" || state === "retrying" ? "↓" : "○"));
            const name = document.createElement("span");
            name.className = "status-lite__download-file-name";
            name.textContent = String(file.name || "Unknown file");
            name.title = String(file.path || file.name || "");
            const stats = document.createElement("span");
            stats.className = "status-lite__download-file-stats";
            stats.textContent = downloadFileStats(file);
            stats.title = String(file.error || "");
            const freshness = document.createElement("span");
            freshness.className = "status-lite__download-file-freshness";
            freshness.textContent = downloadFreshnessText(file, nowSeconds);
            const cycles = document.createElement("span");
            cycles.className = "status-lite__download-file-cycles";
            cycles.textContent = downloadCycleText(file);
            const bar = document.createElement("span");
            bar.className = "status-lite__download-file-bar";
            const fill = document.createElement("span");
            const fileTotal = optionalNumber(file.total);
            const fileDownloaded = optionalNumber(file.downloaded);
            const filePercent = state === "complete"
                ? 100
                : (Number.isFinite(fileTotal) && fileTotal > 0 && Number.isFinite(fileDownloaded)
                    ? clamp(fileDownloaded / fileTotal * 100, 0, 100)
                    : 0);
            fill.style.width = `${filePercent}%`;
            bar.appendChild(fill);
            row.append(icon, name, stats, freshness, cycles, bar);
            fragment.appendChild(row);
        });
        fileContainer.replaceChildren(fragment);
    }

    function stageActivities(state, record) {
        if (!record || record.id !== "input") return [];
        const seen = new Set();
        return state.phaseOrder
            .filter(id => {
                if (seen.has(id)) return false;
                seen.add(id);
                return true;
            })
            .map(id => state.phases[id])
            .filter(phase => phase && phase.stage === "input")
            .map(phase => ({
                label: phase.label,
                state: phase.state,
                elapsed: phase.elapsed
            }));
    }

    function detailMessage(record, hasActivities = false) {
        if (record.preloaded) return record.rawMessage || "The required model was already loaded and ready for this run.";
        if (record.unreported) return record.rawMessage || "Wan2GP did not expose a separately measurable status for this stage.";
        if (record.state === "complete") {
            if (record.id === "input" && hasActivities) return "";
            return Number.isFinite(record.elapsed) ? `Completed in ${formatDuration(record.elapsed)}.` : "Completed.";
        }
        if (record.state === "aborting") return record.rawMessage || "Wan2GP is stopping the current run.";
        if (/^unload/.test(String(record.activity || ""))) return record.rawMessage;
        if (record.id === "decode" && record.state === "current") {
            return "Decoding is running. Wan2GP does not report intermediate VAE progress for this model.";
        }
        let message = String(record.rawMessage || "").trim();
        const rawName = String(record.rawName || "").trim();
        if (rawName && message.toLowerCase().startsWith(rawName.toLowerCase())) {
            message = message.slice(rawName.length).replace(/^[\s|:\-–—]+/, "").trim();
        }
        message = message.replace(/\s+-\s+/g, " · ");
        if (message) return message;
        if (record.id === "input" && hasActivities) return "";
        if (record.state === "pending") return "This stage has not started.";
        return "Live stage timing.";
    }

    function renderStages(namespace) {
        const state = namespace.state;
        const container = namespace.panel.querySelector("[data-sp-stages]");
        const existing = new Map(Array.from(container.querySelectorAll("[data-stage-id]")).map(button => [button.dataset.stageId, button]));

        const visibleDefs = STAGE_DEFS.filter(def => state.records[def.id].visible);
        container.dataset.stageCount = String(visibleDefs.length);
        visibleDefs.forEach((def, visibleIndex) => {
            const record = state.records[def.id];
            let button = existing.get(def.id);
            if (!button) {
                button = document.createElement("button");
                button.type = "button";
                button.className = "status-lite__stage";
                button.dataset.stageId = def.id;
                button.setAttribute("role", "tab");
                const icon = document.createElement("span");
                icon.className = "status-lite__stage-icon";
                icon.setAttribute("aria-hidden", "true");
                const name = document.createElement("span");
                name.className = "status-lite__stage-name";
                const timing = document.createElement("span");
                timing.className = "status-lite__stage-time";
                button.append(icon, name, timing);
            }
            const selected = state.selectedId === def.id;
            button.classList.toggle("status-lite__stage--complete", record.state === "complete");
            const active = record.state === "current" || record.state === "aborting";
            button.classList.toggle("status-lite__stage--current", active);
            button.classList.toggle("status-lite__stage--selected", selected);
            button.setAttribute("aria-selected", selected ? "true" : "false");
            const label = stageDisplayLabel(namespace, record);
            const modelInfo = stageModelInfo(namespace, record);
            const modelSummary = modelInfo ? `, ${modelInfo.text}` : "";
            const accessibleLabel = `${record.rawName || label}: ${statusLabel(record)}, ${stageTimeText(state, record)}${modelSummary}`;
            button.setAttribute("aria-label", accessibleLabel);
            button.title = accessibleLabel;
            button.querySelector(".status-lite__stage-icon").textContent = record.state === "complete" ? "✓" : (record.state === "aborting" ? "!" : (record.state === "current" ? "●" : String(visibleIndex + 1)));
            button.querySelector(".status-lite__stage-name").textContent = label;
            button.querySelector(".status-lite__stage-time").textContent = stageTimeText(state, record);
            const buttonAtIndex = container.children[visibleIndex] || null;
            if (buttonAtIndex !== button) container.insertBefore(button, buttonAtIndex);
        });
        existing.forEach((button, id) => {
            if (!state.records[id] || !state.records[id].visible) button.remove();
        });

        // A fixed breakpoint made four-card runs stack unnecessarily on half-width
        // layouts. Use the actual card count so inline contents are retained whenever
        // every card can keep its intended flex basis without crowding.
        const inlineWidth = visibleDefs.length
            ? (visibleDefs.length * 150) + 80 + (Math.max(0, visibleDefs.length - 1) * 7)
            : 0;
        container.classList.toggle("status-lite__stages--inline", container.clientWidth >= inlineWidth);
    }

    function renderDetail(namespace) {
        const state = namespace.state;
        const selected = state.records[state.selectedId] || state.records[state.currentId] || state.records.prepare;
        const activities = stageActivities(state, selected);
        let etaText = "—";
        const remaining = remainingEstimate(state, selected);
        if (selected.state === "current" && Number.isFinite(remaining)) etaText = formatDuration(remaining, true);
        else if (selected.state === "current" && stageSupportsEta(selected)) etaText = "Calculating…";

        text(namespace.panel, "[data-sp-detail-name]", stageDisplayLabel(namespace, selected));
        const activityElement = namespace.panel.querySelector("[data-sp-detail-activities]");
        if (activityElement) {
            activityElement.hidden = activities.length === 0;
            activityElement.replaceChildren();
            activities.forEach(activity => {
                const line = document.createElement("span");
                line.className = "status-lite__detail-activity-line";
                const active = activity.state === "current" || activity.state === "aborting";
                const icon = activity.state === "complete" ? "✓" : (activity.state === "aborting" ? "!" : "●");
                const timing = Number.isFinite(activity.elapsed)
                    ? `${formatDuration(activity.elapsed)}${active ? " elapsed" : ""}`
                    : (active ? "Running" : "Completed");
                line.textContent = `${icon} ${activity.label} · ${timing}`;
                activityElement.appendChild(line);
            });
        }
        const detail = detailMessage(selected, activities.length > 0);
        const detailElement = namespace.panel.querySelector("[data-sp-detail-message]");
        if (detailElement) {
            detailElement.hidden = !detail;
            detailElement.textContent = detail;
        }
        const modelInfo = stageModelInfo(namespace, selected);
        const modelElement = namespace.panel.querySelector("[data-sp-detail-model]");
        if (modelElement) {
            modelElement.hidden = !modelInfo;
            modelElement.replaceChildren();
            modelElement.classList.toggle("status-lite__detail-model--list", Boolean(modelInfo && modelInfo.names.length > 1));
            if (modelInfo && modelInfo.names.length > 1) {
                modelElement.setAttribute("role", "list");
                for (const name of modelInfo.names) {
                    const line = document.createElement("span");
                    line.className = "status-lite__detail-model-line";
                    line.setAttribute("role", "listitem");
                    line.textContent = `${modelInfo.itemRole}: ${name}`;
                    line.title = name;
                    modelElement.append(line);
                }
            } else {
                modelElement.removeAttribute("role");
                modelElement.textContent = modelInfo ? modelInfo.text : "";
            }
            modelElement.title = modelInfo ? modelInfo.text : "";
        }
        text(namespace.panel, "[data-sp-detail-state]", statusLabel(selected));
        text(namespace.panel, "[data-sp-detail-elapsed]", Number.isFinite(selected.elapsed) ? formatDuration(selected.elapsed) : "—");
        text(namespace.panel, "[data-sp-detail-eta]", etaText);
        text(namespace.panel, "[data-sp-detail-progress]", Number.isFinite(selected.progress) ? `${selected.progress.toFixed(1)}%` : "—");
        const etaMetric = namespace.panel.querySelector("[data-sp-eta-metric]");
        if (etaMetric) etaMetric.hidden = selected.state !== "current" || !stageSupportsEta(selected);
        const progressMetric = namespace.panel.querySelector("[data-sp-progress-metric]");
        if (progressMetric) progressMetric.hidden = selected.id === "decode" || /^unload/.test(String(selected.activity || ""));
        const stepMetric = namespace.panel.querySelector("[data-sp-step-metric]");
        if (stepMetric) {
            const showStepTime = (selected.id === "denoise" || selected.id === "post") &&
                (Number.isFinite(selected.stepCurrent) || Number.isFinite(selected.stepTotal));
            stepMetric.hidden = !showStepTime;
            text(stepMetric, "[data-sp-detail-step-time]", Number.isFinite(selected.stepSeconds)
                ? `${formatLiveStepDuration(selected.stepSeconds)}/step`
                : "Calculating…");
        }
    }

    function applyCollapsed(namespace) {
        const body = namespace.panel.querySelector("[data-sp-body]");
        const button = namespace.panel.querySelector("[data-sp-collapse]");
        if (!body || !button) return;
        body.hidden = namespace.collapsed;
        namespace.panel.classList.toggle("status-lite--collapsed", namespace.collapsed);
        button.setAttribute("aria-expanded", namespace.collapsed ? "false" : "true");
        button.setAttribute("aria-label", namespace.collapsed ? "Expand Status Lite" : "Collapse Status Lite");
        button.title = namespace.collapsed ? "Expand Status Lite" : "Collapse Status Lite";
        button.textContent = namespace.collapsed ? "◀" : "▼";
    }

    function render(namespace) {
        const state = namespace.state;
        const current = state.records[state.currentId];
        const downloading = namespace.download && namespace.download.active;
        const idle = namespace.panel.querySelector("[data-sp-idle]");
        const running = namespace.panel.querySelector("[data-sp-running]");
        if (idle) idle.hidden = true;
        if (running) running.hidden = false;
        text(namespace.panel, "[data-sp-live]", downloading ? "Downloading model files" : (current ? (current.rawName || current.label) : "Waiting for progress"));
        text(namespace.panel, "[data-sp-steps]", Number.isFinite(state.steps.current) && Number.isFinite(state.steps.total) ? `${state.steps.current}/${state.steps.total} steps` : "");
        text(namespace.panel, "[data-sp-overall]", Number.isFinite(state.overallElapsed) ? `${formatDuration(state.overallElapsed)} elapsed` : "");
        const downloadEta = downloading ? downloadHeaderEta(namespace.download) : null;
        const eta = downloading ? downloadEta : totalEta(state);
        const showEta = downloading ? Number.isFinite(downloadEta) : stageSupportsEta(current);
        text(namespace.panel, "[data-sp-eta]", showEta
            ? (downloading ? `~${formatDuration(eta)} transfer ETA` : `${formatDuration(eta, true)} ETA`)
            : "");
        const overallFill = namespace.panel.querySelector("[data-sp-overall-fill]");
        if (overallFill) {
            const stageIsIndeterminate = current && current.state === "current" &&
                (current.id === "decode" || current.activity === "unload");
            overallFill.classList.toggle("status-lite__overall-fill--indeterminate", Boolean(stageIsIndeterminate));
            overallFill.style.width = !stageIsIndeterminate && current && Number.isFinite(current.progress)
                ? `${current.progress}%`
                : "0%";
        }
        renderStages(namespace);
        renderDetail(namespace);
        renderDownloads(namespace);
    }

    function setActive(namespace, active) {
        namespace.container.classList.toggle("status-lite-container--active", active);
        namespace.host.classList.toggle("status-lite-host--active", active);
        namespace.source.classList.toggle("status-lite-source--active", active);
        namespace.panel.hidden = !active;
    }

    function tick(namespace) {
        if (!namespace.host.isConnected || !namespace.source.isConnected) return;
        observeRunOutcome(namespace, visibleFailureNotice(namespace));
        syncRunTelemetry(namespace);
        observeRunOutcome(namespace, visibleFailureNotice(namespace));
        readDownloadSnapshot(namespace);
        let snapshot = readPrepareStatus(namespace) || readReportedPreGenerationStatus(namespace) || readSnapshot(namespace);
        snapshot = reinterpretQwenSilentEncode(namespace, snapshot);
        if (!snapshot && namespace.download && namespace.download.visible &&
            (namespace.download.active || namespace.state.currentId === "prepare")) {
            snapshot = {
                id: "prepare",
                rawName: namespace.download.active ? "Downloading model files" : "Downloads complete",
                rawMessage: namespace.download.active ? "Downloading required model assets" : "Required model assets downloaded",
                metaText: "",
                stageElapsed: null,
                overallElapsed: namespace.state.overallElapsed,
                progress: null,
                steps: namespace.state.steps,
                aborting: false,
                textOnly: true
            };
        }
        if (snapshot) {
            observeRunOutcome(namespace, snapshot.rawName, snapshot.rawMessage);
            recoverMissedPerformanceStages(namespace, snapshot);
            applySnapshot(namespace, snapshot);
            setActive(namespace, true);
            render(namespace);
            return;
        }
        const now = Date.now();
        if (!namespace.state.inactiveSince) namespace.state.inactiveSince = now;
        if (!namespace.activeRun && now - namespace.state.inactiveSince >= RESET_AFTER_MS) {
            finishStage(namespace.state, namespace.state.currentId);
        }
        setActive(namespace, true);
        if (namespace.activeRun) render(namespace);
        else renderIdle(namespace);
    }

    function installStyle(root) {
        if (root.querySelector("#status-lite-styles")) return;
        const style = document.createElement("style");
        style.id = "status-lite-styles";
        style.textContent = STYLE_TEXT;
        root.appendChild(style);
    }

    function bind() {
        const root = appRoot();
        const container = root.querySelector("#status-lite-container");
        const host = root.querySelector("#status-lite-host");
        if (!container || !host) return false;
        const panel = host.querySelector("[data-status-lite]");
        const source = container.previousElementSibling;
        if (!panel || !source) return false;

        installStyle(root);
        const previous = window[NAMESPACE];
        if (previous && previous.timer) window.clearInterval(previous.timer);
        if (previous && previous.resumeHandler) {
            document.removeEventListener("visibilitychange", previous.resumeHandler);
            window.removeEventListener("focus", previous.resumeHandler);
        }
        const initialRunBridge = parseRunBridge(container);
        const runtimeId = runtimeIdFromTelemetry(initialRunBridge.telemetry);

        const namespace = {
            root,
            container,
            host,
            panel,
            source,
            tracker: null,
            state: freshState(),
            download: null,
            downloadRaw: "",
            runTelemetry: initialRunBridge.telemetry,
            runRaw: initialRunBridge.raw,
            qwenEncodeFallbackStartedAt: 0,
            activeRun: null,
            runtimeId,
            sessionId: (window.crypto && window.crypto.randomUUID)
                ? window.crypto.randomUUID()
                : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`,
            lastCompletedAt: null,
            resumeHandler: null,
            collapsed: loadCollapsedPreference(),
            timer: null
        };
        window[NAMESPACE] = namespace;

        panel.querySelector("[data-sp-stages]").addEventListener("click", event => {
            const button = event.target.closest("[data-stage-id]");
            if (!button || !panel.contains(button)) return;
            namespace.state.selectedId = button.dataset.stageId;
            namespace.state.selectionIsManual = true;
            render(namespace);
        });

        panel.querySelector("[data-sp-collapse]").addEventListener("click", () => {
            namespace.collapsed = !namespace.collapsed;
            saveCollapsedPreference(namespace.collapsed);
            applyCollapsed(namespace);
        });

        applyCollapsed(namespace);

        setActive(namespace, true);
        renderIdle(namespace);
        namespace.resumeHandler = () => {
            if (document.hidden) return;
            tick(namespace);
            // Gradio's browser-driven Timer may publish its first refreshed
            // backend bridge value shortly after the page becomes visible.
            window.setTimeout(() => tick(namespace), 750);
        };
        document.addEventListener("visibilitychange", namespace.resumeHandler);
        window.addEventListener("focus", namespace.resumeHandler);
        namespace.timer = window.setInterval(() => tick(namespace), TICK_MS);
        tick(namespace);
        console.info("[Status Lite] Progress timeline initialized");
        return true;
    }

    function boot() {
        if (bind()) return;
        window.setTimeout(boot, 500);
    }

    boot();
})();
"""

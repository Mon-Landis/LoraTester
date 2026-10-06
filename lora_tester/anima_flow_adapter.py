from __future__ import annotations

import inspect
import math
import sys
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .sampling_progress import aggregate_cell_progress


PROJECT_URL = "https://github.com/KeithZ117/Comfyui-anima-sampler"
SAMPLER_NODE = "AnimaFlowCorrectiveSampler"
SETTINGS_NODE = "AnimaFlowSettings"
FLOW_CONTROLS = (
    "seed", "steps", "cfg", "cfg_mode", "flow_solver", "flow_schedule",
    "flow_shift", "denoise", "add_noise",
)
MISSING_OPTION = "__anima_flow_dependency_missing__"


class AnimaFlowDependencyError(RuntimeError):
    pass


def _dependency_error(reason: str) -> AnimaFlowDependencyError:
    return AnimaFlowDependencyError(
        "AnimaFlow XY requires Comfyui-anima-sampler / "
        "AnimaFlow XY 测试器需要安装或更新 Comfyui-anima-sampler. "
        f"{reason} Install/update and restart ComfyUI: {PROJECT_URL}"
    )


def _registered_class(name: str) -> type | None:
    module = sys.modules.get("nodes")
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
    candidate = mappings.get(name) if isinstance(mappings, Mapping) else None
    return candidate if isinstance(candidate, type) else None


def missing_control_schema() -> dict[str, tuple]:
    return {
        "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
        "steps": ("INT", {"default": 30, "min": 1, "max": 1000}),
        "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.1}),
        "cfg_mode": ([MISSING_OPTION],),
        "flow_solver": ([MISSING_OPTION],),
        "flow_schedule": ([MISSING_OPTION],),
        "flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0}),
        "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0}),
        "add_noise": ("BOOLEAN", {"default": True}),
    }


def validate_control(name: str, value: Any, specification: tuple) -> Any:
    kind = specification[0]
    options = specification[1] if len(specification) > 1 else {}
    if isinstance(kind, (list, tuple)):
        if value not in kind:
            raise ValueError(f"AnimaFlow parameter {name!r}: unsupported value {value!r}; options: {kind}")
        return value
    if kind == "BOOLEAN":
        if not isinstance(value, bool):
            raise ValueError(f"AnimaFlow parameter {name!r} must be boolean")
        return value
    if kind in {"INT", "FLOAT"}:
        if isinstance(value, bool):
            raise ValueError(f"AnimaFlow parameter {name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"AnimaFlow parameter {name!r} must be finite")
        if kind == "INT":
            number = int(value)
            if number != value:
                raise ValueError(f"AnimaFlow parameter {name!r} must be an integer")
        if "min" in options and number < options["min"]:
            raise ValueError(f"AnimaFlow parameter {name!r} must be >= {options['min']}")
        if "max" in options and number > options["max"]:
            raise ValueError(f"AnimaFlow parameter {name!r} must be <= {options['max']}")
        return number
    raise _dependency_error(f"Unsupported control type {name}: {kind}")


class AnimaFlowAdapter:
    def __init__(self) -> None:
        self.sampler_class = _registered_class(SAMPLER_NODE)
        if self.sampler_class is None:
            raise _dependency_error(f"Node {SAMPLER_NODE} is missing or failed to load.")
        try:
            schema = self.sampler_class.INPUT_TYPES()
            required = schema["required"]
            optional = schema.get("optional", {})
            expected = {"model", "positive", "negative", "latent_image", *FLOW_CONTROLS}
            if set(required) != expected or optional.get("flow_settings", (None,))[0] != "ANIMA_FLOW_SETTINGS":
                raise ValueError("sampler input contract changed")
            if tuple(self.sampler_class.RETURN_TYPES)[:2] != ("LATENT", "IMAGE"):
                raise ValueError("sampler must return LATENT and IMAGE first")
            self.function_name = self.sampler_class.FUNCTION
            function = getattr(self.sampler_class, self.function_name)
            inspect.signature(function).bind(None, **{name: None for name in expected}, flow_settings=None, vae=None)
            self.controls = deepcopy({name: required[name] for name in FLOW_CONTROLS})
            self.settings_class = _registered_class(SETTINGS_NODE)
            self.advanced_controls = {}
            if self.settings_class is not None:
                self.advanced_controls = deepcopy(self.settings_class.INPUT_TYPES()["required"])
                if tuple(self.settings_class.RETURN_TYPES)[:1] != ("ANIMA_FLOW_SETTINGS",):
                    raise ValueError("settings must return ANIMA_FLOW_SETTINGS first")
                settings_function = getattr(self.settings_class, self.settings_class.FUNCTION)
                inspect.signature(settings_function).bind(None, **{name: None for name in self.advanced_controls})
        except Exception as error:
            raise _dependency_error(f"Incompatible node contract: {error}") from error

    @property
    def parameter_schema(self) -> dict[str, tuple]:
        return {**self.advanced_controls, **self.controls}

    def resolve_parameters(self, values: dict[str, Any], parameters: Mapping) -> dict[str, Any]:
        resolved = dict(values)
        for name, raw in parameters.items():
            if name == "flow_settings":
                resolved[name] = raw
            elif name in self.parameter_schema:
                resolved[name] = validate_control(name, raw, self.parameter_schema[name])
            else:
                raise ValueError(
                    f"Unsupported AnimaFlow axis parameter {name!r}. Native sampler_name/scheduler "
                    "cannot be mapped to flow_solver/flow_schedule; use an AnimaFlow Parameter Axis."
                )
        for name, specification in self.controls.items():
            resolved[name] = validate_control(name, resolved[name], specification)
        overrides = {name: resolved.pop(name) for name in self.advanced_controls if name in resolved}
        if overrides:
            supplied = resolved.get("flow_settings")
            if supplied is not None and not isinstance(supplied, dict):
                raise _dependency_error("Advanced-axis overrides require dictionary settings.")
            supplied = supplied or {}
            settings_values = {}
            for name, specification in self.advanced_controls.items():
                options = specification[1] if len(specification) > 1 else {}
                default = options.get("default")
                if default is None and isinstance(specification[0], (list, tuple)):
                    default = specification[0][0]
                settings_values[name] = overrides.get(name, supplied.get(name, default))
            settings_node = self.settings_class()
            built = getattr(settings_node, settings_node.FUNCTION)(**settings_values)
            if not isinstance(built, (tuple, list)) or not built or not isinstance(built[0], dict):
                raise _dependency_error("Advanced-axis overrides require dictionary settings output.")
            resolved["flow_settings"] = {
                **built[0], **supplied,
                **{name: built[0][name] for name in overrides},
            }
        return resolved

    def sample_cell(self, *, model: Any, positive: Any, negative: Any, latent: dict, values: dict, vae: Any = None, progress: Any = None, completed_tasks: int = 0, total_tasks: int = 1) -> tuple[dict, Any]:
        node = self.sampler_class()
        with aggregate_cell_progress(progress, completed_tasks, total_tasks):
            result = getattr(node, self.function_name)(
                model=model, positive=positive, negative=negative,
                latent_image=latent.copy(), flow_settings=deepcopy(values.get("flow_settings")), vae=vae,
                **{name: values[name] for name in FLOW_CONTROLS},
            )
        if isinstance(result, dict):
            result = result.get("result")
        if not isinstance(result, (tuple, list)) or len(result) < 2 or not isinstance(result[0], dict) or "samples" not in result[0]:
            raise _dependency_error("Sampler returned an incompatible LATENT result.")
        return result[0], result[1]


def anima_flow_status() -> dict[str, Any]:
    try:
        adapter = AnimaFlowAdapter()
        schema = {}
        for name, specification in adapter.parameter_schema.items():
            kind = specification[0]
            options = specification[1] if len(specification) > 1 else {}
            schema[name] = {"kind": "COMBO" if isinstance(kind, (list, tuple)) else kind}
            if isinstance(kind, (list, tuple)):
                schema[name]["values"] = list(kind)
            schema[name].update({key: options[key] for key in ("min", "max", "default") if key in options})
        return {"available": True, "project_url": PROJECT_URL, "node": SAMPLER_NODE, "error": "", "parameters": list(adapter.parameter_schema), "schema": schema}
    except AnimaFlowDependencyError as error:
        return {"available": False, "project_url": PROJECT_URL, "node": SAMPLER_NODE, "error": str(error), "parameters": []}


def register_anima_flow_routes() -> None:
    server_module = sys.modules.get("server")
    instance = getattr(getattr(server_module, "PromptServer", None), "instance", None)
    if instance is None or getattr(instance, "_lora_tester_flow_routes", False):
        return
    from aiohttp import web

    @instance.routes.get("/lora_tester/anima_flow/status")
    async def status(request):
        return web.json_response(anima_flow_status())

    instance._lora_tester_flow_routes = True

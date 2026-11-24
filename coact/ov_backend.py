"""Inference backend for CoAct's candidate-scoring kernel.

Grounding reduces to scoring every candidate object against a query feature
vector. That is a matmul. We compile that matmul as an OpenVINO model and run
it through the OpenVINO runtime when it is installed, so OpenVINO is on the
real execution path rather than being a logo on a slide.

Honesty rules enforced here:
  * `available` is False unless `import openvino` actually succeeded.
  * `device` is whatever OpenVINO actually resolved AUTO to on this machine.
  * `latency_ms` is measured with perf_counter around a real inference call.
  * If OpenVINO is missing we say so and run the identical maths in NumPy.
Nothing in this file invents a number.
"""

from __future__ import annotations

import os
import time

import numpy as np

_PREFERRED_DEVICE = os.getenv("COACT_OV_DEVICE", "AUTO")


class ScoringBackend:
    def __init__(self, feature_dim: int = 8, device: str = _PREFERRED_DEVICE):
        self.feature_dim = feature_dim
        self.requested_device = device
        self.available = False
        self.backend_name = "numpy"
        self.device = "CPU (numpy)"
        self.version = None
        self.available_devices: list[str] = []
        self.last_latency_ms: float | None = None
        self.init_error: str | None = None
        self._compiled = None
        self._try_init_openvino()

    def _try_init_openvino(self) -> None:
        try:
            import openvino as ov  # type: ignore
            from openvino import opset13 as ops  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on host
            self.init_error = f"openvino not importable: {exc.__class__.__name__}"
            return

        try:
            core = ov.Core()
            self.available_devices = list(core.available_devices)
            self.version = ov.__version__

            # score = objects (N x D) @ query (D x 1)
            objs = ops.parameter([-1, self.feature_dim], ov.Type.f32, name="objects")
            query = ops.parameter([self.feature_dim, 1], ov.Type.f32, name="query")
            matmul = ops.matmul(objs, query, False, False)
            model = ov.Model([matmul], [objs, query], "coact_scoring")

            device = self.requested_device
            if device != "AUTO" and device not in self.available_devices:
                device = "AUTO"
            self._compiled = core.compile_model(model, device)
            try:
                self.device = self._compiled.get_property("EXECUTION_DEVICES")
                if isinstance(self.device, (list, tuple)):
                    self.device = ",".join(self.device)
            except Exception:
                self.device = device
            self.available = True
            self.backend_name = "openvino"
        except Exception as exc:  # pragma: no cover - depends on host
            self.init_error = f"openvino init failed: {exc}"
            self._compiled = None

    def score(self, object_features: np.ndarray, query: np.ndarray) -> np.ndarray:
        objs = np.ascontiguousarray(object_features, dtype=np.float32)
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(self.feature_dim, 1)

        start = time.perf_counter()
        if self._compiled is not None:
            result = self._compiled({"objects": objs, "query": q})
            out = np.array(list(result.values())[0]).reshape(-1)
        else:
            out = (objs @ q).reshape(-1)
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        return out

    def telemetry(self) -> dict:
        return {
            "backend": "OpenVINO" if self.available else "NumPy (CPU fallback)",
            "available": self.available,
            "device": self.device,
            "requested_device": self.requested_device,
            "available_devices": self.available_devices,
            "openvino_version": self.version,
            "last_latency_ms": (
                round(self.last_latency_ms, 3) if self.last_latency_ms is not None else None
            ),
            "note": self.init_error
            or "Scoring kernel executing through the OpenVINO runtime.",
        }


_BACKEND: ScoringBackend | None = None


def get_backend() -> ScoringBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = ScoringBackend()
    return _BACKEND

"""CoAct backend — FastAPI."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from . import speech
from .ov_backend import get_backend
from .pipeline import Session
from .state_machine import SafetyViolation

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="CoAct", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# EGL contexts belong to the thread that created them, and FastAPI runs sync
# endpoints on a threadpool — so rendering from a request handler fails with
# EGL_BAD_ACCESS the moment a second worker picks up a request. Every call that
# touches physics or the GPU is funnelled onto one dedicated thread, including
# the Session's own construction.
_SIM = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coact-sim")


def in_sim(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return _SIM.submit(fn, *args, **kwargs).result()


SESSION: Session = in_sim(Session)


class Text(BaseModel):
    text: str
    source_label: str | None = None


@app.get("/api/health")
def health() -> dict:
    ov = get_backend().telemetry()
    return {
        "ok": True,
        "version": "2.0.0",
        "services": {
            "backend": {"connected": True, "detail": "FastAPI"},
            "openvino": {
                "connected": ov["available"],
                "detail": f"{ov['backend']} · {ov['device']}",
            },
            "speechmatics": {
                "connected": speech.configured(),
                "detail": speech.status()["mode"],
            },
            "mujoco": {
                "connected": _mujoco_available(),
                "detail": (getattr(SESSION.rt, "name", "unknown") if _mujoco_available()
                           else "not available on this host"),
            },
        },
        "telemetry": ov,
    }


def _mujoco_available() -> bool:
    from .sim import mujoco_available

    return mujoco_available() and SESSION.physics


@app.get("/api/state")
def state() -> dict:
    return in_sim(SESSION.snapshot)


@app.post("/api/instruct")
def instruct(body: Text) -> dict:
    if not body.text.strip():
        raise HTTPException(400, "empty instruction")
    return in_sim(SESSION.instruct, body.text.strip())


@app.post("/api/correct")
def correct(body: Text) -> dict:
    label = body.source_label or ("LIVE SPEECHMATICS" if speech.configured() else "DEMO REPLAY")
    return in_sim(SESSION.correct, body.text.strip(), source_label=label)


@app.post("/api/execute")
def execute() -> dict:
    try:
        return in_sim(SESSION.execute)
    except SafetyViolation as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/reset")
def reset() -> dict:
    in_sim(SESSION.reset)
    return in_sim(SESSION.snapshot)


def _png(rgb) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(500, "encode failed")
    return buf.tobytes()


@app.get("/api/frame/{camera}")
def frame(camera: str, annotated: bool = True) -> Response:
    """A live camera frame. 'A' and 'B' are the agents' own views; 'obs' is the
    third-person camera, which no agent ever uses for grounding."""
    if not SESSION.physics:
        raise HTTPException(503, "no physics runtime on this host")
    if camera not in ("A", "B", "obs"):
        raise HTTPException(404, "unknown camera")
    if camera == "obs":
        return Response(_png(in_sim(SESSION.rt.render, "cam_obs")), media_type="image/png")

    from .perception import annotate

    def grab():
        agent = SESSION.agents[camera]
        if not agent.percepts or agent.last_image is None:
            agent.observe()
        return (annotate(agent.last_image, [p.det for p in agent.percepts])
                if annotated else agent.last_image)

    return Response(_png(in_sim(grab)), media_type="image/png")


@app.get("/api/benchmark")
def benchmark() -> dict:
    """The stored benchmark report, if one has been run."""
    path = Path(__file__).resolve().parent.parent / "results" / "benchmark.json"
    if not path.exists():
        raise HTTPException(404, "no benchmark report; run python -m coact.benchmark")
    import json

    report = json.loads(path.read_text())
    report.pop("trials", None)       # summary only; the full log stays on disk
    return report


@app.get("/api/speechmatics/status")
def sm_status() -> dict:
    return speech.status()


@app.post("/api/speechmatics/token")
async def sm_token() -> dict:
    return await speech.mint_temp_key()


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))


def main() -> None:
    import uvicorn

    uvicorn.run(
        "coact.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()

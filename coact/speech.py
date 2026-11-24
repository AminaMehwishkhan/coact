"""Speechmatics integration.

Two modes, and the UI is told which one is active — we never dress a canned
string up as live API output.

LIVE SPEECHMATICS
    The backend mints a short-lived Speechmatics real-time JWT with the
    server-side SPEECHMATICS_API_KEY and hands it to the browser, which opens
    a WebSocket straight to the Speechmatics RT endpoint. The long-lived key
    never reaches the client.

DEMO REPLAY
    No key configured (or no microphone in the deployment target). A canned
    correction utterance is pushed through the SAME Session.correct() pipeline
    and the UI badge reads DEMO REPLAY.
"""

from __future__ import annotations

import os

import httpx

SPEECHMATICS_MP_URL = "https://mp.speechmatics.com/v1/api_keys"
DEFAULT_RT_URL = os.getenv("SPEECHMATICS_RT_URL", "wss://eu2.rt.speechmatics.com/v2")


def api_key() -> str | None:
    key = os.getenv("SPEECHMATICS_API_KEY", "").strip()
    return key or None


def configured() -> bool:
    return api_key() is not None


def status() -> dict:
    return {
        "mode": "LIVE SPEECHMATICS" if configured() else "DEMO REPLAY",
        "configured": configured(),
        "rt_url": DEFAULT_RT_URL,
        "note": (
            "Server-side key present; browser will receive a short-lived JWT."
            if configured()
            else "SPEECHMATICS_API_KEY not set — corrections replay through the same "
            "correction pipeline and are labelled DEMO REPLAY."
        ),
    }


async def mint_temp_key(ttl: int = 120) -> dict:
    """Exchange the long-lived key for a short-lived RT JWT.

    Returns {"ok": False, "reason": ...} rather than raising, so the UI can
    fall back to DEMO REPLAY cleanly instead of crashing mid-demo.
    """
    key = api_key()
    if not key:
        return {"ok": False, "reason": "SPEECHMATICS_API_KEY not configured"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                SPEECHMATICS_MP_URL,
                params={"type": "rt"},
                headers={"Authorization": f"Bearer {key}"},
                json={"ttl": ttl},
            )
        if resp.status_code >= 400:
            return {"ok": False, "reason": f"Speechmatics returned {resp.status_code}"}
        data = resp.json()
        return {
            "ok": True,
            "jwt": data.get("key_value"),
            "ttl": ttl,
            "rt_url": DEFAULT_RT_URL,
        }
    except Exception as exc:
        return {"ok": False, "reason": f"{exc.__class__.__name__}: {exc}"}


# Canned corrections used only in DEMO REPLAY mode. Always surfaced to the UI
# with the DEMO REPLAY label attached.
DEMO_CORRECTIONS = {
    "size": "No, the large plate.",
    "colour": "No, the red cup.",
    "spatial": "No, my left.",
}

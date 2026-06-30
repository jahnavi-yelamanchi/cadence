"""Cadence FastAPI server.

Endpoints:
  GET  /health          — liveness check + model info
  GET  /metrics/{sid}   — per-session FIR / dead-air summary
  WS   /stream/{sid}    — live audio stream (20ms PCM float32 LE chunks)

WebSocket message protocol:
  Client → Server: raw bytes  (320 float32 samples = 1280 bytes per chunk)
  Server → Client: JSON event {
    "cadence":  {"label": "turn_end"|"mid_thought", "confidence": 0.91, "latency_ms": 18},
    "vad":      {"label": "turn_end"|"mid_thought"},
    "ts":       1234567890.123   # server timestamp (ms since epoch)
  }
"""

import json
import struct
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from server.baseline import SileroBaseline
from server.endpointer import Endpointer

ONNX_PATH = Path("model/onnx/cadence.onnx")
CHUNK_SAMPLES = 320  # 20ms @ 16kHz


app = FastAPI(title="Cadence", version="0.1.0", description="Smart turn-taking endpointer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Per-session state
_sessions: dict[str, dict] = defaultdict(
    lambda: {
        "endpointer": None,
        "baseline": None,
        "false_interruptions": 0,
        "vad_false_interruptions": 0,
        "total_decisions": 0,
        "dead_air_samples": [],
        "last_turn_end_ts": None,
    }
)


def _get_or_create_session(sid: str) -> dict:
    s = _sessions[sid]
    if s["endpointer"] is None:
        s["endpointer"] = Endpointer(ONNX_PATH)
        s["baseline"] = SileroBaseline()
    return s


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": str(ONNX_PATH),
        "model_loaded": ONNX_PATH.exists(),
    }


@app.get("/metrics/{sid}")
def session_metrics(sid: str):
    s = _sessions.get(sid)
    if not s or s["total_decisions"] == 0:
        return {"error": "session not found or no decisions yet"}

    mean_da = float(np.mean(s["dead_air_samples"])) if s["dead_air_samples"] else 0.0
    fir = s["false_interruptions"] / max(s["total_decisions"], 1)
    vad_fir = s["vad_false_interruptions"] / max(s["total_decisions"], 1)

    return {
        "session_id": sid,
        "total_decisions": s["total_decisions"],
        "cadence_FIR": round(fir, 4),
        "vad_FIR": round(vad_fir, 4),
        "mean_dead_air_ms": round(mean_da, 1),
    }


@app.websocket("/stream/{sid}")
async def stream(websocket: WebSocket, sid: str):
    await websocket.accept()
    s = _get_or_create_session(sid)
    endpointer: Endpointer = s["endpointer"]
    baseline: SileroBaseline = s["baseline"]

    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) != CHUNK_SAMPLES * 4:
                continue  # skip malformed chunks

            chunk = np.frombuffer(data, dtype=np.float32)
            ts = time.time() * 1000  # ms epoch

            cadence_result = endpointer.push(chunk)
            vad_result = baseline.push(chunk)

            if cadence_result is not None or vad_result is not None:
                s["total_decisions"] += 1

                # Track dead air: time since last confirmed turn_end
                if cadence_result and cadence_result["label"] == "turn_end":
                    if s["last_turn_end_ts"] is not None:
                        da = ts - s["last_turn_end_ts"]
                        s["dead_air_samples"].append(da)
                    s["last_turn_end_ts"] = ts

                # Count VAD false interruptions (VAD says turn_end, Cadence says mid_thought)
                if (
                    vad_result and vad_result["label"] == "turn_end"
                    and cadence_result and cadence_result["label"] == "mid_thought"
                ):
                    s["vad_false_interruptions"] += 1

                event = {
                    "cadence": cadence_result,
                    "vad": vad_result,
                    "ts": ts,
                }
                await websocket.send_text(json.dumps(event))

    except WebSocketDisconnect:
        pass
    finally:
        endpointer.reset()
        baseline.reset()


@app.on_event("startup")
def startup():
    if not ONNX_PATH.exists():
        import warnings
        warnings.warn(
            f"ONNX model not found at {ONNX_PATH}. "
            "The /stream endpoint will fail until you run: make export"
        )

"""Homework 4 - from prototype to product: a server and a web page around the hardware-in-the-loop system of Homework 3.

    pip install fastapi uvicorn
    export OPENROUTER_API_KEY=...            # typed in the terminal, never in a file
    python3 web_app.py --model "<openrouter model id>"            # then open http://127.0.0.1:8000
    python3 web_app.py --model "<id>" --fake                      # the synthetic camera, loudspeaker and microphone (buttons on the page break them)

The page has two modes, both on the same graph (orchestrator, scientist, general, executor, hardware, human) and the same MCP instrument server:
  * WATCH  the watchdog (watchdog.py) measures the camera and the microphone every few seconds WITHOUT a model. When the numbers change
           twice in a row it hands an ALERT to the graph: the agent chooses its own checks (each call shows its own reason, `why`), asks
           the person to fix what a person can fix (the pause of Homework 3, now buttons on the page), checks that all is normal again.
  * ASK    type a request or choose one of the use cases of run_hw3.py: the same run as in the terminal, shown live.

Every run writes runs/<time>-<ask|alert>/ (transcript.md, report.md, evidence.md, events.json, images/) like run_hw3.py.
Whole camera pictures are never written, only the crops of the measured region; sound is never written.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cam_platform as plat
import hw3_agents as A
import llm_utils as U
import watchdog as W
from mcp_bridge import McpBridge, default_env

try:                                         # at module level: FastAPI reads the type of every parameter from this module
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
except ImportError:                          # the tests of the other parts do not need it; main() says what to install
    Body = FastAPI = HTTPException = Request = FileResponse = HTMLResponse = JSONResponse = Response = StreamingResponse = None

HERE = Path(__file__).resolve().parent
SHOWN_KEYS = ("tone_db", "noise_db", "snr_db", "heard", "freq_hz", "level_db", "mean_brightness", "sharpness_norm", "noise_rms_db",
              "correct", "n_frames", "saturated_fraction", "coupling_db", "silent_input", "interference", "accepted")


WATCH_LIMITS = dict(interval=(2.0, 600.0), beep=(6.0, 3600.0))      # seconds; the fields of the page (a beep check of 0 means off); tests lower them


def parse_watch(body: dict) -> tuple:
    """(interval, tone_every) from the page, or ValueError with the sentence for the person: a letter, a negative interval or a beep every
    second must not reach the watchdog (it would measure without pauses or beep at every sample)."""
    try:
        interval, tone_every = float(body.get("interval", 4.0)), float(body.get("tone_every", 0.0))
    except (TypeError, ValueError):
        raise ValueError("'check every' and 'beep check every' must be numbers") from None
    lo, hi = WATCH_LIMITS["interval"]
    if not lo <= interval <= hi:
        raise ValueError(f"'check every' must be between {lo:g} and {hi:g} seconds")
    blo, bhi = WATCH_LIMITS["beep"]
    if tone_every != 0 and not blo <= tone_every <= bhi:
        raise ValueError(f"'beep check every' must be 0 (off) or between {blo:g} and {bhi:g} seconds")
    return interval, tone_every


class Aborted(Exception):
    pass


class Busy(Exception):
    pass


class Hub:
    """The log of everything that happens: the page reads it as a stream. The id of an event is its number (1, 2, 3 ...)."""

    def __init__(self):
        self.events: list[dict] = []
        self.lock = threading.Lock()

    def publish(self, ev: dict) -> dict:
        with self.lock:
            ev = dict(ev)
            ev["id"] = len(self.events) + 1
            ev.setdefault("ts", round(time.time(), 2))
            self.events.append(ev)
            return ev

    def since(self, last_id: int) -> list[dict]:
        with self.lock:
            return self.events[max(0, int(last_id)):]


class SharedBridge:
    """One instrument, one command at a time: the watchdog and the agent never talk to the device together."""

    def __init__(self, bridge):
        self.b = bridge
        self.lock = threading.RLock()

    def call(self, name, args=None, timeout: float = 180.0):
        with self.lock:
            return self.b.call(name, args, timeout)

    def tool_specs(self):
        return self.b.tool_specs()

    def tool_names(self):
        return self.b.tool_names()

    def close(self):
        self.b.close()


def scalars(r: dict) -> dict:
    out = {k: r[k] for k in SHOWN_KEYS if k in r and isinstance(r[k], (int, float, bool))}
    if isinstance(r.get("resolution"), list):
        out["resolution"] = "x".join(str(v) for v in r["resolution"])
    return out


class Session:
    SETTLE_AFTER_HUMAN = 10.0     # s: after the person changes something at the device the camera needs time to readjust exposure and focus

    def __init__(self, models: dict, bridge, work: Path, runs: Path, max_decisions: int = A.MAX_DECISIONS,
                 info: Optional[dict] = None, model_name: str = "", fake: bool = False):
        self.models, self.bridge = models, bridge
        self.work, self.runs = Path(work), Path(runs)
        self.max_decisions = max_decisions
        self.info = info or {}
        self.model_name, self.fake = model_name, fake
        self.hub = Hub()
        self.mode = "idle"                       # idle | watching | investigating | running | paused
        self.mode_before_pause = "running"
        self.pending: Optional[dict] = None
        self.reply_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self.stop_ev = threading.Event()
        self.guard = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.run_id = ""
        self.watch_opts: dict = {}
        self._install_hooks()

    # ---- what the page sees
    def emit(self, ev: dict) -> dict:
        ev = dict(ev)
        ev.setdefault("run", self.run_id)
        return self.hub.publish(ev)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.emit(dict(type="mode", mode=mode))

    def state(self) -> dict:
        from run_hw3 import USE_CASES
        return dict(mode=self.mode, pending=self.pending, info=self.info, model=self.model_name, fake=self.fake,
                    watch=self.watch_opts, last_id=len(self.hub.events),
                    presets={k: v[0] for k, v in USE_CASES.items()})

    def _install_hooks(self) -> None:
        def before(ev):
            self.emit(dict(type="tool_call", n=ev["n"], tool=ev["tool"], args=ev["args"], why=ev.get("why", "")))

        def after(ev):
            r = ev["result"]
            self.emit(dict(type="tool_result", n=ev["n"], tool=ev["tool"], ok=ev["ok"], summary=A.short_result(ev["tool"], r),
                           why=ev.get("why", ""), crops=list(r.get("saved_crops", []) or []), values=scalars(r)))

        A.HOOKS["before_tool"] = before
        A.HOOKS["after_tool"] = after

    # ---- starting and stopping
    def _begin(self, mode: str, target, *args) -> None:
        with self.guard:
            if self.mode != "idle" or (self.thread and self.thread.is_alive()):
                raise Busy(f"busy: {self.mode}")
            self.stop_ev = threading.Event()
            while not self.reply_q.empty():
                self.reply_q.get_nowait()
            self.set_mode(mode)
            self.thread = threading.Thread(target=target, args=args, name=f"web-{mode}", daemon=True)
            self.thread.start()

    def start_watch(self, interval: float = 4.0, tone_every: float = 0.0) -> None:
        self.watch_opts = dict(interval=float(interval), tone_every=float(tone_every))
        self._begin("watching", self._watch_main, float(interval), float(tone_every))

    def run_request(self, request: str) -> None:
        self._begin("running", self._ask_main, request)

    def stop(self) -> None:
        self.stop_ev.set()
        self.reply_q.put(None)
        self.emit(dict(type="note", text="stopping (a run that is in progress ends after the step it is doing now, a few seconds)"))

    def reply(self, text: str) -> None:
        if self.pending is None:
            raise Busy("nothing is waiting for a reply")
        self.reply_q.put(str(text))

    # ---- the two modes
    def _watch_main(self, interval: float, tone_every: float) -> None:
        wd = W.Watchdog(self.bridge.call, self.bridge.tool_names(), self.emit, self._investigate, interval, tone_every)
        try:
            wd.run(self.stop_ev)
        except Exception as e:  # noqa: BLE001
            self.emit(dict(type="error", text=f"the watchdog stopped: {type(e).__name__}: {str(e)[:300]}"))
        finally:
            self.pending = None
            self.set_mode("idle")

    def _investigate(self, request: str, facts: list) -> None:
        self.set_mode("investigating")
        try:
            self._run_graph(request, "alert")
        finally:
            self.run_id = ""
            if not self.stop_ev.is_set():
                self.set_mode("watching")

    def _ask_main(self, request: str) -> None:
        try:
            self._run_graph(request, "ask")
        finally:
            self.pending = None
            self.run_id = ""
            self.set_mode("idle")

    # ---- one run of the graph of Homework 3
    def _ask(self, payload: dict) -> str:
        self.mode_before_pause = self.mode if self.mode != "paused" else self.mode_before_pause
        self.pending = payload
        self.set_mode("paused")
        self.emit(dict(type="pause", payload=payload))
        while True:
            try:
                r = self.reply_q.get(timeout=0.5)
            except queue.Empty:
                if self.stop_ev.is_set():
                    raise Aborted()
                continue
            break
        if r is None:
            self.pending = None
            raise Aborted()
        self.pending = None
        self.emit(dict(type="reply", text=r))
        self.set_mode(self.mode_before_pause)
        if payload.get("kind") == "scene_action" and self.SETTLE_AFTER_HUMAN > 0:
            # a real run: right after the lens was uncovered the picture was still 3x less sharp than normal, and the agent blamed a smudge
            self.emit(dict(type="note", text=f"waiting {self.SETTLE_AFTER_HUMAN:g} s: after a change at the device the camera needs a few "
                                             "seconds to readjust exposure and focus, then the agent measures again"))
            self.stop_ev.wait(self.SETTLE_AFTER_HUMAN)
        return r

    def _on_update(self, node: str, upd: dict) -> None:
        if self.stop_ev.is_set():
            raise Aborted()                      # Stop ends the run after the step that was in progress (a step is one model request and its instrument calls)
        for t in upd.get("dialogue", []):
            text = t["text"].strip()
            if t["role"] == "hardware":
                text = (upd.get("hw_report") or "").strip() or "(no closing words)"
            self.emit(dict(type="turn", role=t["role"], text=text[:1800], next=upd.get("next", "")))

    def _run_graph(self, request: str, source: str) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = (self.runs / f"{stamp}-{source}").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_dir.name
        U.STATS["model_calls"] = 0               # the counter shown in the final line and written to the run folder counts THIS run only
        self.emit(dict(type="run_start", source=source, request=request))
        log_path = self.work / "instrument_log.jsonl"
        log_start = log_path.stat().st_size if log_path.exists() else 0
        try:
            graph = A.build_graph(self.models, self.bridge, self.work, max_decisions=self.max_decisions)
            final = A.drive(graph, A.build_task(request), self._ask, thread_id=f"web-{stamp}-{source}", on_update=self._on_update)
        except Aborted:
            self.emit(dict(type="note", text="the run was stopped"))
            self._drop_if_empty(run_dir)
            return
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:400]}"
            if "402" in msg or "credits" in msg.lower():
                msg += (" || OpenRouter says the account has too few credits for this model: top up the credits on openrouter.ai "
                        "(a few dollars are enough) or choose a cheaper model with --model. The instrument was NOT asked anything.")
            self.emit(dict(type="error", text=msg))
            self._drop_if_empty(run_dir)
            return
        finally:
            self.pending = None
        events = final.get("events", [])
        report = A.final_report_md(final)
        (run_dir / "transcript.md").write_text(A.transcript_md(final), encoding="utf-8")
        (run_dir / "report.md").write_text(report, encoding="utf-8")
        (run_dir / "evidence.md").write_text(A.evidence_md(final["dialogue"]), encoding="utf-8")
        (run_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
        (run_dir / "run.json").write_text(json.dumps(dict(
            request=request, source=source, model=self.model_name, fake=self.fake, decisions=final.get("decisions"),
            camera_calls=final.get("n_calls"), refused=final.get("refused"), stop_reason=final.get("stop_reason"),
            model_requests=U.STATS["model_calls"]), indent=2), encoding="utf-8")
        if log_path.exists():
            with log_path.open("rb") as fh:
                fh.seek(log_start)
                (run_dir / "instrument_log.jsonl").write_bytes(fh.read())
        for ev in events:
            for f in list(ev["result"].get("saved_crops", []) or []):
                src = self.work / f
                if src.exists():
                    (run_dir / "images").mkdir(exist_ok=True)
                    shutil.copy2(src, run_dir / "images" / f"{ev['n']:02d}_{src.name}")
        self.emit(dict(type="final", report=report, decisions=final.get("decisions"), calls=final.get("n_calls"),
                       stop_reason=final.get("stop_reason"), model_requests=U.STATS["model_calls"], folder=run_dir.name))

    @staticmethod
    def _drop_if_empty(run_dir: Path) -> None:
        try:
            run_dir.rmdir()                      # only succeeds when nothing was written into it
        except OSError:
            pass

    # ---- the synthetic instrument (only with --fake): the page breaks it, the watchdog has to notice
    def fake_set(self, **kw) -> dict:
        if not self.fake:
            raise Busy("only available with --fake")
        files = {"fake_state.json": ("covered", "light", "shift"),
                 "fake_audio_state.json": ("muted", "noise_db", "volume_db", "coupling_db")}
        done = {}
        for name, keys in files.items():
            upd = {k: kw[k] for k in keys if k in kw}
            if not upd:
                continue
            p = self.work / name
            st = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            st.update(upd)
            p.write_text(json.dumps(st), encoding="utf-8")
            done.update(upd)
        self.emit(dict(type="note", text="simulated change: " + ", ".join(f"{k}={v}" for k, v in done.items())))
        return done


# ----------------------------------------------------------------------------- the web application
def create_app(session: Session):
    if FastAPI is None:
        raise SystemExit("pip install fastapi uvicorn")
    app = FastAPI(title="Hardware-in-the-loop lab")
    app.state.session = session

    def busy(e: Exception):
        return JSONResponse(status_code=409, content=dict(ok=False, error=str(e)))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (HERE / "web_ui.html").read_text(encoding="utf-8")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.get("/api/state")
    def state():
        return session.state()

    @app.get("/api/log")
    def log(since: int = 0):
        return dict(events=session.hub.since(since))

    @app.get("/api/stream")
    async def stream(request: Request, since: int = 0):
        async def gen():
            i = since
            while not await request.is_disconnected():
                evs = session.hub.since(i)
                for e in evs:
                    i = e["id"]
                    yield f"id: {i}\ndata: {json.dumps(e)}\n\n"
                if not evs:
                    yield ": ping\n\n"
                    await asyncio.sleep(0.3)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/watch/start")
    def watch_start(body: dict = Body(default={})):
        try:
            interval, tone_every = parse_watch(body)
        except ValueError as e:
            return JSONResponse(status_code=400, content=dict(ok=False, error=str(e)))
        try:
            session.start_watch(interval, tone_every)
        except Busy as e:
            return busy(e)
        return dict(ok=True)

    @app.post("/api/ask")
    def ask(body: dict = Body(...)):
        from run_hw3 import USE_CASES
        request = (body.get("request") or "").strip()
        if not request and body.get("use_case") in USE_CASES:
            request = USE_CASES[body["use_case"]][1]
        if not request:
            raise HTTPException(status_code=400, detail="give a request or a use_case")
        try:
            session.run_request(request)
        except Busy as e:
            return busy(e)
        return dict(ok=True)

    @app.post("/api/reply")
    def reply(body: dict = Body(...)):
        try:
            session.reply(str(body.get("text", "")))
        except Busy as e:
            return busy(e)
        return dict(ok=True)

    @app.post("/api/stop")
    def stop():
        session.stop()
        return dict(ok=True)

    @app.post("/api/fake")
    def fake(body: dict = Body(...)):
        try:
            return dict(ok=True, set=session.fake_set(**body))
        except Busy as e:
            return busy(e)

    @app.get("/api/crop")
    def crop(path: str):
        p = (session.work / path).resolve()
        if session.work.resolve() not in p.parents or p.suffix.lower() != ".png" or not p.exists():
            raise HTTPException(status_code=404, detail="no such picture")
        return FileResponse(p)

    return app


def build_session(args) -> Session:
    plat.utf8_console()
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("Set OPENROUTER_API_KEY (or GOOGLE_API_KEY / ANTHROPIC_API_KEY) first.")
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    roi, controls = Path(args.roi), Path(args.controls)
    env = default_env(str(work), roi_file=str(roi) if roi.exists() else None, camera_index=args.camera_index, fake=args.fake,
                      fake_state=str(work / "fake_state.json") if args.fake else None,
                      controls_file=str(controls) if controls.exists() else None,
                      fake_audio=args.fake, fake_audio_state=str(work / "fake_audio_state.json") if args.fake else None)
    print("starting the instrument server (MCP, stdio) ...", flush=True)
    bridge = McpBridge(env=env)
    bridge.start()
    names = bridge.tool_names()
    print("tools offered by the server:", ", ".join(names))
    info: dict = {}
    if "camera_info" in names:
        c = json.loads(bridge.call("camera_info"))
        info["camera"] = c.get("instrument") if c.get("ok", True) and "error" not in c else "NOT READY: " + str(c.get("error"))
        info["frame"] = c.get("frame_size")
    if "audio_info" in names:
        a = json.loads(bridge.call("audio_info"))
        if a.get("ok"):
            info["loudspeaker"], info["microphone"] = a["loudspeaker"], a["microphone"]
        else:
            info["sound"] = "NOT READY: " + str(a.get("error"))
    models = U.default_models(args.model, args.provider, args.hardware_model)
    return Session(models, SharedBridge(bridge), work, Path(args.runs).resolve(), args.max_decisions, info,
                   model_name=args.model or os.environ.get("AGENT_MODEL", ""), fake=args.fake)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="model id (env AGENT_MODEL)")
    ap.add_argument("--hardware-model", default=None, help="a different model for the hardware agent")
    ap.add_argument("--provider", choices=["openrouter", "google", "anthropic"], default=None)
    ap.add_argument("--work", default="hw3_work")
    ap.add_argument("--roi", default="roi.json")
    ap.add_argument("--controls", default="controls.json")
    ap.add_argument("--camera-index", type=int, default=None)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--max-decisions", type=int, default=A.MAX_DECISIONS)
    ap.add_argument("--fake", action="store_true", help="synthetic camera, loudspeaker and microphone (NOT a measurement)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    try:
        import uvicorn
        import fastapi  # noqa: F401
    except ImportError:
        sys.exit("pip install fastapi uvicorn")
    session = build_session(args)
    app = create_app(session)
    print(f"\nopen  http://{args.host}:{args.port}   (Ctrl+C stops the server)\n", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        session.stop()
        session.bridge.close()


if __name__ == "__main__":
    main()

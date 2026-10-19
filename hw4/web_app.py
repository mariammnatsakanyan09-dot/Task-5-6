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

  * GUARD  --guard / --guard-hours 22:00-07:00: the room is watched too (a person, the light, a flame colour, the noise), with the instrument tested first;
           --sensor-port: a small separate door for outside sensors (a smoke detector): its alarm is announced at once (signed messages: sensor_send.py).
  * KEEP   the guard tells when IT is in trouble (WATCH TROUBLE pushes; --heartbeat-url: a ping that stops when the loop is not healthy), pauses for
           somebody who is allowed in (the page button), stays silent at night, rotates its logs and writes watch_summary.csv (night_report.py);
           --autostart / start_guard.sh start it with the computer. See README.md.

Every run writes runs/<time>-<ask|alert>/ (transcript.md, report.md, evidence.md, events.json, images/) like run_hw3.py.
Whole camera pictures are never written, only the crops of the measured region; sound is never written.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import queue
import re
import shutil
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cam_platform as plat
import hw3_agents as A
import llm_utils as U
import watchdog as W
from mcp_bridge import McpBridge, default_env

try:                                         # at module level: FastAPI reads the type of every parameter from this module
    from fastapi import Body, FastAPI, Header, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
except ImportError:                          # the tests of the other parts do not need it; main() says what to install
    Body = FastAPI = Header = HTTPException = Request = FileResponse = HTMLResponse = JSONResponse = Response = StreamingResponse = None

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
                 info: Optional[dict] = None, model_name: str = "", fake: bool = False,
                 guard_room: bool = False, ref_path=None, fresh_reference: bool = False, notify=None, region: str = "scene",
                 guard_hours=None, guard_ref_path=None, heartbeat_url=None, heartbeat_every: float = 60.0, keep_days: float = 7.0,
                 summary_path=None):
        self.models, self.bridge = models, bridge
        self.heartbeat_url, self.heartbeat_every, self.keep_days = heartbeat_url, float(heartbeat_every), float(keep_days)
        self.summary = W.MinuteSummary(summary_path) if summary_path else None      # one CSV line per minute (night_report.py reads it)
        self.wd: Optional[W.Watchdog] = None
        self.guard_hours, self.guard_ref_path = guard_hours, guard_ref_path      # (start, end) minutes of the day, the file of the reference of the night
        self.guard_room, self.ref_path = bool(guard_room), ref_path     # guard mode (the room is watched too), the reference file (None = memory only)
        self.fresh_reference, self.notify, self.region = bool(fresh_reference), notify, region
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
        self.sensors: dict = {}                  # name of an outside sensor (a smoke detector) -> is it in alarm now
        self._sensor_sent: dict = {}             # name -> when the last alarm of it was announced (a detector repeats its message)
        self._sensor_lock = threading.Lock()
        self._install_hooks()

    # ---- what the page sees
    def emit(self, ev: dict) -> dict:
        ev = dict(ev)
        ev.setdefault("run", self.run_id)
        out = self.hub.publish(ev)
        if self.summary is not None:
            try:
                self.summary.feed(ev)
            except Exception:  # noqa: BLE001
                pass
        return out

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.emit(dict(type="mode", mode=mode))

    def state(self) -> dict:
        from run_hw3 import USE_CASES
        return dict(mode=self.mode, pending=self.pending, info=self.info, model=self.model_name, fake=self.fake,
                    guard_room=self.guard_room, guard_hours=self._hours_text(), reference_file=(Path(self.ref_path).name if self.ref_path else None),
                    watch=self.watch_opts, last_id=len(self.hub.events),
                    presets={k: v[0] for k, v in USE_CASES.items()})

    # ---- outside sensors (a smoke detector): an alarm of the detector needs no confirmation and no investigation
    def sensor_event(self, name: str, alarm: bool, detail: str = "") -> dict:
        """A sensor outside of the program (a smoke detector, or the hub that it is connected to) reports its state. An ALARM is announced at once:
        the page, the log file, the push message; no model, no confirmation by the camera (the detector is the better sensor for smoke), and it works
        whether the watch is running or not. The same sensor is announced again at most every ROOM_QUIET_S seconds while it keeps reporting."""
        now = time.time()
        with self._sensor_lock:
            was = self.sensors.get(name, False)
            self.sensors[name] = bool(alarm)
            if not alarm:
                self._sensor_sent.pop(name, None)
                if was:
                    self.emit(dict(type="note", text=f"the sensor {name!r} reports: all clear"))
                return dict(ok=True, alarm=False)
            if now - self._sensor_sent.get(name, 0.0) < W.ROOM_QUIET_S:
                return dict(ok=True, alarm=True, repeated=True)
            self._sensor_sent[name] = now
        text = (f"SENSOR ALERT: the sensor {name!r} reports an ALARM" + (f" ({detail})" if detail else "") + ". This comes from a detector, not from the camera: "
                "no confirmation is needed. A person must look NOW (and the detector's own siren and the fire service do not depend on this program).")
        done: list = []
        if self.notify is not None:
            try:
                done = self.notify(text)
            except Exception as e:  # noqa: BLE001
                done = [f"notification FAILED: {type(e).__name__}"]
        self.emit(dict(type="room_alert", facts=[f"the sensor {name!r} reports an alarm"], text=text, evidence=[], notified=done, sensor=name))
        threading.Thread(target=self._sensor_picture, args=(name,), daemon=True, name="sensor-picture").start()
        return dict(ok=True, alarm=True, notified=done)

    def _sensor_picture(self, name: str) -> None:
        """A picture of the measured region at the time of the alarm (after the message has gone out; it waits for the instrument if an agent has it)."""
        try:
            if "capture" in self.bridge.tool_names():
                r = json.loads(self.bridge.call("capture", dict(label="sensor_alert", n_frames=1)))
                if r.get("ok") and r.get("saved_crops"):
                    self.emit(dict(type="room_alert", facts=[], text=f"the picture of the region at the time of the alarm of the sensor {name!r}",
                                   evidence=list(r["saved_crops"]), notified=[], sensor=name))
        except Exception:  # noqa: BLE001
            pass

    def _hours_text(self) -> Optional[str]:
        h = self.guard_hours
        return "%02d:%02d-%02d:%02d" % (h[0] // 60, h[0] % 60, h[1] // 60, h[1] % 60) if h else None

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

    def shutdown(self, wait: float = 20.0) -> None:
        """The program ends (Ctrl+C, a stop signal of the system): the watch finishes its step, the summary of the last minute is written, then the
        instrument is closed. Without the wait the thread of the watch is killed with the process and the last minute of the night is lost."""
        self.stop()
        th = self.thread
        if th is not None and th is not threading.current_thread():
            th.join(wait)
        if self.summary is not None:
            self.summary.flush()

    def reply(self, text: str) -> None:
        if self.pending is None:
            raise Busy("nothing is waiting for a reply")
        self.reply_q.put(str(text))

    # ---- the two modes
    def _watch_main(self, interval: float, tone_every: float) -> None:
        wd = W.Watchdog(self.bridge.call, self.bridge.tool_names(), self.emit, self._investigate, interval, tone_every, ask=self._ask_person,
                        guard=self.guard_room, ref_path=self.ref_path, fresh_reference=self.fresh_reference, notify=self.notify,
                        region=self.region, guard_hours=self.guard_hours, guard_ref_path=self.guard_ref_path)
        self.wd = wd
        stop_ev = self.stop_ev
        threading.Thread(target=self._maintenance, args=(stop_ev, wd), daemon=True, name="maintenance").start()
        try:
            wd.run(stop_ev)
        except Exception as e:  # noqa: BLE001
            self.emit(dict(type="error", text=f"the watchdog stopped: {type(e).__name__}: {str(e)[:300]}"))
            self._push_trouble(f"WATCH TROUBLE: the watchdog CRASHED and the room is NOT watched: {type(e).__name__}: {str(e)[:200]}")
        finally:
            if self.summary is not None:
                self.summary.flush()
            self.wd = None
            self.pending = None
            self.set_mode("idle")

    def _push_trouble(self, text: str) -> None:
        done: list = []
        if self.notify is not None:
            try:
                done = self.notify(text)
            except Exception as e:  # noqa: BLE001
                done = [f"notification FAILED: {type(e).__name__}"]
        self.emit(dict(type="note", text=text + (f"  [{'; '.join(done)}]" if done else "")))

    def _maintenance(self, stop_ev: threading.Event, wd: "W.Watchdog") -> None:
        """While the watch runs: the heartbeat (a ping to HW3_HEARTBEAT_URL, sent only while the loop is healthy; a service such as healthchecks.io
        alerts when the pings stop: that is how a dead program, a sleeping Mac or a lost camera at night is noticed by somebody), and once an hour
        the deletion of evidence pictures older than keep_days. The address of the heartbeat is a secret and is never written to the page."""
        last_prune, last_state = 0.0, None
        while not stop_ev.is_set():
            now = time.time()
            if self.heartbeat_url:
                ok, why = wd.health()
                state, msg = ("ok", "heartbeat: sent (the loop is healthy)")
                if not ok:
                    state, msg = "unhealthy", f"heartbeat: NOT sent, so the service will raise the alarm: {why}"
                else:
                    try:
                        with urllib.request.urlopen(self.heartbeat_url, timeout=10) as r:
                            r.read(64)
                    except Exception as e:  # noqa: BLE001
                        state, msg = "failed", f"heartbeat: the ping FAILED ({type(e).__name__}): no internet?"
                if state != last_state:                         # only a change is written, not every minute
                    self.emit(dict(type="note", text=msg))
                    last_state = state
            if self.keep_days > 0 and now - last_prune >= 3600.0:
                last_prune = now
                n = plat.prune_old(self.work / "crops", self.keep_days)
                if n:
                    self.emit(dict(type="note", text=f"{n} evidence picture(s) older than {self.keep_days:g} days were deleted"))
            stop_ev.wait(max(0.2, min(self.heartbeat_every, 3600.0)) if self.heartbeat_url else 30.0)

    def _ask_person(self, question: str):
        """The watchdog asks the person itself (no model): the same pause bar of the page. None = the run was stopped."""
        try:
            return self._ask(dict(kind="scene_action", question=question), timeout=W.ASK_TIMEOUT_S)
        except Aborted:
            return None

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
    def _ask(self, payload: dict, timeout: Optional[float] = None) -> str:
        """Wait for the person. `timeout` (the program's own questions only, never the agent's): after that many seconds without an answer the
        question counts as an empty answer and the watch goes on, so that a night guard does not wait for ever for somebody who is asleep."""
        self.mode_before_pause = self.mode if self.mode != "paused" else self.mode_before_pause
        self.pending = payload
        self.set_mode("paused")
        self.emit(dict(type="pause", payload=payload))
        t0 = time.time()
        while True:
            try:
                r = self.reply_q.get(timeout=0.5)
            except queue.Empty:
                if self.stop_ev.is_set():
                    raise Aborted()
                if timeout and time.time() - t0 >= timeout:
                    self.pending = None
                    self.emit(dict(type="note", text=f"nobody answered in {timeout:g} s: the question counts as an empty answer (keep), the watch goes on"))
                    self.set_mode(self.mode_before_pause)
                    return ""
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
        files = {"fake_state.json": ("covered", "light", "shift", "person", "warm"),
                 "fake_audio_state.json": ("muted", "noise_db", "volume_db", "coupling_db")}
        done = {}
        if "smoke" in kw:                                        # the synthetic smoke detector goes through the same door as a real one
            self.sensor_event("smoke-detector", bool(kw["smoke"]), "simulated")
            done["smoke"] = bool(kw["smoke"])
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

    @app.post("/api/guard/postpone")
    def guard_postpone(body: dict = Body(default={})):
        """'I am in the lab': pause the guard for N minutes (0 = resume now). The only way to tell the program that somebody is allowed to be there."""
        try:
            minutes = float(body.get("minutes", 60))
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content=dict(ok=False, error="minutes must be a number"))
        if not 0 <= minutes <= 720:
            return JSONResponse(status_code=400, content=dict(ok=False, error="minutes must be from 0 (resume now) to 720"))
        wd = session.wd
        if wd is None or not (wd.guard or wd.guard_hours):
            return JSONResponse(status_code=409, content=dict(ok=False, error="the guard is not running (start watching with --guard or --guard-hours)"))
        return dict(ok=True, text=wd.postpone(minutes))

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


SENSOR_NAME = re.compile(r"[A-Za-z0-9_.-]{1,40}")


SIG_WINDOW_S = 300.0                 # a signed message older than this (or from the future) is refused, and a signature is accepted only once


def sign_message(secret: str, timestamp: str, body: bytes) -> str:
    """The signature of a sensor message: HMAC-SHA256 of 'timestamp.body' with the shared secret (hex). The secret itself never travels, and a
    message that somebody copied from the network cannot be sent again (the timestamp is old, the signature was used)."""
    return hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()


def create_sensor_app(session: Session, token: str, signed_only: bool = False):
    """The door for outside sensors: one route. It is a SEPARATE little server (own port) so that the page, which can start agents and spend
    money, stays on the loopback address while a hub on the network may reach this door only.
    Two ways in: X-Signature + X-Timestamp (HMAC of the message, replay-safe: use this one on a network) or the plain secret in X-Token
    (simple, for a hub that cannot compute a signature; the secret travels in clear text). signed_only refuses the second."""
    if FastAPI is None:
        raise SystemExit("pip install fastapi uvicorn")
    app = FastAPI(title="sensor input")
    seen: dict = {}

    def refuse(msg: str):
        return JSONResponse(status_code=401, content=dict(ok=False, error=msg))

    @app.post("/api/sensor")
    async def sensor(request: Request):
        raw = await request.body()
        sig, ts, tok = request.headers.get("x-signature"), request.headers.get("x-timestamp"), request.headers.get("x-token")
        if sig:
            try:
                age = abs(time.time() - float(ts))
            except (TypeError, ValueError):
                return refuse("X-Timestamp must be the time in seconds since 1970")
            if age > SIG_WINDOW_S:
                return refuse("the message is too old (or the clocks differ by more than 5 minutes)")
            if not hmac.compare_digest(sig.encode("utf-8"), sign_message(token, ts, raw).encode("utf-8")):
                return refuse("wrong signature")
            now = time.time()
            for k in [k for k, t in seen.items() if now - t > SIG_WINDOW_S]:
                seen.pop(k, None)
            if sig in seen:
                return refuse("this signed message was already used (a copy cannot be sent again)")
            seen[sig] = now
        elif tok and not signed_only:
            if not hmac.compare_digest(tok.encode("utf-8"), token.encode("utf-8")):
                return refuse("wrong or missing X-Token header")
        else:
            return refuse("send X-Signature and X-Timestamp" + ("" if signed_only else " (or the secret in X-Token)"))
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            body = None
        name, alarm = (body.get("sensor"), body.get("alarm")) if isinstance(body, dict) else (None, None)
        if not isinstance(name, str) or not SENSOR_NAME.fullmatch(name) or not isinstance(alarm, bool):
            return JSONResponse(status_code=400, content=dict(ok=False, error='send {"sensor": "smoke-1", "alarm": true} (name: letters, digits, _ . -; alarm: true or false)'))
        return await asyncio.to_thread(session.sensor_event, name, alarm, str(body.get("detail", ""))[:120])

    return app


def build_session(args) -> Session:
    plat.utf8_console()
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("Set OPENROUTER_API_KEY (or GOOGLE_API_KEY / ANTHROPIC_API_KEY) first.")
    if args.hints:
        try:
            print("hints replaced by", args.hints + ":", ", ".join(W.load_hints(args.hints)))
        except (OSError, ValueError) as e:
            sys.exit(f"--hints {args.hints}: {e}")
    hours = None   # (a mistake in a setting is said before the instrument is started)
    if args.guard_hours:
        try:
            hours = W.parse_hours(args.guard_hours)
        except ValueError as e:
            sys.exit(str(e))
        args.guard = True
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
    W.GUARD_SETTLE_S = float(args.guard_settle)
    gref = (work / "fake_guard_reference.json") if args.fake else Path(args.guard_reference).resolve()
    ref = None if args.no_reference else (work / "fake_reference.json" if args.fake else Path(args.reference).resolve())   # a synthetic reference never replaces the real one
    url = args.notify_url or os.environ.get("HW3_NOTIFY_URL") or None
    notify = W.make_notifier(Path(args.alerts).resolve(), url)         # also for the alarms of outside sensors
    if args.guard:
        print((f"GUARD HOURS {args.guard_hours}: from {W.GUARD_SETTLE_S:g} s after the window begins the room is watched against the reference of the night"
               if hours else "GUARD MODE: the room is watched against the reference")
              + f"; alerts go to {Path(args.alerts).name}"
              + (", to the push address (HW3_NOTIFY_URL)" if url else " (set HW3_NOTIFY_URL for a push message to a phone)"))
    if ref:
        print("reference file:", ref.name, "(deleted or --fresh-reference = measured anew)")
    return Session(models, SharedBridge(bridge), work, Path(args.runs).resolve(), args.max_decisions, info,
                   model_name=args.model or os.environ.get("AGENT_MODEL", ""), fake=args.fake,
                   guard_room=args.guard, ref_path=ref, fresh_reference=args.fresh_reference, notify=notify, region=args.region,
                   guard_hours=hours, guard_ref_path=gref, heartbeat_url=(args.heartbeat_url or os.environ.get("HW3_HEARTBEAT_URL") or None),
                   heartbeat_every=args.heartbeat_every, keep_days=args.keep_days, summary_path=work / "watch_summary.csv")


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
    ap.add_argument("--guard", action="store_true", help="GUARD MODE: nobody should be in the room; a change of the picture, the light or the noise "
                                                         "(with the instrument tested and in order) is a ROOM ALERT")
    ap.add_argument("--guard-hours", default=None, metavar="HH:MM-HH:MM",
                    help="GUARD HOURS, for example 22:00-07:00 (the clock of this computer; may cross midnight): people are normal by day, so the room is "
                         "watched only inside this window; implies --guard")
    ap.add_argument("--guard-settle", type=float, default=W.GUARD_SETTLE_S, help="guard hours: seconds after the window begins before the room is taken as the reference of the night")
    ap.add_argument("--guard-reference", default="guard_reference.json", help="guard hours: file of the reference of the night (kept for restarts inside the same night)")
    ap.add_argument("--reference", default="reference.json", help="file of the reference state (read at the start; only a person's 'accept' changes it)")
    ap.add_argument("--no-reference", action="store_true", help="keep the reference only in memory (nothing is read or written)")
    ap.add_argument("--fresh-reference", action="store_true", help="ignore the stored reference and measure a new one (it replaces the file)")
    ap.add_argument("--region", default="scene", choices=["scene", "main_view", "panorama"],
                    help="the part of the picture that the watchdog measures (guard mode: it must cover where a person could appear; 'scene' is the region of roi.json)")
    ap.add_argument("--sensor-port", type=int, default=None,
                    help="open the door for outside sensors (a smoke detector or its hub) on this port: POST /api/sensor with the header X-Token (secret in env HW3_SENSOR_TOKEN)")
    ap.add_argument("--sensor-signed-only", action="store_true", help="the sensor door refuses the plain secret in X-Token and accepts only signed messages "
                                                                      "(X-Signature, X-Timestamp; sensor_send.py makes them)")
    ap.add_argument("--heartbeat-url", default=None, help="a ping address (env HW3_HEARTBEAT_URL, a secret; for example the ping URL of healthchecks.io): "
                                                         "called every --heartbeat-every seconds while the loop is healthy; when it stops, the service raises the alarm")
    ap.add_argument("--heartbeat-every", type=float, default=60.0)
    ap.add_argument("--keep-days", type=float, default=7.0, help="evidence pictures (work/crops) older than this are deleted (0 = keep all)")
    ap.add_argument("--hints", default=None, help="JSON file that replaces the hints that the agent gets (keys: " + ", ".join(W.HINT_KEYS) + ")")
    ap.add_argument("--autostart", action="store_true", help="start watching at once, without pressing the button (for a guard that starts with the computer)")
    ap.add_argument("--sensor-host", default="127.0.0.1", help="address of that door (0.0.0.0 = reachable from the network, the token is then the only protection)")
    ap.add_argument("--alerts", default="alerts.log", help="guard mode: file with one line per ROOM ALERT")
    ap.add_argument("--notify-url", default=None, help="guard mode: POST the alert text to this address (env HW3_NOTIFY_URL), for example "
                                                      "https://ntfy.sh/<your-secret-topic> (phone app ntfy)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    try:
        import uvicorn
        import fastapi  # noqa: F401
    except ImportError:
        sys.exit("pip install fastapi uvicorn")
    if args.heartbeat_every < 5:
        sys.exit("--heartbeat-every must be at least 5 seconds")
    token = os.environ.get("HW3_SENSOR_TOKEN", "")
    if args.sensor_port and len(token) < 12:
        sys.exit("--sensor-port needs a secret of at least 12 characters in the environment: export HW3_SENSOR_TOKEN=...   (signed messages need it too: sensor_send.py)")
    session = build_session(args)
    app = create_app(session)
    if args.autostart:
        session.start_watch(4.0, 0.0)
        print("autostart: watching has begun (interval 4 s, no beep check)", flush=True)
    if args.sensor_port:
        srv = uvicorn.Server(uvicorn.Config(create_sensor_app(session, token, args.sensor_signed_only), host=args.sensor_host, port=args.sensor_port,
                                            log_level="warning"))
        threading.Thread(target=srv.run, daemon=True, name="sensor-door").start()
        print(f"sensor door open: POST http://{args.sensor_host}:{args.sensor_port}/api/sensor  (signed: X-Signature + X-Timestamp, see sensor_send.py{'; ONLY signed messages' if args.sensor_signed_only else '; or the secret in X-Token'})", flush=True)
    print(f"\nopen  http://{args.host}:{args.port}   (Ctrl+C stops the server)\n", flush=True)

    def on_term(signum, frame):                   # a stop of the system (launchd, kill, a shutdown) ends the program the same orderly way as Ctrl+C
        raise SystemExit(0)
    try:
        signal.signal(signal.SIGTERM, on_term)
    except (ValueError, OSError):
        pass
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        session.shutdown()
        session.bridge.close()


if __name__ == "__main__":
    main()

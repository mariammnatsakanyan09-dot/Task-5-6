"""The link between the (synchronous) agent graph and the MCP camera server.

The graph is run with graph.stream(...), which is synchronous, but an MCP session is asynchronous and must stay open for
the whole run (a new server process per call would forget the camera). So the session lives in a background thread with
its own event loop: the thread opens the session once, the graph calls `call()` from its own thread, and `close()` shuts
the server down at the end.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

HERE = Path(__file__).resolve().parent


def text_of(result: Any) -> str:
    """MCP tool results arrive as a list of content blocks (or a string): join the text parts."""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        parts = []
        for b in result:
            if isinstance(b, dict):
                parts.append(str(b.get("text", "")))
            elif isinstance(b, str):
                parts.append(b)
            else:
                parts.append(str(getattr(b, "text", b)))
        return "".join(parts)
    return str(result)


class McpBridge:
    def __init__(self, server_script: Optional[str] = None, env: Optional[dict] = None, cwd: Optional[str] = None,
                 python: Optional[str] = None):
        self.script = str(server_script or HERE / "mcp_camera_server.py")
        self.env = dict(env or {})
        self.cwd = cwd
        self.python = python or sys.executable
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._stop: Optional[asyncio.Event] = None
        self._main_future = None
        self._tools: dict[str, Any] = {}

    # ---- life cycle
    def start(self, timeout: float = 90.0) -> list:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-bridge", daemon=True)
        self._thread.start()
        self._main_future = asyncio.run_coroutine_threadsafe(self._main(), self._loop)
        if not self._ready.wait(timeout):
            self.close()
            raise RuntimeError(f"the MCP server did not start within {timeout:.0f} s")
        if self._error is not None:
            err = self._error
            self.close()
            raise RuntimeError(f"the MCP server could not be started: {err}") from err
        return self.tools()

    async def _main(self) -> None:
        try:
            self._stop = asyncio.Event()
            conn: dict = dict(command=self.python, args=[self.script], transport="stdio", env=self.env)
            if self.cwd:
                conn["cwd"] = self.cwd
            client = MultiServerMCPClient({"camera": conn})
            # the session is entered and left in this same task, as the mcp library requires
            async with client.session("camera") as session:
                for t in await load_mcp_tools(session):
                    self._tools[t.name] = t
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001
            self._error = e
            self._ready.set()

    def close(self) -> None:
        loop = self._loop
        if loop is None:
            return
        if self._stop is not None:
            loop.call_soon_threadsafe(self._stop.set)
        if self._main_future is not None:
            try:
                self._main_future.result(timeout=20)
            except Exception:  # noqa: BLE001
                pass
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- use
    def tools(self) -> list:
        """The langchain tools of the server (their calls must go through call(), not through .invoke())."""
        return list(self._tools.values())

    def tool_names(self) -> list[str]:
        return list(self._tools)

    def tool_specs(self) -> list[dict]:
        """The tools in the OpenAI function format (name, description, JSON schema) that chat models accept in
        bind_tools. The model only sees these descriptions; the calls themselves go through call()."""
        specs = []
        for t in self._tools.values():
            schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
            schema = {k: v for k, v in schema.items() if k != "title"}
            for prop in (schema.get("properties") or {}).values():
                prop.pop("title", None)
            specs.append({"type": "function",
                          "function": {"name": t.name, "description": t.description or "", "parameters": schema}})
        return specs

    def call(self, name: str, args: Optional[dict] = None, timeout: float = 180.0) -> str:
        """Call one tool of the server and return its answer as text. Never raises for a tool-level problem: the text
        then says what went wrong."""
        if name not in self._tools:
            return '{"ok": false, "error": "unknown tool %s; available: %s"}' % (name, ", ".join(self._tools))
        if self._loop is None:
            return '{"ok": false, "error": "the camera server is not running"}'
        fut = asyncio.run_coroutine_threadsafe(self._tools[name].ainvoke(args or {}), self._loop)
        try:
            text = text_of(fut.result(timeout)).strip()
            if not text.startswith("{"):                 # e.g. the library's "Error executing tool ..." (bad arguments)
                text = json.dumps(dict(ok=False, error="rejected: " + " ".join(text.split())[:400]))
            return text
        except Exception as e:  # noqa: BLE001  (a dead server, a timeout)
            fut.cancel()
            msg = str(e).replace('"', "'")[:400]
            return '{"ok": false, "error": "tool call failed: %s: %s"}' % (type(e).__name__, msg)


def default_env(workdir: str, roi_file: Optional[str] = None, camera_index: Optional[int] = None,
                fake: bool = False, fake_state: Optional[str] = None, controls_file: Optional[str] = None,
                fake_audio: bool = False, fake_audio_state: Optional[str] = None) -> dict:
    """Environment given to the camera server (the mcp client merges it with the default safe environment)."""
    env = {"HW3_WORKDIR": str(Path(workdir).resolve()), "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
    if roi_file:
        env["HW3_ROI_FILE"] = str(Path(roi_file).resolve())
    if controls_file:
        env["HW3_CONTROLS_FILE"] = str(Path(controls_file).resolve())
    for k in ("HW3_CV_API", "HW3_FOURCC", "HW3_UVC_UTIL", "HW3_UVC_TARGET", "HW3_UVC", "HW3_FAKE_UVC_STATE",
              "HW3_SCREEN_BACKEND", "HW3_SCREEN_GEOM", "HW3_SETTLE_S", "HW3_FAKE_SCREEN", "HW3_VIEWER_HEADLESS",
              "HW3_AUDIO", "HW3_AUDIO_BACKEND", "HW3_AUDIO_NAME", "HW3_AUDIO_OUT", "HW3_AUDIO_IN", "HW3_AUDIO_MAX_DB", "HW3_FAKE_AUDIO"):   # platform choices made in the shell are passed on
        if os.environ.get(k):
            env[k] = os.environ[k]
    env["HW3_AUDIO_FILE"] = str(Path(os.environ.get("HW3_AUDIO_FILE") or "audio.json").resolve())   # the pair chosen by audio_check.py --pairs
    if camera_index is not None:
        env["HW3_CAMERA_INDEX"] = str(camera_index)
    if fake:
        env["HW3_CAMERA_BACKEND"] = "fake"
        if fake_state:
            env["HW3_FAKE_STATE"] = str(Path(fake_state).resolve())
    if fake_audio:                                  # the synthetic loudspeaker and microphone (tests only)
        env["HW3_AUDIO_BACKEND"] = "fake"
        env["HW3_AUDIO"] = "on"
        if fake_audio_state:
            env["HW3_FAKE_AUDIO"] = str(Path(fake_audio_state).resolve())
    # the server must see the same python packages and the project folder
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
    return env

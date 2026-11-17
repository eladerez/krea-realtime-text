"""Phase 1 of the live demo (LIVE.md): a web server and a generation loop. While the generator makes block after block,
the viewer types on the page; before each block the loop hands the generator the latest typed text and the key sequence
number, and the generator writes it into the video (Krea: runner/gen_krea.py --live, its memory write; stub: our font on
a still image, to test the web loop without a GPU). Frames go to the browser as JPEG over a WebSocket as soon as a block
is done, each tagged with the last key sequence number the generator applied before that block, so the page can time
keypress -> screen. Every block is logged to --log (JSON lines).
Usage (stub, on the Jetson): uv run --with fastapi --with uvicorn python runner/live_server.py --stub \
  --start data/start_frames_swaptest/race_track_switch-s0.png --text "LAP 7" --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import traceback
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # module level: FastAPI resolves the WebSocket annotation
from fastapi.responses import HTMLResponse                   # from module globals (inside a function it gave HTTP 403)
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAGE = Path(__file__).with_name("live_page.html")


class Generator(Protocol):
    def step(self, typed: str, seq: int) -> tuple[list[np.ndarray], dict]:
        """One block: write `typed` if it differs from what is shown, then make the block's frames (uint8 HxWx3)."""


class LiveState:
    """The latest typed text and its key sequence number, shared by the web handler and the generation thread."""

    def __init__(self, text: str) -> None:
        self._lock = threading.Lock()
        self.text, self.seq = text, 0
        self.start = text  # code-live-lcd-r1 #1: the start number, for the page's Reset after a reconnect

    def set(self, text: str, seq: int) -> None:
        with self._lock:
            if seq >= self.seq:
                self.text, self.seq = text, seq

    def get(self) -> tuple[str, int]:
        with self._lock:
            return self.text, self.seq


class LabelBox:
    """Where the sign's text is, kept across writes (LIVE.md Changes, 15 Nov). Finding the label by its old text failed
    after the first write (OCR cannot read a lone "2" in our font; a lone digit also matches the keypad), so the box is
    found once, by OCR of the start text, widened towards `align`'s far side to fit `max_chars` characters (a
    calculator is right-aligned), and then kept: the prompts hold the camera steady, and write_memory's box tracking
    follows small motion between memory frames."""

    def __init__(self, max_chars: int = 5, align: str = "right") -> None:
        self.box: tuple[int, int, int, int] | None = None
        self.max_chars, self.align = max_chars, align

    def update(self, frame: np.ndarray, shown: str) -> tuple[int, int, int, int] | None:
        if self.box is None and shown.strip():
            from textbench.erase import text_boxes
            found = text_boxes(frame, [shown])
            if found:
                x0, y0, x1, y1 = max(found, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                want = int((x1 - x0) / len(shown.strip()) * self.max_chars)
                if self.align == "right":
                    x0 = max(0, x1 - want)
                elif self.align == "left":
                    x1 = min(frame.shape[1], x0 + want)
                self.box = (x0, y0, x1, y1)
        return self.box


class StubGenerator:
    """No model: the start image with the shown text drawn in our font (textbench/write_text.py), a little motion
    noise, 12 frames per block at ~3.3 s per block (Krea's measured pace, SPEED.md 40d), so latency looks realistic.
    check: the display checker as in the Krea loop (textbench/display_check.py), with a fix (font write) of a wrong
    reading before the next block; drift: the chance per block that one shown digit turns into another, standing in
    for Krea's redraws, so the check and the fix can be tested without a GPU."""

    def __init__(self, start: np.ndarray, text: str, block_s: float = 3.3, frames: int = 12, check: bool = False,
                 drift: float = 0.0) -> None:
        from textbench.write_text import write
        self.write, self.base, self.shown, self.block_s, self.n = write, start, text, block_s, frames
        self.frame = start.copy()
        self.applied = 0
        self.label = LabelBox()
        self.label.update(start, text)
        self.check, self.drift = check, drift
        self.counts = {"ok": 0, "wrong": 0, "unread": 0}
        self.fix: str | None = None
        self.rng = np.random.default_rng(32)

    def _draw(self, old: str, new: str) -> bool:
        if self.label.box is None or old == new:
            return False
        self.frame, _ = self.write(self.frame, [(old, new)], str(ROOT), self.label.align, {old: self.label.box})
        return True

    def step(self, typed: str, seq: int) -> tuple[list[np.ndarray], dict]:
        t0 = time.perf_counter()
        wrote, info = False, {}
        if typed != self.shown and typed.strip():
            if (self.fix or self.shown) == typed or self._draw(self.fix or self.shown, typed):  # code-check-r1 #13
                self.shown, wrote, self.applied, self.fix = typed, True, seq, None
        elif self.fix is not None:
            info["fix"] = {"from": self.fix, "to": self.shown, "boxes": int(self._draw(self.fix, self.shown))}
            self.fix = None
        if self.drift and self.shown and self.rng.random() < self.drift:  # a stand-in for Krea redrawing a digit
            i = int(self.rng.integers(len(self.shown)))
            bad = self.shown[:i] + str((int(self.shown[i]) + 1 + int(self.rng.integers(8))) % 10) + self.shown[i + 1:]
            self._draw(self.fix or self.shown, bad)
            self.fix = bad
            info["drift"] = bad
        rng = np.random.default_rng(int(t0 * 1000) % 2**32)
        frames = [np.clip(self.frame.astype(np.int16) + rng.integers(-3, 4, self.frame.shape), 0, 255).astype(np.uint8)
                  for _ in range(self.n)]
        if self.check and self.label.box is not None:
            from textbench.display_check import read_display, verdict
            r = read_display(frames[-1], self.label.box)
            v = verdict(r["reads"], self.shown)
            self.counts[v] += 1
            self.fix = r["reads"] if v == "wrong" and r["score"] >= 0.8 else None
            total = sum(self.counts.values())
            info["display"] = {"reads": r["reads"], "score": r["score"], "verdict": v, "want": self.shown,
                               "ok_pct": round(100 * self.counts["ok"] / total, 1), "checked": total,
                               "counts": dict(self.counts), "fix_queued": self.fix is not None}
        time.sleep(max(0.0, self.block_s - (time.perf_counter() - t0)))
        return frames, {"wrote": wrote, "shown": self.shown, "applied_seq": self.applied, **info}


def jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def run_loop(gen: Generator, state: LiveState, send: Callable[[dict, bytes], None], stop: threading.Event,
             log_path: Path) -> None:
    """Block after block until stopped; each block's frames go out with the key it applied."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # LIVE.md 17 Nov (the LinkedIn recording): every frame sent is also kept as the same JPEG, <log>_frames/<id>.jpg,
    # so a live session can be cut into a video with the keypresses (<log>_keys.jsonl) afterwards
    frames_dir = log_path.with_name(log_path.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_id = 0
    with log_path.open("a") as log:
        while not stop.is_set():
            typed, seq = state.get()
            t0 = time.time()
            frames, info = gen.step(typed, seq)
            t1 = time.time()
            rec = {"t_start": t0, "block_s": round(t1 - t0, 3), "typed": typed, "seq": seq, "frames": len(frames),
                   "first_frame": frame_id, **info}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            for f in frames:
                data = jpeg(f)
                (frames_dir / f"{frame_id:06d}.jpg").write_bytes(data)
                send({"frame": frame_id, "applied_seq": info.get("applied_seq", 0), "shown": info.get("shown"),
                      "wrote": info.get("wrote", False), "block_s": rec["block_s"], "n": len(frames),
                      "display": info.get("display"), "fixed": bool((info.get("fix") or {}).get("boxes"))}, data)
                frame_id += 1


def make_app(state: LiveState, start_loop: Callable[[Callable[[dict, bytes], None]], None], log_path: Path,
             title: str = "Live Sign") -> FastAPI:
    app = FastAPI()
    events = log_path.with_name(log_path.stem + "_keys.jsonl")  # code-live-r1 #5: keys and page-measured latencies

    def note(rec: dict) -> None:
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("a") as f:
            f.write(json.dumps({"t_server": time.time(), **rec}) + "\n")
    clients: set[asyncio.Queue] = set()
    holder: dict = {}

    def send(meta: dict, data: bytes) -> None:  # called from the generation thread
        loop = holder.get("loop")
        if loop is None:
            return
        header = json.dumps(meta).encode()
        msg = len(header).to_bytes(4, "big") + header + data
        for q in list(clients):
            loop.call_soon_threadsafe(_offer, q, msg)

    def _offer(q: asyncio.Queue, msg: bytes) -> None:
        if q.qsize() > 120:  # a slow client drops old frames rather than lagging forever
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(msg)

    @app.on_event("startup")
    async def _startup() -> None:
        holder["loop"] = asyncio.get_running_loop()
        start_loop(send)

    @app.get("/")
    async def page() -> HTMLResponse:
        return HTMLResponse(PAGE.read_text())

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        q: asyncio.Queue = asyncio.Queue()
        clients.add(q)
        text, seq = state.get()
        await socket.send_text(json.dumps({"type": "hello", "text": text, "start": state.start, "seq": seq,
                                           "title": title}))  # r1 #6

        async def pump() -> None:
            while True:
                await socket.send_bytes(await q.get())
        task = asyncio.create_task(pump())
        try:
            while True:
                msg = json.loads(await socket.receive_text())
                if msg.get("type") == "text":
                    state.set(str(msg["text"])[:24], int(msg["seq"]))
                    note({"type": "key", "seq": int(msg["seq"]), "text": str(msg["text"])[:24]})
                elif msg.get("type") == "lat":  # measured on the page: key event -> first frame shown with it
                    note({"type": "lat", "seq": int(msg["seq"]), "ms": float(msg["ms"]), "frame": int(msg["frame"]),
                          "oldest_ms": float(msg.get("oldest_ms", msg["ms"]))})
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()
            clients.discard(q)
    return app


def serve(gen_factory: Callable[[], Generator], text: str, port: int, log_path: Path, host: str = "127.0.0.1",
          title: str = "Live Sign") -> None:
    """Start the server; the generator is built in the generation thread (Krea's CUDA work stays on one thread)."""
    import uvicorn
    state, stop = LiveState(text), threading.Event()

    def start_loop(send: Callable[[dict, bytes], None]) -> None:
        def worker() -> None:
            try:
                run_loop(gen_factory(), state, send, stop, log_path)
            except Exception:  # code-live-r1 #3: a dead generator ends the process, so the VM does not idle
                traceback.print_exc()
                sys.stdout.flush()
                os._exit(1)
        threading.Thread(target=worker, daemon=True).start()

    uvicorn.run(make_app(state, start_loop, log_path, title), host=host, port=port, log_level="warning")
    stop.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true", help="no model: the start image with our font (tests the web loop)")
    ap.add_argument("--start", type=Path, required=True)
    ap.add_argument("--text", default="LAP 7")
    ap.add_argument("--title", default="Live Sign")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="stub only; e.g. the Jetson's Tailscale address for a phone")
    ap.add_argument("--log", type=Path, default=ROOT / "logs/live/stub.jsonl")
    ap.add_argument("--check", action="store_true", help="stub: run the display checker and fix (as live_check)")
    ap.add_argument("--drift", type=float, default=0.0, help="stub: chance per block that a shown digit changes")
    a = ap.parse_args()
    if not a.stub:
        raise SystemExit("the real generator runs from runner/gen_krea.py --live")
    start = np.asarray(Image.open(a.start).convert("RGB"))
    serve(lambda: StubGenerator(start, a.text, check=a.check, drift=a.drift), a.text, a.port, a.log, a.host, a.title)


if __name__ == "__main__":
    main()

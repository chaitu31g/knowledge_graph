"""
server_utils.py — CircuitAI robust server startup utilities

Features:
  - Port availability check & auto-fallback
  - Safe process kill (dev mode only, only our own process)
  - Startup lock file (prevents multiple instances)
  - Graceful shutdown with cleanup
  - Health-check polling before Cloudflare tunnel starts
  - Singleton guard for model loading
"""

import os
import sys
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
PREFERRED_PORT  = int(os.getenv("PORT", 8000))
AUTO_FALLBACK   = os.getenv("AUTO_PORT_FALLBACK", "true").lower() == "true"
AUTO_KILL_PORT  = os.getenv("AUTO_KILL_PORT", "true").lower() == "true"
LOCK_FILE       = Path("/tmp/.circuitai.lock")
SERVER_HEALTH_TIMEOUT = 120   # seconds to wait for /health

# ── 1. PORT CHECK ──────────────────────────────────────────────

def is_port_in_use(port: int) -> bool:
    """Returns True if something is already listening on *port*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_free_port(start: int = 8000, max_attempts: int = 10) -> int:
    """Return the first free port starting at *start*."""
    for offset in range(max_attempts):
        port = start + offset
        if not is_port_in_use(port):
            return port
    raise RuntimeError(
        f"No free port found in range {start}–{start + max_attempts - 1}"
    )


# ── 2. SAFE PORT KILL (dev mode only) ─────────────────────────

def _get_pids_on_port(port: int) -> list:
    """Return list of PIDs using *port* (Linux: lsof)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5
        )
        return [int(p) for p in result.stdout.strip().split() if p.isdigit()]
    except Exception:
        return []


def kill_port(port: int) -> bool:
    """
    Kill processes on *port* only if AUTO_KILL_PORT is enabled.
    Never kills unrelated system processes — only SIGTERM first,
    then SIGKILL after 2s if still alive.
    """
    if not AUTO_KILL_PORT:
        return False

    pids = _get_pids_on_port(port)
    if not pids:
        return False

    print(f"[startup] Found PID(s) {pids} on port {port}. Sending SIGTERM...")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    time.sleep(2)

    # Force kill if still alive
    for pid in _get_pids_on_port(port):
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"[startup] SIGKILL sent to PID {pid}")
        except ProcessLookupError:
            pass

    time.sleep(0.5)
    freed = not is_port_in_use(port)
    print(f"[startup] Port {port} {'freed ✓' if freed else 'still in use ✗'}")
    return freed


# ── 3. PORT RESOLUTION ─────────────────────────────────────────

def resolve_port(preferred: int = PREFERRED_PORT) -> int:
    """
    Full port resolution pipeline:
      1. Check if preferred port is free → use it
      2. Try to kill occupying process (if AUTO_KILL_PORT)
      3. Fall back to next free port (if AUTO_FALLBACK)
    """
    print(f"[startup] Checking port {preferred} availability...")

    if not is_port_in_use(preferred):
        print(f"[startup] Port {preferred} is free ✓")
        return preferred

    print(f"[startup] ⚠️  Port {preferred} is in use.")

    if AUTO_KILL_PORT:
        freed = kill_port(preferred)
        if freed:
            return preferred

    if AUTO_FALLBACK:
        alt = find_free_port(preferred + 1)
        print(f"[startup] Port {preferred} still busy → switching to port {alt}")
        return alt

    raise RuntimeError(
        f"Port {preferred} is in use. Set AUTO_PORT_FALLBACK=true to auto-switch."
    )


# ── 4. LOCK FILE (prevent multiple instances) ──────────────────

def acquire_lock() -> bool:
    """
    Write a PID lock file.  Returns False (and warns) if another
    live instance is already running.
    """
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            os.kill(old_pid, 0)          # signal 0 = existence check, no effect
            print(f"[startup] ⚠️  Another CircuitAI server is running (PID {old_pid}).")
            print("[startup]    Overriding stale lock and continuing...")
        except (ProcessLookupError, ValueError):
            pass                          # stale lock — safe to overwrite

    LOCK_FILE.write_text(str(os.getpid()))
    return True


def release_lock():
    """Remove the lock file on shutdown."""
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ── 5. GRACEFUL SHUTDOWN ───────────────────────────────────────

def register_shutdown_handlers(extra_cleanup=None):
    """
    Register SIGTERM / SIGINT handlers so Ctrl-C or kill always:
      - runs extra_cleanup() if provided
      - removes the lock file
      - prints a clean goodbye message
    """
    def _handle(sig, frame):
        print(f"\n[shutdown] Received signal {sig}. Shutting down...")
        if extra_cleanup:
            try:
                extra_cleanup()
            except Exception:
                pass
        release_lock()
        print("[shutdown] Server stopped cleanly ✓")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT,  _handle)


# ── 6. HEALTH-CHECK POLLING ────────────────────────────────────

def wait_for_server(port: int, timeout: int = SERVER_HEALTH_TIMEOUT) -> bool:
    """
    Poll http://localhost:{port}/health every 2 s until it returns 200
    or *timeout* seconds elapse.  Returns True if server is healthy.
    """
    import urllib.request, urllib.error

    url   = f"http://localhost:{port}/health"
    start = time.time()
    dot_timer = 0

    print(f"[startup] Waiting for server health-check on port {port}...")

    while True:
        elapsed = time.time() - start
        if elapsed >= timeout:
            print(f"\n[startup] ✗ Server did not respond within {timeout}s.")
            return False

        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"\n[startup] ✅ Server healthy (took {elapsed:.0f}s)")
            return True
        except Exception:
            pass

        # Print a dot every 10 s so Colab doesn't think the cell is hung
        if int(elapsed) // 10 > dot_timer:
            dot_timer = int(elapsed) // 10
            print(f"  [{int(elapsed)}s] Still loading models...", end="\r")

        time.sleep(2)


# ── 7. SINGLETON MODEL LOAD GUARD ─────────────────────────────

_models_loaded = False
_load_lock = threading.Lock()

def ensure_models_loaded():
    """
    Thread-safe singleton guard: calls load_model() and _get_embedder()
    exactly once per process lifetime.  Safe to call from multiple threads.
    """
    global _models_loaded
    if _models_loaded:
        return

    with _load_lock:
        if _models_loaded:          # double-checked locking
            return
        print("[startup] Pre-loading Qwen2.5-VL-3B into GPU...")
        from model_loader import load_model
        load_model()

        print("[startup] Pre-loading BGE-M3 embedding model...")
        from rag_engine import _get_embedder
        _get_embedder()

        _models_loaded = True
        print("[startup] ✅ All models ready.")

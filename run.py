"""
run.py  —  ASL Translator launcher
────────────────────────────────────
Usage:  python run.py
"""

import sys
import time
import socket
import threading
import webbrowser
import subprocess
from pathlib import Path

ROOT          = Path(__file__).resolve().parent
PYTHON        = ROOT / "venv" / "Scripts" / "python.exe"
BACKEND_PORT  = 8000
FRONTEND_PORT = 5500


# ─────────────────────────────────────────────────────────────
def port_ready(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("localhost", port)) == 0


def wait_for_port(port: int, timeout: int = 30) -> bool:
    print(f"     waiting", end="", flush=True)
    for _ in range(timeout * 2):
        if port_ready(port):
            print("  ready ✓")
            return True
        print(".", end="", flush=True)
        time.sleep(0.5)
    print("  timed out ✗")
    return False


def open_browser_when_ready():
    for _ in range(60):
        if port_ready(FRONTEND_PORT) and port_ready(BACKEND_PORT):
            time.sleep(0.3)
            webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
            return
        time.sleep(0.5)


# ─────────────────────────────────────────────────────────────
def main():
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║      🤟  ASL Translator  —  FYP          ║")
    print("  ╠══════════════════════════════════════════╣")
    print(f"  ║  Frontend  →  http://localhost:{FRONTEND_PORT}        ║")
    print(f"  ║  Backend   →  http://localhost:{BACKEND_PORT}        ║")
    print("  ╠══════════════════════════════════════════╣")
    print("  ║  Press  Ctrl+C  to stop                  ║")
    print("  ╚══════════════════════════════════════════╝")
    print()

    # ── Backend ────────────────────────────────────────────────
    print("  [1/2]  Starting backend  (FastAPI + uvicorn)…")
    backend = subprocess.Popen(
        [str(PYTHON), "-m", "uvicorn", "backend.app:app",
         "--port", str(BACKEND_PORT)],
        cwd=str(ROOT),
    )
    if not wait_for_port(BACKEND_PORT):
        print("\n  ✗  Backend failed to start. Check requirements.\n")
        backend.terminate()
        sys.exit(1)

    # ── Frontend ───────────────────────────────────────────────
    print("  [2/2]  Starting frontend  (static file server)…")
    frontend = subprocess.Popen(
        [str(PYTHON), "-m", "http.server", str(FRONTEND_PORT),
         "--directory", "frontend"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_port(FRONTEND_PORT, timeout=10):
        print("\n  ✗  Frontend server failed to start.\n")
        backend.terminate()
        sys.exit(1)

    # ── Open browser ───────────────────────────────────────────
    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    print()
    print(f"  ✓  App running at  http://localhost:{FRONTEND_PORT}")
    print()

    # ── Keep alive until Ctrl+C ────────────────────────────────
    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\n\n  Shutting down…")
    finally:
        backend.terminate()
        frontend.terminate()
        print("  Servers stopped.\n")


if __name__ == "__main__":
    main()

"""Serves runs/dashboard on this computer and prints its local link. Run with
uv run python -m dashboard.serve

Only listens on 127.0.0.1, nothing leaves the machine. Ctrl+C stops it.
"""

import functools
import http.server
import socket
import sys
from pathlib import Path


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        """Keeps the terminal to the one line with the link."""

    def end_headers(self):
        # every build changes the files, the browser must never reuse an old copy
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class QuietServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        """The browser drops downloads it no longer needs, that is not an error worth a traceback."""
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def free_port(start=8000):
    """First free port from `start`, so a second dashboard does not clash with the first."""
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)):
                return port
    raise RuntimeError("no free port between 8000 and 8049")


def serve(folder="runs/dashboard"):
    folder = Path(folder).resolve()
    if not (folder / "index.html").exists():
        print(f"No hay dashboard en {folder}, ejecuta make dashboard primero.")
        raise SystemExit(1)
    port = free_port()
    handler = functools.partial(QuietHandler, directory=str(folder))
    with QuietServer(("127.0.0.1", port), handler) as server:
        print(f"Dashboard en http://localhost:{port}/  (Ctrl+C para pararlo)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Dashboard parado.")


if __name__ == "__main__":
    serve(sys.argv[1] if len(sys.argv) > 1 else "runs/dashboard")

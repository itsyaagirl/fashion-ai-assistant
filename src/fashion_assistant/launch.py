"""Start the model server and give it a public HTTPS address.

In Colab (re-uses the assistant you already loaded, no second model load):

    from fashion_assistant.launch import launch
    launch(fa, frontend_url="https://finishing-touch.vercel.app")

It prints two links:
  * the Vercel front end, pre-connected to this server (…vercel.app/?api=https://….trycloudflare.com)
  * the same UI served straight from this server, in case you have no Vercel site yet

The public address comes from a Cloudflare "quick tunnel": free, no account,
and it lives as long as the Colab session. Every new session gets a new
address, so just reopen the printed link.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
_server_thread = None
_tunnel_proc = None


def _start_server(app, port: int) -> None:
    global _server_thread
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning"))
    _server_thread = threading.Thread(target=server.run, daemon=True)
    _server_thread.start()
    for _ in range(60):                      # wait until it answers
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise RuntimeError(f"Server did not start on port {port}")


def _cloudflared() -> str:
    path = Path(os.environ.get("FA_CLOUDFLARED", "/tmp/cloudflared"))
    if not path.exists():
        print("[tunnel] downloading cloudflared ...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, path)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def start_tunnel(port: int, timeout_s: int = 60) -> str:
    """Open a Cloudflare quick tunnel to localhost:port and return its https URL."""
    global _tunnel_proc
    _tunnel_proc = subprocess.Popen(
        [_cloudflared(), "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.time() + timeout_s
    for line in _tunnel_proc.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            threading.Thread(target=lambda: [None for _ in _tunnel_proc.stdout], daemon=True).start()  # keep draining
            return m.group(0)
        if time.time() > deadline:
            break
    raise RuntimeError("Could not open a Cloudflare tunnel. Re-run the cell, or use the Colab-only link below.")


def launch(assistant=None, port: int = 8000, share: bool = True, frontend_url: str | None = None) -> str | None:
    from urllib.parse import quote

    from .api import create_app

    if _server_thread is None or not _server_thread.is_alive():
        _start_server(create_app(assistant), port)
    print(f"✓ model server running on port {port}")
    if not share:
        _colab_local_link(port)
        return None
    url = start_tunnel(port)
    print(f"✓ public API: {url}\n")
    if frontend_url:
        print(f"Open the app (Vercel):  {frontend_url.rstrip('/')}/?api={quote(url, safe='')}")
    print(f"Or open it directly:     {url}")
    print("\nKeep this Colab tab open; the links stop working when the session ends.")
    return url


def _colab_local_link(port: int) -> None:
    try:
        from google.colab.output import serve_kernel_port_as_window

        serve_kernel_port_as_window(port, anchor_text="Open the app (only works in your browser)")
    except ImportError:
        print(f"Open http://localhost:{port}")

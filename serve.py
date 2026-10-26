"""Run the model server outside Colab.

    python serve.py                      # http://localhost:8000 (UI + API)
    python serve.py --share              # also print a public https link (Cloudflare quick tunnel)
    python serve.py --share --frontend https://your-app.vercel.app
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fashion_assistant.launch import launch  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--share", action="store_true")
    p.add_argument("--frontend", default=None, help="your Vercel URL, to print a pre-connected link")
    a = p.parse_args()
    launch(port=a.port, share=a.share, frontend_url=a.frontend)
    if not a.share:
        print(f"Open http://localhost:{a.port}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass

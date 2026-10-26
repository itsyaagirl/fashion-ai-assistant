"""Notebook helpers for showing products and the agent's trace."""
from __future__ import annotations

import json
from pathlib import Path


def show_items(items, title: str = "", cols: int = 5, caption_key: str = "prod_name"):
    """items: list of dicts or a DataFrame with image_path + prod_name."""
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = items.to_dict("records") if hasattr(items, "to_dict") else list(items)
    if not rows:
        print("(nothing to show)")
        return
    n = len(rows)
    r = (n + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(3 * cols, 3.6 * r))
    axes = [axes] if n == 1 else (axes.flatten() if hasattr(axes, "flatten") else axes)
    for ax in axes:
        ax.axis("off")
    for ax, it in zip(axes, rows):
        p = Path(str(it.get("image_path", "")))
        if p.exists():
            ax.imshow(Image.open(p))
        cap = str(it.get(caption_key, ""))[:28]
        if "influencer_score" in it:
            cap += f"\nscore {it['influencer_score']:.2f}"
        ax.set_title(cap, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def print_trace(result: dict) -> None:
    """Print what the agent did: each tool call and a summary of what came back."""
    from langchain_core.messages import AIMessage, ToolMessage

    for m in result.get("messages", []):
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                print(f"🔧 {tc['name']}({json.dumps(tc['args'])})")
        elif isinstance(m, ToolMessage):
            try:
                payload = json.loads(m.content)
                n = len(payload.get("results", payload.get("rules", [])))
                print(f"   ↳ {m.name}: {n} results")
            except Exception:  # noqa: BLE001
                print(f"   ↳ {m.name}")
    print(f"\nagent steps: {result.get('steps')}, candidates pooled: {len(result.get('candidates', []))}")

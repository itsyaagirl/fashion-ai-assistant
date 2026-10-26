"""Gradio demo: upload an outfit photo, get Influencer Suggestions.

Run after building the index:   python app.py      (in Colab, the notebook launches it with share=True)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402

from fashion_assistant import FashionAssistant  # noqa: E402
from fashion_assistant.ranking.personas import PERSONAS  # noqa: E402

assistant = FashionAssistant()
PERSONA_CHOICES = [k for k in PERSONAS if k != "editor"]


def run(image, request, personas):
    if image is None:
        return "Upload an outfit photo first.", []
    out = assistant.recommend(image, request=request or "", personas=personas)
    gallery = [(r["image_path"], f"{r['prod_name']} - score {r['influencer_score']:.2f}") for r in out.get("ranked", [])]
    for key, picks in out.get("persona_picks", {}).items():
        gallery += [(r["image_path"], f"{PERSONAS[key].name}: {r['prod_name']}") for r in picks[:1]]
    return out["final_answer"], gallery


with gr.Blocks(title="AI Fashion Assistant") as demo:
    gr.Markdown("# AI Fashion Assistant\nUpload an outfit photo and get accessory picks ranked by style, trend and colour fit.")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Outfit photo")
            req = gr.Textbox(label="Anything to add? (optional)", placeholder="e.g. for a summer wedding")
            per = gr.CheckboxGroup(PERSONA_CHOICES, value=["minimalist", "streetwear"], label="Influencer personas")
            btn = gr.Button("Style me", variant="primary")
        with gr.Column():
            txt = gr.Markdown()
            gal = gr.Gallery(label="Picks", columns=3, height="auto")
    btn.click(run, [img, req, per], [txt, gal])

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)

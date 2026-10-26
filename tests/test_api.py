"""The web API, end to end, with the scripted fake Gemini (no key, no downloads)."""
import io
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from conftest import fake_vec  # noqa: E402
from test_graph_fake_llm import FakeKB, ScriptedGemini  # noqa: E402


@pytest.fixture
def client(index):
    from fashion_assistant.api import create_app
    from fashion_assistant.assistant import FashionAssistant

    fa = FashionAssistant(index=index, kb=FakeKB(), llm=ScriptedGemini(), encode_image=lambda pil: fake_vec("o"))
    return TestClient(create_app(fa))


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (80, 120), "navy").save(buf, format="JPEG")
    return buf.getvalue()


def test_health_personas_trending(client):
    assert client.get("/api/health").json()["catalog_size"] == 80
    keys = [p["key"] for p in client.get("/api/personas").json()]
    assert keys == ["minimalist", "streetwear", "luxe", "boho"]
    tr = client.get("/api/trending?k=4").json()
    assert len(tr) == 4 and tr[0]["image_url"].startswith("/api/items/")


def test_recommend_json(client):
    r = client.post("/api/recommend", files={"image": ("o.jpg", _jpeg(), "image/jpeg")},
                    data={"request": "date night", "personas": "minimalist,luxe,not-a-persona"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert len(out["ranked"]) == 5 and out["ranked"][0]["image_url"].endswith("/image")
    assert set(out["persona_picks"]) == {"minimalist", "luxe"}
    assert out["persona_picks"]["luxe"]["name"] == "Lena the Luxe Stylist"
    assert [t["tool"] for t in out["trace"]].count("search_accessories") == 2
    assert all(t["results"] is not None for t in out["trace"])


def test_recommend_stream_events(client):
    with client.stream("POST", "/api/recommend/stream", files={"image": ("o.jpg", _jpeg(), "image/jpeg")},
                       data={"personas": "boho"}) as r:
        body = "".join(r.iter_text())
    events = [blk for blk in body.split("\n\n") if blk.strip()]
    kinds = [e.split("\n")[0].removeprefix("event: ") for e in events]
    assert kinds[-1] == "result" and "error" not in kinds
    nodes = [json.loads(e.split("data: ", 1)[1])["node"] for e in events if e.startswith("event: step")]
    assert nodes[0] == "analyze_outfit" and nodes[-1] == "write_up" and "tools" in nodes
    result = json.loads(events[-1].split("data: ", 1)[1])
    assert result["persona_picks"]["boho"]["picks"]
    # the slot is released afterwards: a second run works
    r2 = client.post("/api/recommend", files={"image": ("o.jpg", _jpeg(), "image/jpeg")})
    assert r2.status_code == 200


def test_bad_upload_and_missing_image(client):
    r = client.post("/api/recommend", files={"image": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400 and "isn't an image" in r.json()["detail"]
    assert client.get("/api/items/1/image").status_code == 404
    assert client.get("/api/items/100000/image").status_code == 404   # in catalog, but no photo file


def test_cors_allows_vercel(client):
    r = client.get("/api/health", headers={"Origin": "https://finishing-touch.vercel.app"})
    assert r.headers.get("access-control-allow-origin") in ("*", "https://finishing-touch.vercel.app")

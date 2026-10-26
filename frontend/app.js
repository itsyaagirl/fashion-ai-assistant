/* Finishing Touch front end: plain JS, no build step, so Vercel serves it as static files. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const STORE_KEY = "ft_api_url";
  const PERSONA_DEFAULTS = ["minimalist", "streetwear"];
  const FALLBACK_PERSONAS = [
    { key: "minimalist", name: "Mia the Minimalist", voice: "Less is more, quality basics, clean lines" },
    { key: "streetwear", name: "Jay the Streetwear Creator", voice: "Trend-aware, bold, functional" },
    { key: "luxe", name: "Lena the Luxe Stylist", voice: "Elegant, refined investment pieces" },
    { key: "boho", name: "Bea the Boho Influencer", voice: "Relaxed, earthy, festival-ready" },
  ];
  const STEP_ORDER = [
    ["analyze_outfit", "Reading the outfit"],
    ["agent", "Planning searches"],
    ["tools", "Searching the catalog"],
    ["collect_candidates", "Gathering candidates"],
    ["rank", "Ranking by style, trend and colour"],
    ["write_up", "Writing the styling notes"],
  ];
  const COLOUR_HEX = {
    black: "#1b1b1b", white: "#fafafa", cream: "#f3ead7", ivory: "#f6f1e3", beige: "#d9c6a5", camel: "#b98a55",
    brown: "#6f4a2f", tan: "#c19a6b", grey: "#9a9aa0", gray: "#9a9aa0", charcoal: "#3d3d42", navy: "#1f2a4d",
    blue: "#3c64b1", denim: "#4a6489", green: "#3f7a4a", olive: "#6b6b2f", khaki: "#9b8f5a", sage: "#9caf88",
    red: "#b3262f", burgundy: "#6d1a2c", pink: "#e8a3bd", purple: "#6b3e8c", lilac: "#c3a6d9", lavender: "#b9a7e0",
    yellow: "#e8c547", mustard: "#c9a227", orange: "#e07b2e", coral: "#ec7c6b", teal: "#2f7f7f", turquoise: "#3cb3ad",
    gold: "#c9a227", silver: "#c0c3c8", rust: "#a34c25", mint: "#a8dcc1", nude: "#e0bfa6", taupe: "#8b7d70",
  };

  let api = "";
  let personas = FALLBACK_PERSONAS;
  let file = null;
  let busy = false;

  // ------------------------------------------------------------------ server address
  function resolveApi() {
    const fromQuery = new URLSearchParams(location.search).get("api");
    if (fromQuery) {
      saveApi(fromQuery);
      history.replaceState(null, "", location.pathname);   // keep the address bar clean
    }
    let stored = "";
    try { stored = localStorage.getItem(STORE_KEY) || ""; } catch (_) { /* storage blocked */ }
    if (fromQuery) return clean(fromQuery);
    if (stored) return stored;
    if (window.FT_DEFAULT_API) return clean(window.FT_DEFAULT_API);
    // Served by the model server itself (tunnel / localhost): same origin.
    if (!/vercel\.app$/.test(location.hostname) && location.protocol.startsWith("http")) return location.origin;
    return "";
  }
  const clean = (u) => (u || "").trim().replace(/\/+$/, "");
  function saveApi(u) { try { localStorage.setItem(STORE_KEY, clean(u)); } catch (_) { /* ignore */ } }
  const url = (path) => api + path;

  async function checkServer() {
    const pill = $("serverPill"), text = $("serverText");
    if (!api) {
      pill.dataset.state = "down"; text.textContent = "Connect model server";
      return false;
    }
    pill.dataset.state = "checking"; text.textContent = "Checking model server";
    try {
      const r = await fetch(url("/api/health"), { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      const h = await r.json();
      pill.dataset.state = "ok";
      text.textContent = "Connected";
      pill.title = `Model server connected: ${h.catalog_size.toLocaleString()} accessories, ${h.model}`;
      return true;
    } catch (_) {
      pill.dataset.state = "down"; text.textContent = "Model server offline";
      return false;
    }
  }

  // ------------------------------------------------------------------ personas + trending
  async function loadPersonas() {
    try {
      const r = await fetch(url("/api/personas"));
      if (r.ok) personas = await r.json();
    } catch (_) { /* keep fallback list */ }
    $("personaGrid").innerHTML = personas.map((p) => `
      <div class="persona-opt">
        <input type="checkbox" id="p-${p.key}" value="${esc(p.key)}" ${PERSONA_DEFAULTS.includes(p.key) ? "checked" : ""}>
        <label for="p-${p.key}"><b>${esc(p.name)}</b><span>${esc(shortVoice(p.voice))}</span></label>
      </div>`).join("");
  }
  const shortVoice = (v) => { const s = (v || "").split(":").slice(-1)[0].trim(); return s.charAt(0).toUpperCase() + s.slice(1); };

  async function loadTrending() {
    try {
      const r = await fetch(url("/api/trending?k=10"));
      if (!r.ok) return;
      const items = await r.json();
      $("trendRail").innerHTML = items.map((it) => productHTML(it, {
        sub: it.trend_lift > 1.05
          ? `${esc(it.colour_group_name)} ${esc(it.product_type_name).toLowerCase()}<br><span class="lift">Selling ${it.trend_lift.toFixed(1)}× its usual rate</span>`
          : `${esc(it.colour_group_name)} ${esc(it.product_type_name).toLowerCase()}`,
        tag: null,
      })).join("");
      $("trending").hidden = items.length === 0;
    } catch (_) { /* server offline: leave hidden */ }
  }

  // ------------------------------------------------------------------ photo input
  function setFile(f) {
    if (!f) return;
    if (!/^image\/(jpeg|png|webp)$/.test(f.type)) { formMsg("Use a JPG, PNG or WebP photo.", true); return; }
    if (f.size > 10 * 1024 * 1024) { formMsg("That photo is over 10 MB. Use a smaller image.", true); return; }
    file = f;
    const img = $("outfitImg");
    img.src = URL.createObjectURL(f);
    img.hidden = false; $("plateEmpty").hidden = true; $("changePhoto").hidden = false;
    $("plate").classList.add("filled");
    formMsg("");
    updateGo();
  }
  function updateGo() { $("goBtn").disabled = !file || busy; }
  function formMsg(t, err = false) { const m = $("formMsg"); m.textContent = t; m.classList.toggle("error", err); }

  // ------------------------------------------------------------------ run the stylist
  async function run(ev) {
    ev.preventDefault();
    if (!file || busy) return;
    if (!api || !(await checkServer())) { openConnect("The model server isn't reachable. Start it from Colab, then connect."); return; }

    busy = true; updateGo();
    $("askForm").hidden = true; $("results").hidden = true; $("outfitRead").hidden = true;
    resetSteps(); $("progress").hidden = false;

    const fd = new FormData();
    fd.append("image", file);
    fd.append("request", $("requestInput").value.trim());
    fd.append("personas", [...document.querySelectorAll(".persona-opt input:checked")].map((i) => i.value).join(","));

    try {
      const result = await streamRun(fd).catch(async (err) => {
        if (err && err.fallback) return plainRun(fd);   // tunnel didn't stream: use the simple endpoint
        throw err;
      });
      showResult(result);
    } catch (err) {
      $("progress").hidden = true; $("askForm").hidden = false;
      formMsg(err.message || "Something went wrong. Try again.", true);
    } finally {
      busy = false; updateGo();
    }
  }

  async function streamRun(fd) {
    const r = await fetch(url("/api/recommend/stream"), { method: "POST", body: fd });
    if (!r.ok) throw new Error(await errorText(r));
    if (!r.body || !r.body.getReader) throw Object.assign(new Error("no stream"), { fallback: true });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "", gotEvent = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        const type = (block.match(/^event: (.+)$/m) || [])[1];
        const data = (block.match(/^data: (.+)$/m) || [])[1];
        if (!type || !data) continue;
        gotEvent = true;
        const payload = JSON.parse(data);
        if (type === "step") markStep(payload);
        else if (type === "result") return payload;
        else if (type === "error") throw new Error(payload.message);
      }
    }
    if (!gotEvent) throw Object.assign(new Error("empty stream"), { fallback: true });
    throw new Error("The connection closed before the stylist finished. Try again.");
  }

  async function plainRun(fd) {
    const r = await fetch(url("/api/recommend"), { method: "POST", body: fd });
    if (!r.ok) throw new Error(await errorText(r));
    STEP_ORDER.forEach(([n]) => markStep({ node: n, label: null, detail: "" }));
    return r.json();
  }

  async function errorText(r) {
    try { const j = await r.json(); return j.detail || `Server error ${r.status}.`; }
    catch (_) { return `Server error ${r.status}.`; }
  }

  // ------------------------------------------------------------------ progress list
  function resetSteps() {
    $("steps").innerHTML = `<li class="waiting" data-node="analyze_outfit"><b>Reading the outfit</b><span>Gemini is looking at your photo</span></li>`;
  }
  function markStep(s) {
    const ol = $("steps");
    ol.querySelectorAll("li.waiting").forEach((li) => li.remove());
    if (s.label) {
      const li = document.createElement("li");
      li.innerHTML = `<b>${esc(s.label)}</b>${s.detail ? `<span>${esc(s.detail)}</span>` : ""}`;
      ol.appendChild(li);
    }
    if (s.outfit) showOutfit(s.outfit);
    const idx = STEP_ORDER.findIndex(([n]) => n === s.node);
    let next = s.node === "agent" && /Planned/.test(s.label || "") ? "tools" : (STEP_ORDER[idx + 1] || [])[0];
    if (s.node === "tools") next = "agent";
    const label = (STEP_ORDER.find(([n]) => n === next) || [])[1];
    if (label && s.node !== "write_up") {
      const li = document.createElement("li");
      li.className = "waiting"; li.innerHTML = `<b>${esc(label)}</b>`;
      ol.appendChild(li);
    }
  }

  // ------------------------------------------------------------------ results
  function showOutfit(o) {
    $("vibe").textContent = o.vibe || "";
    $("garments").textContent = (o.garments || []).join(", ") || "—";
    $("colours").innerHTML = (o.colors || []).map((c) => `<span class="swatch"><i style="background:${swatchHex(c)}"></i>${esc(c)}</span>`).join("") || "—";
    $("style").textContent = cap(o.style || "—");
    $("occasion").textContent = cap(o.occasion || "—");
    $("outfitRead").hidden = false;
  }

  function showResult(res) {
    $("progress").hidden = true;
    if (res.outfit) showOutfit(res.outfit);
    const ranked = res.ranked || [];
    if (!ranked.length) {
      $("askForm").hidden = false;
      formMsg(res.final_answer || "No picks came back for that photo. Try a clearer outfit photo.", true);
      return;
    }
    const [top, ...rest] = ranked;
    $("heroPick").innerHTML = `
      ${productHTML(top, { titleless: true, big: true })}
      <div>
        <p class="kicker">The Influencer Suggestion</p>
        <h2>${esc(top.prod_name)}</h2>
        <p class="meta">${esc(top.colour_group_name)} ${esc(top.product_type_name).toLowerCase()}</p>
        <div class="bars">
          ${bar("Style", top.style_fit)}${bar("Trend", top.trend_fit)}${bar("Colour", top.color_fit)}
        </div>
        <p class="evidence">Score ${fmt(top.influencer_score)}: how well it suits the outfit, how fast it's selling and how the colours work together.</p>
      </div>`;
    $("alsoRail").innerHTML = rest.map((it) => productHTML(it)).join("");

    const pp = Object.entries(res.persona_picks || {});
    $("personaPicks").innerHTML = pp.length ? pp.map(([, p]) => `
      <div class="persona-card">
        <h3 class="who">${esc(p.name)}</h3>
        <p class="voice">${esc(shortVoice(p.voice))}</p>
        ${p.picks[0] ? productHTML(p.picks[0]) : "<p>No pick.</p>"}
      </div>`).join("") : `<p class="trending-note">No influencers selected. Pick some before styling to see their choices.</p>`;
    $("personaPicks").previousElementSibling.hidden = false;

    $("notes").innerHTML = markdown(res.final_answer || "");
    $("traceList").innerHTML = (res.trace || []).map((t) => `
      <li><code>${esc(t.tool)}</code> ${esc(describeArgs(t.args))}${t.results != null ? ` <span class="trace-meta">(${t.results} results)</span>` : ""}</li>`).join("")
      || "<li>The agent didn't call any tools; the server searched the suggested categories itself.</li>";
    $("traceMeta").textContent = `${res.agent_steps} agent turns, ${res.candidates} candidates ranked` + (res.seconds ? `, ${res.seconds} seconds in total.` : ".");

    const results = $("results");
    results.hidden = false;
    results.classList.remove("reveal"); void results.offsetWidth; results.classList.add("reveal");
    results.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  function productHTML(it, opts = {}) {
    const tag = opts.tag === null ? "" : `
      <div class="tag" aria-hidden="true"><span class="tag-hole"></span><span class="tag-score">${fmt(it.influencer_score)}</span><span class="tag-label">match</span></div>`;
    const sub = opts.sub || `${esc(it.colour_group_name)} ${esc(it.product_type_name).toLowerCase()}`;
    return `
      <figure class="product" style="margin:0">
        <div class="photo">
          <img src="${url(it.image_url)}" alt="${esc(it.prod_name)}" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'noimg',textContent:'Photo unavailable'}))">
          ${tag}
        </div>
        ${opts.titleless ? "" : `<figcaption><h3>${esc(it.prod_name)}</h3><p>${sub}</p></figcaption>`}
      </figure>`;
  }

  const bar = (name, v) => `<div class="bar"><span>${name}</span><span class="track"><span class="fill" style="width:${Math.round((v || 0) * 100)}%"></span></span><output>${fmt(v)}</output></div>`;
  const fmt = (v) => (typeof v === "number" ? v.toFixed(2).replace(/^0/, "") : "–");
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  function describeArgs(a) {
    if (!a) return "";
    if (a.query) return `“${a.query}”${a.category ? ` in ${a.category}` : ""}`;
    if (a.question) return `“${a.question}”`;
    if (a.category) return a.category;
    return "";
  }
  function swatchHex(name) {
    const n = name.toLowerCase();
    const key = Object.keys(COLOUR_HEX).sort((a, b) => b.length - a.length).find((k) => n.includes(k));
    return key ? COLOUR_HEX[key] : "#bbb";
  }
  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  // Tiny, safe markdown: escape first, then headings, bold, italics, lists, paragraphs.
  function markdown(md) {
    const lines = esc(md).split(/\r?\n/);
    let html = "", list = false;
    const inline = (t) => t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<em>$2</em>");
    for (const raw of lines) {
      const line = raw.trim();
      const h = line.match(/^(#{1,3})\s+(.*)/);
      const li = line.match(/^[-*]\s+(.*)|^\d+\.\s+(.*)/);
      if (li) { if (!list) { html += "<ul>"; list = true; } html += `<li>${inline(li[1] || li[2])}</li>`; continue; }
      if (list) { html += "</ul>"; list = false; }
      if (h) html += `<h3>${inline(h[2])}</h3>`;
      else if (line) html += `<p>${inline(line)}</p>`;
    }
    return html + (list ? "</ul>" : "");
  }

  // ------------------------------------------------------------------ connect dialog
  function openConnect(msg = "") {
    $("apiInput").value = api;
    $("connectMsg").textContent = msg;
    $("connectMsg").classList.toggle("error", !!msg);
    $("connectDialog").showModal();
  }
  async function connect() {
    const v = clean($("apiInput").value);
    if (!/^https?:\/\//.test(v)) { $("connectMsg").textContent = "Enter the full address, starting with https://"; $("connectMsg").classList.add("error"); return; }
    api = v;
    $("connectMsg").classList.remove("error"); $("connectMsg").textContent = "Connecting…";
    if (await checkServer()) {
      saveApi(v);
      $("connectDialog").close();
      loadPersonas(); loadTrending();
    } else {
      $("connectMsg").textContent = "No model server answered at that address. Check the Colab cell is still running.";
      $("connectMsg").classList.add("error");
    }
  }

  // ------------------------------------------------------------------ wire up
  function init() {
    api = resolveApi();
    const plate = $("plate");
    $("fileInput").addEventListener("change", (e) => setFile(e.target.files[0]));
    ["dragenter", "dragover"].forEach((t) => plate.addEventListener(t, (e) => { e.preventDefault(); plate.classList.add("dragging"); }));
    ["dragleave", "drop"].forEach((t) => plate.addEventListener(t, (e) => { e.preventDefault(); plate.classList.remove("dragging"); }));
    plate.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
    plate.addEventListener("click", (e) => { if (plate.classList.contains("filled")) e.preventDefault(); });
    $("changePhoto").addEventListener("click", () => $("fileInput").click());
    $("askForm").addEventListener("submit", run);
    $("againBtn").addEventListener("click", () => {
      $("results").hidden = true; $("outfitRead").hidden = true; $("askForm").hidden = false;
      window.scrollTo({ top: 0 });
    });
    $("serverPill").addEventListener("click", () => openConnect());
    $("connectBtn").addEventListener("click", connect);
    $("apiInput").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); connect(); } });

    loadPersonas();   // shows the fallback list immediately, real list once the server answers
    checkServer().then((ok) => {
      if (ok) { loadPersonas(); loadTrending(); }
      else if (!api) openConnect();
    });
  }
  document.addEventListener("DOMContentLoaded", init);
})();

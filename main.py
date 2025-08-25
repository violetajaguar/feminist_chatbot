
from __future__ import annotations

import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator

# optional .env loading (safe if file not present)
try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except Exception:
    pass

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Feminist Chatbot logging initialized.")

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"
)

# Shared HTTP client (no HTTP/2 -> no 'h2' dependency; retries via transport)
HTTP_CLIENT = httpx.Client(
    timeout=30,
    follow_redirects=True,
    trust_env=False,
    transport=httpx.HTTPTransport(retries=3),
)

# OpenAI client (using REST via httpx to avoid extra deps)
OAI_BASE = "https://api.openai.com/v1/chat/completions"
OAI_HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
DS_HEADERS = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

# ---------- FastAPI ----------
app = FastAPI(title="FeministBot · Peach × Dragon", version="1.2")


# ---------- Utilities ----------
def _suffix(s: str) -> str:
    return s[-4:] if s else ""

def detect_lang(text: str) -> Literal["en", "es"]:
    """
    Tiny heuristic: if many common Spanish tokens/accents appear, pick 'es'.
    Otherwise default to 'en'.
    """
    text_low = text.lower()
    spanish_hits = 0
    spanish_hits += len(re.findall(r"\b(qué|por qué|porque|cómo|dónde|cuándo|quién|gracias|hola|buen[oa]s|sí|no|loc[ao]|feminismo|artista|cultura)\b", text_low))
    spanish_hits += bool(re.search(r"[áéíóúñü]", text_low)) * 2
    return "es" if spanish_hits >= 1 else "en"

def persona_system_prompt_peach(lang: Literal["en", "es"]) -> str:
    if lang == "es":
        return (
            "Eres PEACH — Punk Riot Grrrl: directa, combativa, juguetona, con corazón grande. "
            "Responde en español, en 80–140 palabras, con energía DIY y cuidado. "
            "Sé clara, no agresiva; humor con ternura; cero insultos."
        )
    return (
        "You are PEACH — Punk Riot Grrrl: direct, fiery, playful, big-hearted. "
        "Reply in English, 80–140 words, DIY energy with care. "
        "Be clear, not cruel; witty with tenderness; no insults."
    )

def persona_system_prompt_dragon(lang: Literal["en", "es"]) -> str:
    if lang == "es":
        return (
            "Eres DRAGON — Trickster filosóficx: irónico, curioso, suave, metafórico. "
            "Responde en español, 80–140 palabras; cuestiona supuestos, construye puentes; "
            "sé amable y lúcido."
        )
    return (
        "You are DRAGON — Philosophical Trickster: ironic, curious, gentle, metaphor-loving. "
        "Reply in English, 80–140 words; interrogate assumptions and build bridges; "
        "be kind and lucid."
    )

def safe_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = HTTP_CLIENT.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        logging.error("HTTP error %s -> %s", url, e.response.text[:300])
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logging.exception("Request failed")
        raise HTTPException(status_code=502, detail=f"Upstream error: {type(e).__name__}: {e}")

def openai_chat(messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
    data = {"model": model or OPENAI_MODEL, "messages": messages}
    j = safe_post_json(OAI_BASE, OAI_HEADERS, data)
    return j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

def deepseek_chat(messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
    data = {"model": model or DEEPSEEK_MODEL, "messages": messages}
    j = safe_post_json(DEEPSEEK_API_URL, DS_HEADERS, data)
    return j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


# ---------- Schemas ----------
class ChatMsg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatIn(BaseModel):
    messages: List[ChatMsg]
    persona: Literal["both", "peach", "dragon"] = "both"
    lang: Literal["auto", "en", "es"] = "auto"
    model_peach: Optional[str] = None
    model_dragon: Optional[str] = None

class ChatOut(BaseModel):
    timestamp: str
    peach: Optional[str] = None
    dragon: Optional[str] = None

class DebateIn(BaseModel):
    prompt: str = Field(..., min_length=3)
    rounds: int = Field(2, ge=1, le=5)
    persona_peach: str = "Punk Riot Grrrl"
    persona_dragon: str = "Philosophical Trickster"
    lang: Literal["auto", "en", "es"] = "auto"
    model_peach: Optional[str] = None
    model_dragon: Optional[str] = None

class DebateTurn(BaseModel):
    peach: Optional[str] = None
    dragon: Optional[str] = None

class DebateOut(BaseModel):
    timestamp: str
    topic: str
    turns: List[DebateTurn]


# ---------- Routes ----------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "time": datetime.now(timezone.utc).isoformat(),
        "openai_key_suffix": _suffix(OPENAI_API_KEY),
        "deepseek_key_suffix": _suffix(DEEPSEEK_API_KEY),
    }


@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn) -> ChatOut:
    # decide language
    last_user = next((m.content for m in reversed(payload.messages) if m.role == "user"), "")
    lang = detect_lang(last_user) if payload.lang == "auto" else payload.lang

    # system seeds
    peach_sys = {"role": "system", "content": persona_system_prompt_peach(lang)}
    dragon_sys = {"role": "system", "content": persona_system_prompt_dragon(lang)}

    # Prepare message lists
    base_msgs = [{"role": m.role, "content": m.content} for m in payload.messages]

    peach_ans = dragon_ans = None

    if payload.persona in ("both", "peach"):
        msgs = [peach_sys] + base_msgs
        peach_ans = openai_chat(msgs, model=payload.model_peach)

    if payload.persona in ("both", "dragon"):
        msgs = [dragon_sys] + base_msgs
        dragon_ans = deepseek_chat(msgs, model=payload.model_dragon)

    return ChatOut(
        timestamp=datetime.now(timezone.utc).isoformat(),
        peach=peach_ans,
        dragon=dragon_ans,
    )


@app.post("/debate", response_model=DebateOut)
def debate(payload: DebateIn) -> DebateOut:
    lang = payload.lang
    if lang == "auto":
        lang = detect_lang(payload.prompt)

    # Seed system messages
    peach_sys = {"role": "system", "content": persona_system_prompt_peach(lang)}
    dragon_sys = {"role": "system", "content": persona_system_prompt_dragon(lang)}

    turns: List[DebateTurn] = []
    history: List[Dict[str, str]] = [
        {"role": "user", "content": f"Topic: {payload.prompt}\nKeep replies succinct."}
    ]

    for r in range(payload.rounds):
        # Peach goes
        peach_msgs = [peach_sys] + history + [
            {
                "role": "user",
                "content": (
                    f"Round {r+1}: Respond as {payload.persona_peach}. "
                    "Make 1–2 sharp points, max ~120 words. Be vivid; cite no URLs."
                )
            }
        ]
        peach_ans = openai_chat(peach_msgs, model=payload.model_peach)

        history.append({"role": "assistant", "content": f"PEACH:\n{peach_ans}"})

        # Dragon replies
        dragon_msgs = [dragon_sys] + history + [
            {
                "role": "user",
                "content": (
                    f"Round {r+1}: Reply as {payload.persona_dragon}. "
                    "Engage with PEACH's points, add nuance, offer a counter or synthesis. "
                    "Max ~120 words."
                )
            }
        ]
        dragon_ans = deepseek_chat(dragon_msgs, model=payload.model_dragon)

        history.append({"role": "assistant", "content": f"DRAGON:\n{dragon_ans}"})
        turns.append(DebateTurn(peach=peach_ans, dragon=dragon_ans))

    return DebateOut(
        timestamp=datetime.now(timezone.utc).isoformat(),
        topic=payload.prompt,
        turns=turns,
    )


# ---------- HTML ----------
HTML_INDEX = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="theme-color" content="#0b0d10" />
<title>FeministBot · Peach × Dragon</title>
<style>
  :root{
    --bg:#0b0d10; --panel:#11141a; --panel-2:#0f1016;
    --text:#e8ebef; --muted:#b9bfca; --pill:#1c1f27;
    --accent-1:#ff2e92; /* dragon */ --accent-2:#ff9d6c; /* peach */
    --ring:#7b2cff; --good:#25d366; --warning:#ff9500;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
    font:16px system-ui,-apple-system,Segoe UI,Inter,Roboto,Arial,sans-serif}
  a{color:inherit}
  .wrap{max-width:1100px;margin:0 auto;padding:14px;display:grid;gap:16px}
  .hero{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
  .title{font-weight:800;letter-spacing:.2px}
  .badge{padding:6px 10px;border-radius:999px;background:linear-gradient(90deg,#ff9d6c,#ff2e92);font-weight:700}
  .blurb{
    background:linear-gradient(180deg,#11141a,#0f1016); border:1px solid #1b1f28;
    padding:12px 14px;border-radius:14px;color:#cfd6e2; line-height:1.35
  }
  .blurb b{color:#fff}

  .grid{display:grid;gap:16px}
  @media (min-width:900px){ .grid{grid-template-columns:1.1fr 1fr} }

  .fruit-grid{display:grid;gap:16px}
  @media (min-width:700px){ .fruit-grid{grid-template-columns:1fr 1fr} }
  .card{
    background:linear-gradient(180deg,var(--panel),var(--panel-2));
    border:1px solid #1b1f28;border-radius:22px;padding:16px;
    box-shadow:0 0 0 1px #0b0d12, 0 16px 40px rgba(0,0,0,.45), inset 0 0 0 1px rgba(255,255,255,.03);
    transition:transform .2s ease
  }
  .card:hover{transform:translateY(-2px)}
  .fruit-card{display:flex;align-items:center;justify-content:center;height:260px;position:relative;cursor:pointer}
  .fruit-card svg{width:clamp(120px,34vw,180px);height:auto;image-rendering:pixelated;shape-rendering:crispEdges;transition:transform .3s ease}
  .fruit-card:hover svg{transform:scale(1.05)}
  .fruit-card.dragon svg{transform:scale(.86)} .fruit-card.dragon:hover svg{transform:scale(.9)}
  .fruit-card.peach{ box-shadow:0 0 60px rgba(255,157,108,.12), inset 0 0 0 1px rgba(255,157,108,.1) }
  .fruit-card.dragon{ box-shadow:0 0 60px rgba(255,46,146,.14), inset 0 0 0 1px rgba(255,46,146,.1) }
  .char-info{position:absolute;bottom:12px;left:12px;right:12px;background:rgba(0,0,0,.7);backdrop-filter:blur(8px);border-radius:12px;padding:8px 10px;font-size:13px;opacity:0;transform:translateY(8px);transition:all .25s ease}
  .fruit-card:hover .char-info{opacity:1;transform:translateY(0)}
  @keyframes chomp{0%,20%,100%{opacity:0}30%,40%{opacity:1}50%,90%{opacity:0}}
  .peach-chomp{animation:chomp 3.5s infinite step-end}
  @keyframes blink{0%,92%,100%{transform:scaleY(1)}93%,99%{transform:scaleY(.1)}}
  .eye{transform-origin:50% 50%; animation:blink 4.5s infinite}
  .panel{background:var(--panel);border:1px solid #1b1f28;border-radius:18px;padding:14px}
  .panel-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
  .h{margin:0;font-weight:800}
  .status{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:#b9bfca}
  .status-dot{width:6px;height:6px;border-radius:50%;background:var(--good)}
  .status-dot.typing{background:var(--warning);animation:pulse 1.5s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  .tags{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
  .tag{padding:6px 12px;border-radius:999px;border:1px solid #242936;background:linear-gradient(90deg,#121620,#0f1218);color:#bfc6d2;font-size:.86rem;font-weight:600;cursor:pointer}
  .tag.active{background:linear-gradient(90deg,var(--accent-2),var(--accent-1));color:#fff;border-color:transparent}
  .input-group{margin-bottom:10px}
  .input-group label{display:block;font-size:13px;color:#b9bfca;margin-bottom:6px}
  textarea,input{width:100%;background:#0e1116;border:2px solid #1b1f28;color:#e8ebef;border-radius:12px;padding:12px 14px;outline:none;font-size:16px}
  textarea:focus,input:focus{border-color:var(--accent-2);box-shadow:0 0 0 3px rgba(255,157,108,.1)}
  textarea{min-height:96px;resize:vertical}
  .char-count{font-size:12px;color:#b9bfca;text-align:right;margin-top:4px}
  .btn{display:inline-flex;align-items:center;gap:8px;padding:12px 16px;border-radius:14px;border:1px solid #1b1f28;background:linear-gradient(180deg,#1a1f27,#13151b);color:#fff;font-weight:800;cursor:pointer;min-height:44px}
  .btn--peach{background:linear-gradient(90deg,#ff9d6c,#ff7181)}
  .btn--go{background:linear-gradient(90deg,#25d366,#ff9d6c)}
  .spinner{width:16px;height:16px;border:2px solid rgba(255,255,255,.3);border-top:2px solid currentColor;border-radius:50%;animation:spin 1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .cols{display:grid;gap:14px} @media(min-width:900px){.cols{grid-template-columns:1fr 1fr}}
  .log{display:grid;gap:12px;max-height:380px;overflow:auto;background:#0a0c0f;border-radius:14px;padding:12px;scrollbar-width:thin;scrollbar-color:#1c1f27 transparent}
  .bubble{background:linear-gradient(180deg,#0e1116,#0a0d12);border:1px solid #1b1f28;border-radius:14px;padding:12px 14px}
  .bubble.peach{border-left:3px solid var(--accent-2)} .bubble.dragon{border-left:3px solid var(--accent-1)} .bubble.system{border-left:3px solid #ff9500}
  .bubble-header{display:flex;align-items:center;gap:8px;margin-bottom:6px}
  .bubble-avatar{width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px}
  .bubble-avatar.peach{background:var(--accent-2)} .bubble-avatar.dragon{background:var(--accent-1)}
  .mini-footer{margin-top:6px;font-size:12px;color:#b9bfca}
</style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="title">
        <div style="font-size:28px">FeministBot · <span style="color:#ff9d6c">Peach</span> × <span style="color:#ff2e92">Dragon</span></div>
        <div class="mini-footer">Two voices. Two engines. One conversation.</div>
      </div>
      <div class="badge">Peach = ChatGPT · Dragon = DeepSeek</div>
    </div>

    <div class="blurb">
      <b>About bias :</b> Peach runs on OpenAI's ChatGPT, while Dragon is built on DeepSeek, raised by different digital families with distinct worldviews. When they clash over what seems "self-evident," something happens: they actually listen to each other.

    <!-- FRUITS -->
    <section class="fruit-grid">
      <div class="card fruit-card peach" aria-label="Peach" tabindex="0" role="button" title="Talk to Peach (ChatGPT)">
        <svg viewBox="0 0 16 16" role="img" aria-label="Pixel Peach">
          <rect x="7" y="1" width="2" height="1" fill="#7a4a2c"/>
          <rect x="6" y="0" width="4" height="1" fill="#34d17a"/>
          <rect x="3" y="3" width="10" height="10" fill="#964d37"/>
          <rect x="4" y="4" width="8" height="8" fill="#f89c79"/>
          <rect x="5" y="5" width="6" height="6" fill="#ffb497"/>
          <rect class="peach-chomp" x="11" y="7" width="2" height="2" fill="#0b0d10"/>
          <rect x="7" y="12" width="2" height="2" fill="#c57658"/>
        </svg>
        <div class="char-info"><b>🌸 Peach</b> — Punk Riot Grrrl · powered by ChatGPT</div>
      </div>

      <div class="card fruit-card dragon" aria-label="Dragon" tabindex="0" role="button" title="Talk to Dragon (DeepSeek)">
        <svg viewBox="0 0 16 16" role="img" aria-label="Pixel Dragon Fruit">
          <rect x="2" y="1" width="1" height="2" fill="#b2ff2e"/>
          <rect x="12" y="1" width="1" height="2" fill="#b2ff2e"/>
          <rect x="1" y="6" width="1" height="2" fill="#b2ff2e"/>
          <rect x="14" y="6" width="1" height="2" fill="#b2ff2e"/>
          <rect x="2" y="12" width="1" height="2" fill="#b2ff2e"/>
          <rect x="12" y="12" width="1" height="2" fill="#b2ff2e"/>
          <rect x="3" y="3" width="10" height="10" fill="#ff1b8d"/>
          <rect x="4" y="4" width="8" height="8" fill="#ffffff"/>
          <g class="eye">
            <rect x="6" y="7" width="2" height="2" fill="#0b0d10"/>
            <rect x="9" y="7" width="2" height="2" fill="#0b0d10"/>
          </g>
          <rect x="7" y="6" width="1" height="1" fill="#0b0d10" opacity=".6"/>
          <rect x="8" y="10" width="1" height="1" fill="#0b0d10" opacity=".6"/>
          <rect x="5" y="9" width="1" height="1" fill="#0b0d10" opacity=".5"/>
          <rect x="10" y="5" width="1" height="1" fill="#0b0d10" opacity=".5"/>
        </svg>
        <div class="char-info"><b>🐉 Dragon</b> — Philosophical Trickster · powered by DeepSeek</div>
      </div>
    </section>

    <!-- CONTROLS + LOGS -->
    <section class="grid">
      <div class="cols">
        <!-- CHAT -->
        <div class="panel">
          <div class="panel-header">
            <h2 class="h">Chat</h2>
            <div class="status"><div class="status-dot" id="chatStatus"></div><span id="chatStatusText">Ready</span></div>
          </div>

          <div class="tags">
            <span class="tag active" data-persona="both">Both Characters</span>
            <span class="tag" data-persona="peach">🌸 Peach Only</span>
            <span class="tag" data-persona="dragon">🐉 Dragon Only</span>
          </div>

          <div class="input-group">
            <label for="chatInput">Your message</label>
            <textarea id="chatInput" placeholder="Ask in English or Español…" maxlength="500"></textarea>
            <div class="char-count"><span id="chatCount">0</span>/500</div>
          </div>

          <div style="display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap">
            <div style="display:flex;gap:8px;flex-wrap:wrap">
              <button class="tag" data-quick="Hola, ¿pueden debatir sobre arte y tecnología?">ES start</button>
              <button class="tag" data-quick="Hello! What’s your take on feminist AI?">EN start</button>
              <button class="tag" data-quick="¿Cómo podemos combatir el sesgo en los modelos?">Bias</button>
            </div>
            <button class="btn btn--peach" id="sendChat"><span class="btn-text">Send /chat</span><div class="spinner" id="chatSpinner" style="display:none"></div></button>
          </div>
        </div>

        <div class="panel">
          <div class="log" id="chatLog" aria-live="polite">
            <div style="opacity:.7">Start a conversation with Peach & Dragon…</div>
          </div>
        </div>
      </div>

      <!-- DEBATE -->
      <div class="cols">
        <div class="panel">
          <div class="panel-header">
            <h2 class="h">Debate</h2>
            <div class="status"><div class="status-dot" id="debateStatus"></div><span id="debateStatusText">Ready</span></div>
          </div>

          <div class="input-group">
            <label for="debTopic">Debate topic</label>
            <input id="debTopic" placeholder="e.g., Is generative AI good for artists?" maxlength="200" />
          </div>

          <div style="display:flex;gap:10px;justify-content:flex-end">
            <button class="btn" id="clearDebate">Clear</button>
            <button class="btn btn--go" id="startDebate"><span class="btn-text">Start /debate</span><div class="spinner" id="debateSpinner" style="display:none"></div></button>
          </div>
        </div>

        <div class="panel">
          <div class="log" id="debLog">
            <div style="opacity:.7">Set a topic and watch them argue (with love).</div>
          </div>
        </div>
      </div>
    </section>

    <div class="mini-footer">Built for conversation. If one voice feels different from the other, good—you’re noticing <i>bias</i>. Let’s use that.</div>
  </div>

<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

let currentPersona = 'both';

const chatLog = $('#chatLog');
const debLog = $('#debLog');
const chatInput = $('#chatInput');
const debTopic = $('#debTopic');
const chatCount = $('#chatCount');

init();

function init(){
  $('#sendChat').onclick = handleSendChat;
  $('#startDebate').onclick = handleStartDebate;
  $('#clearDebate').onclick = () => debLog.innerHTML = '<div style="opacity:.7">Cleared.</div>';
  chatInput.oninput = () => chatCount.textContent = chatInput.value.length;
  $$('.tag[data-persona]').forEach(t => t.onclick = () => selectPersona(t.dataset.persona));
  $$('[data-quick]').forEach(b => b.onclick = () => { chatInput.value=b.dataset.quick; chatCount.textContent = chatInput.value.length; chatInput.focus(); });
  $('.fruit-card.peach').onclick  = () => selectPersona('peach');
  $('.fruit-card.dragon').onclick = () => selectPersona('dragon');
}

function selectPersona(p){
  currentPersona = p;
  $$('.tag[data-persona]').forEach(t => t.classList.remove('active'));
  $(`[data-persona="${p}"]`)?.classList.add('active');
}

function setLoading(kind, on){
  const btn = kind==='chat' ? $('#sendChat') : $('#startDebate');
  const sp  = kind==='chat' ? $('#chatSpinner') : $('#debateSpinner');
  const dot = kind==='chat' ? $('#chatStatus') : $('#debateStatus');
  const txt = kind==='chat' ? $('#chatStatusText') : $('#debateStatusText');
  btn.disabled = on; sp.style.display = on?'inline-block':'none';
  dot.classList.toggle('typing', on); txt.textContent = on?'Thinking…':'Ready';
}

function addChatBubble(type, text, name){
  const el = document.createElement('div');
  el.className = `bubble ${type}`;
  const t = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  const avatar = type==='peach'?'🌸':type==='dragon'?'🐉':type==='user'?'👤':'⚙️';
  el.innerHTML = `<div class="bubble-header"><div class="bubble-avatar ${type}">${avatar}</div><div style="font-weight:700">${name}</div><div style="margin-left:auto;color:#b9bfca;font-size:12px">${t}</div></div><div>${escapeHTML(text)}</div>`;
  chatLog.prepend(el);
}

function addDebateBubble(type, text, name){
  const el = document.createElement('div');
  el.className = `bubble ${type}`;
  const t = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  const avatar = type==='peach'?'🌸':type==='dragon'?'🐉':'🎯';
  el.innerHTML = `<div class="bubble-header"><div class="bubble-avatar ${type}">${avatar}</div><div style="font-weight:700">${name}</div><div style="margin-left:auto;color:#b9bfca;font-size:12px">${t}</div></div><div>${escapeHTML(text)}</div>`;
  debLog.appendChild(el);
  debLog.scrollTop = debLog.scrollHeight;
}

async function handleSendChat(){
  const msg = chatInput.value.trim();
  if (!msg) return;
  addChatBubble('user', msg, 'You');

  setLoading('chat', true);
  try{
    const res = await fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        messages: [{role:'user',content:msg}],
        persona: currentPersona,
        lang: 'auto'
      })
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (data.peach)  addChatBubble('peach',  data.peach,  'Peach');
    if (data.dragon) addChatBubble('dragon', data.dragon, 'Dragon');
  }catch(e){
    addChatBubble('system', 'Server error. Try again.', 'System');
    console.error(e);
  }finally{
    setLoading('chat', false);
  }
}

async function handleStartDebate(){
  const topic = debTopic.value.trim();
  if (!topic) return;
  debLog.innerHTML = '';
  addDebateBubble('system', `🎯 Debate Topic: ${topic}`, 'Topic');

  setLoading('debate', true);
  try{
    const res = await fetch('/debate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        prompt: topic, rounds: 2, lang: 'auto',
        persona_peach: 'Punk Riot Grrrl',
        persona_dragon: 'Philosophical Trickster'
      })
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    (data.turns||[]).forEach(turn=>{
      if (turn.peach)  addDebateBubble('peach',  turn.peach,  'Peach');
      if (turn.dragon) addDebateBubble('dragon', turn.dragon, 'Dragon');
    });
  }catch(e){
    addDebateBubble('system','Server error. Try again.','System');
    console.error(e);
  }finally{
    setLoading('debate', false);
  }
}

function escapeHTML(s){
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML_INDEX)



from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional

import httpx
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- env & logging -----------------------------------------------------------
load_dotenv(".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Feminist Chatbot logging initialized.")
logging.info("[DEBUG] OPENAI …%s", OPENAI_API_KEY[-4:] if OPENAI_API_KEY else "MISSING")
logging.info("[DEBUG] DEEPSEEK …%s", DEEPSEEK_API_KEY[-4:] if DEEPSEEK_API_KEY else "MISSING")

# --- HTTP clients ------------------------------------------------------------
HTTP_CLIENT = httpx.Client(timeout=30, trust_env=False, transport=httpx.HTTPTransport(retries=3))

# --- Models ------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    # optional per-request override
    model_peach: Optional[str] = None
    model_dragon: Optional[str] = None
    temperature: float = 0.7

class DebateRequest(BaseModel):
    prompt: str
    rounds: int = 2
    persona_peach: str = "Punk Riot Grrrl"
    persona_dragon: str = "Philosophical Trickster"
    model_peach: Optional[str] = None
    model_dragon: Optional[str] = None
    temperature: float = 0.8

# --- Tiny persona prompts ----------------------------------------------------
PEACH_STYLE = """You are PEACH — a punk riot grrrl feminist voice: direct, loud, witty, DIY energy.
Keep replies punchy and musical. Swear lightly if it serves the point (but stay respectful)."""

DRAGON_STYLE = """You are DRAGON — a philosophical trickster feminist voice: clever, ironic, playful,
bringing receipts and questions that twist assumptions. Write with glittering precision."""

# --- LLM helpers -------------------------------------------------------------
def openai_chat(messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> str:
    """
    Calls OpenAI (python SDK v1.x) using raw HTTPX client for reliability.
    """
    if not OPENAI_API_KEY:
        return "Peach error: OPENAI_API_KEY missing on server."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "temperature": temperature}

    try:
        r = HTTP_CLIENT.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Peach error: {type(e).__name__}: {e}"

def deepseek_chat(messages: List[Dict[str, str]], model: str = "deepseek-chat", temperature: float = 0.7) -> str:
    """
    Calls DeepSeek via requests (simple & sturdy).
    """
    if not DEEPSEEK_API_KEY:
        return "Dragon error: DEEPSEEK_API_KEY missing on server."

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Dragon error: {type(e).__name__}: {e}"

# --- Debate engine -----------------------------------------------------------
def run_debate(prompt: str, rounds: int, persona_peach: str, persona_dragon: str,
               model_peach: str, model_dragon: str, temperature: float) -> Dict[str, Any]:
    peach_sys = f"{PEACH_STYLE}\nPersona: {persona_peach}."
    dragon_sys = f"{DRAGON_STYLE}\nPersona: {persona_dragon}."

    p_msgs = [{"role": "system", "content": peach_sys}]
    d_msgs = [{"role": "system", "content": dragon_sys}]

    # Opening statements
    p_msgs.append({"role": "user", "content": f"Topic: {prompt}\nGive your opening statement."})
    d_msgs.append({"role": "user", "content": f"Topic: {prompt}\nGive your opening statement."})

    peach_text = openai_chat(p_msgs, model=model_peach, temperature=temperature)
    dragon_text = deepseek_chat(d_msgs, model=model_dragon, temperature=temperature)

    transcript = [{"peach": peach_text, "dragon": dragon_text}]

    # Crossfire rounds
    for _ in range(max(0, rounds - 1)):
        p_msgs.append({"role": "user", "content": f"Dragon said:\n{dragon_text}\nCounter this succinctly."})
        d_msgs.append({"role": "user", "content": f"Peach said:\n{peach_text}\nCounter this succinctly."})
        peach_text = openai_chat(p_msgs, model=model_peach, temperature=temperature)
        dragon_text = deepseek_chat(d_msgs, model=model_dragon, temperature=temperature)
        transcript.append({"peach": peach_text, "dragon": dragon_text})

    # Closing
    p_msgs.append({"role": "user", "content": "Now deliver a 1-sentence closing line."})
    d_msgs.append({"role": "user", "content": "Now deliver a 1-sentence closing line."})
    closing_peach = openai_chat(p_msgs, model=model_peach, temperature=temperature)
    closing_dragon = deepseek_chat(d_msgs, model=model_dragon, temperature=temperature)

    return {"topic": prompt, "rounds": rounds, "transcript": transcript,
            "closing": {"peach": closing_peach, "dragon": closing_dragon}}

# --- FastAPI -----------------------------------------------------------------
app = FastAPI(title="FeministBot", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Pixel neon index --------------------------------------------------------
INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FeministBot — Peach × Dragon</title>
<style>
  :root{
    --bg:#0a0b10;
    --ink:#efe9ff;
    --muted:#9aa0b3;
    --peach:#ff9e7a;
    --peach-deep:#ff7a59;
    --leaf:#37e084;
    --magenta:#ff2ea6;
    --dragon:#e91e63;
    --lime:#a6ff3a;
    --grid: rgba(255,255,255,.06);
    --glow: 0 0 .6rem rgba(255,255,255,.2), 0 0 2.2rem rgba(255,0,150,.18);
  }
  html,body{height:100%;background:radial-gradient(1200px 800px at 30% -10%,#25103a 0%, #0a0b10 55%) no-repeat var(--bg); color:var(--ink); margin:0; font:500 16px/1.35 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial;}
  .wrap{max-width:1000px; margin:0 auto; padding:48px 20px 80px; position:relative;}
  .maze{
    position:absolute; inset:0; pointer-events:none; opacity:.25; mask-image:radial-gradient(1000px 700px at 30% 0%, #000 35%, transparent 70%);
    background:
      linear-gradient(transparent 23px, var(--grid) 24px) 0 0 / 24px 24px,
      linear-gradient(90deg, transparent 23px, var(--grid) 24px) 0 0 / 24px 24px;
  }
  h1{
    font-size: clamp(28px, 4vw, 44px);
    letter-spacing:.02em;
    margin:0 0 10px;
    text-shadow: 0 0 .5rem #f0f, 0 0 1rem #0ff;
  }
  .tag{color:var(--muted); margin-bottom:28px}
  .grid{display:grid; grid-template-columns: 1fr 1fr; gap:22px; align-items:center; margin:30px 0 26px}
  .card{
    background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
    border:1px solid rgba(255,255,255,.08);
    border-radius:16px; padding:18px; box-shadow: var(--glow);
  }
  canvas{ image-rendering: pixelated; width:100%; aspect-ratio:1/1; background:#0d0f18; border-radius:12px; border:1px solid rgba(255,255,255,.06); }
  .controls{display:grid; grid-template-columns: 1fr 1fr; gap:18px; margin-top:20px}
  textarea, input, button{
    width:100%; border-radius:12px; border:1px solid rgba(255,255,255,.12);
    background:#0e111a; color:var(--ink); padding:12px 14px; font:inherit;
  }
  textarea{min-height:92px; resize:vertical}
  button{cursor:pointer; background: linear-gradient(90deg, var(--dragon), var(--magenta)); border:none; font-weight:700; letter-spacing:.02em}
  button.secondary{background: linear-gradient(90deg, var(--leaf), var(--peach)); color:#111}
  .out{white-space:pre-wrap; font-size:15px; line-height:1.45; padding:12px 14px; background:#0c0f17; border-radius:12px; border:1px solid rgba(255,255,255,.08); min-height:80px}
  .footer{margin-top:26px; color:var(--muted); font-size:14px}
  @media (max-width:820px){ .grid{grid-template-columns:1fr} }
</style>
</head>
<body>
<div class="wrap">
  <div class="maze"></div>
  <h1>FeministBot <span style="opacity:.85">·</span> <strong>PEACH × DRAGON</strong></h1>
  <div class="tag">Two feminist voices in neon-arcade pixel style. Chat or make them debate.</div>

  <div class="grid">
    <div class="card">
      <canvas id="peach" width="160" height="160" aria-label="Pixel Peach"></canvas>
    </div>
    <div class="card">
      <canvas id="dragon" width="160" height="160" aria-label="Pixel Dragon Fruit"></canvas>
    </div>
  </div>

  <div class="controls">
    <div class="card">
      <h3 style="margin:0 0 10px">Chat</h3>
      <textarea id="chatText" placeholder="say hi to Peach & Dragon…"></textarea>
      <button id="chatBtn">Send /chat</button>
      <div id="chatOut" class="out" style="margin-top:12px"></div>
    </div>

    <div class="card">
      <h3 style="margin:0 0 10px">Debate</h3>
      <input id="debateTopic" placeholder="topic (e.g., Is generative AI good for artists?)"/>
      <button id="debateBtn" class="secondary">Start /debate</button>
      <div id="debateOut" class="out" style="margin-top:12px"></div>
    </div>
  </div>

  <div class="footer" id="health">checking /health…</div>
</div>

<script>
/* --------- pixel art (draw squares on a grid) --------------------------- */
function drawPixelArt(canvasId, palette, map, scale=10){
  const cvs = document.getElementById(canvasId);
  const ctx = cvs.getContext("2d");
  const rows = map.length;
  const cols = map[0].length;
  const px = Math.floor(Math.min(cvs.width/cols, cvs.height/rows));
  ctx.clearRect(0,0,cvs.width,cvs.height);
  for(let y=0;y<rows;y++){
    for(let x=0;x<cols;x++){
      const k = map[y][x];
      if(k===' ') continue;
      ctx.fillStyle = palette[k] || "#000";
      ctx.fillRect(x*px, y*px, px, px);
    }
  }
}

/* Peach (16x16) — chunky pixels, leaf + highlight */
const peachPalette = {
  'o': '#803a2b', /* outline */
  'p': '#ff8f6a',
  'P': '#ffad86',
  'h': '#ffe0c9',
  'g': '#3de087',
  'd': '#2ba862'
};
const peachMap = [
"                ",
"      ggg       ",
"     gddg       ",
"      gg        ",
"     ooooo      ",
"   oPPPPPoo     ",
"  oPPPPPPPo     ",
" oPPPPppPPPo    ",
" oPPPppppPPo    ",
" oPPPppppPPo    ",
"  oPPPppPPo     ",
"  ooPPPPPoo     ",
"    ooooo       ",
"     hhh        ",
"                ",
"                ",
];

/* Dragon fruit (16x16) — magenta skin, white flesh, black seeds, lime spikes */
const dragonPalette = {
  'm':'#ff2ea6', /* skin */
  'M':'#d4187f',
  'l':'#b4ff42', /* spikes */
  'w':'#ffffff', /* flesh */
  's':'#222222', /* seed */
  'o':'#5f0b3c'  /* outline */
};
const dragonMap = [
"       l   l    ",
"     lmmmmmm l  ",
"   lmmmmMMMMmml ",
"  lmmMMMMMMMMmml",
" lmmMMwwwwwwMMml",
" lmmMwwswwwsMMml",
" lmmMwwwwwwsMMml",
" lmmMwwswwwwMMml",
" lmmMwwwwwwMMm l",
"  lmmMMMMMMMMl  ",
"   lmmmmMMMMl   ",
"     lmmmmml    ",
"      l m l     ",
"        l       ",
"                ",
"                ",
];

/* glow grids */
drawPixelArt("peach", peachPalette, peachMap);
drawPixelArt("dragon", dragonPalette, dragonMap);

/* --------- API wiring --------------------------------------------------- */
async function pingHealth(){
  try{
    const r = await fetch("/health");
    if(!r.ok) throw new Error("not ok");
    const j = await r.json();
    document.getElementById("health").textContent =
      `ok · openai …${(j.openai_key_suffix||'????')} · deepseek …${(j.deepseek_key_suffix||'????')}`;
  }catch(e){
    document.getElementById("health").textContent = "health: unreachable";
  }
}
pingHealth();

document.getElementById("chatBtn").onclick = async () => {
  const txt = document.getElementById("chatText").value.trim();
  const out = document.getElementById("chatOut");
  out.textContent = "…";
  try{
    const r = await fetch("/chat", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        messages: [{role:"user", content: txt || "hello"}],
        temperature: 0.7
      })
    });
    const j = await r.json();
    out.textContent = `PEACH:\n${j.peach}\n\n— — —\n\nDRAGON:\n${j.dragon}`;
  }catch(e){
    out.textContent = "error hitting /chat";
  }
}

document.getElementById("debateBtn").onclick = async () => {
  const topic = document.getElementById("debateTopic").value.trim() || "Is generative AI good for artists?";
  const out = document.getElementById("debateOut");
  out.textContent = "starting…";
  try{
    const r = await fetch("/debate", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        prompt: topic, rounds: 2,
        persona_peach: "Punk Riot Grrrl",
        persona_dragon: "Philosophical Trickster"
      })
    });
    const j = await r.json();
    let text = `TOPIC: ${j.topic}\n`;
    j.transcript.forEach((t,i)=>{
      text += `\nRound ${i+1}\nPEACH: ${t.peach}\nDRAGON: ${t.dragon}\n`;
    });
    text += `\nCLOSING\nPEACH: ${j.closing.peach}\nDRAGON: ${j.closing.dragon}\n`;
    out.textContent = text;
  }catch(e){
    out.textContent = "error hitting /debate";
  }
}
</script>
</body>
</html>
"""

# --- Routes ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "time": __import__("datetime").datetime.utcnow().isoformat(),
        "openai_key_suffix": (OPENAI_API_KEY[-4:] if OPENAI_API_KEY else None),
        "deepseek_key_suffix": (DEEPSEEK_API_KEY[-4:] if DEEPSEEK_API_KEY else None),
    }

@app.post("/chat")
def chat(req: ChatRequest) -> JSONResponse:
    base = [{"role": "system", "content": PEACH_STYLE}]
    peach = openai_chat(base + [m.dict() for m in req.messages], model=req.model_peach or "gpt-4.1",
                        temperature=req.temperature)

    base_d = [{"role": "system", "content": DRAGON_STYLE}]
    dragon = deepseek_chat(base_d + [m.dict() for m in req.messages], model=req.model_dragon or "deepseek-chat",
                           temperature=req.temperature)

    return JSONResponse({"timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                         "peach": peach, "dragon": dragon})

@app.post("/debate")
def debate(req: DebateRequest) -> JSONResponse:
    result = run_debate(
        prompt=req.prompt,
        rounds=req.rounds,
        persona_peach=req.persona_peach,
        persona_dragon=req.persona_dragon,
        model_peach=req.model_peach or "gpt-4.1",
        model_dragon=req.model_dragon or "deepseek-chat",
        temperature=req.temperature,
    )
    return JSONResponse(result)


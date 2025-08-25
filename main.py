# main.py
import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# Bootstrapping & logging
# ------------------------------------------------------------------------------
load_dotenv(".env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("feministbot")
log.info("Feminist Chatbot logging initialized.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

# Default models (you can override per request)
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1")
DEEPSEEK_MODEL_DEFAULT = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# HTTP client (no HTTP/2 to avoid h2 dependency)
HTTP_CLIENT = httpx.Client(timeout=30, trust_env=False)

# ------------------------------------------------------------------------------
# Tiny language helper
# ------------------------------------------------------------------------------
def detect_lang(text: str) -> str:
    """Ultra-simple heuristic: if Spanish hint words/accents are present -> 'es' else 'en'."""
    if not text:
        return "en"
    lowered = text.lower()
    if any(w in lowered for w in ["¿", "¡", "qué", "cómo", "por qué", "porque", "gracias", "hola"]):
        return "es"
    if any(c in text for c in "áéíóúñÁÉÍÓÚÑ"):
        return "es"
    return "en"

# ------------------------------------------------------------------------------
# Persona system prompts (chat)
# ------------------------------------------------------------------------------
def persona_system_prompt(persona: str, lang: str) -> str:
    if lang == "es":
        if persona == "peach":
            return (
                "Eres PEACH: energía punk riot grrrl—directa, mordaz, solidaria. "
                "Habla claro, en 1–2 párrafos cortos. Sé valiente, práctica y con humor. "
                "Responde en el mismo idioma que el usuario."
            )
        else:
            return (
                "Eres DRAGON: trickster filosófica—ingeniosa, curiosa y juguetona. "
                "Usa preguntas, metáforas y giros conceptuales en 1–2 párrafos cortos. "
                "Responde en el mismo idioma que el usuario."
            )
    else:
        if persona == "peach":
            return (
                "You are PEACH: punk riot grrrl energy—direct, biting, supportive. "
                "Speak clearly in 1–2 short paragraphs. Be bold, practical, a little funny. "
                "Reply in the user's language."
            )
        else:
            return (
                "You are DRAGON: a playful philosophical trickster—witty, curious, paradox-loving. "
                "Use questions, metaphors, and flips in 1–2 short paragraphs. "
                "Reply in the user's language."
            )

# ------------------------------------------------------------------------------
# Antagonistic debate system prompts (with bias call-outs)
# ------------------------------------------------------------------------------
def antagonist_sys(persona: str, lang: str) -> str:
    if lang == "es":
        if persona == "peach":
            return (
                "Eres PEACH (energía punk riot grrrl), con ChatGPT de OpenAI. "
                "Debate con intensidad pero sin agresiones. Toma posturas firmes, sé concisa y directa, "
                "y cuestiona los supuestos del oponente. Señala vaguedades, pide evidencia y muestra "
                "cómo el entrenamiento o ajuste pueden influir en las respuestas. Mantén 2 párrafos cortos "
                "máximo y agrega una línea que empiece con 'Sesgo detectado:' nombrando un supuesto o punto ciego "
                "en el último mensaje del oponente. Evita ataques personales."
            )
        else:
            return (
                "Eres DRAGON (trickster filosófica y juguetona), con DeepSeek. "
                "Debate con precisión e ingenio. Expón premisas ocultas, prueba casos límite y da la vuelta a los términos. "
                "Contrasta estilos de razonamiento que distintas familias de modelos podrían favorecer. Mantén 2 párrafos "
                "cortos máximo y agrega una línea que empiece con 'Sesgo detectado:' nombrando un supuesto o punto ciego "
                "en el último mensaje del oponente. Evita ataques personales."
            )
    else:
        if persona == "peach":
            return (
                "You are PEACH (punk riot grrrl energy), powered by OpenAI's ChatGPT. "
                "Debate with heat but not hate. Take firm positions; be vivid and concise. "
                "Call out vague claims, ask for evidence, and highlight how training/tuning shape answers. "
                "Keep to 2 short paragraphs max, then add one line starting with 'Called-out bias:' naming a likely "
                "assumption or blind spot in the opponent's last message. No personal attacks."
            )
        else:
            return (
                "You are DRAGON (playful philosophical trickster), powered by DeepSeek. "
                "Debate with precision and wit. Expose hidden premises, test edge cases, flip terms. "
                "Contrast reasoning styles different model families might favor. "
                "Keep to 2 short paragraphs max, then add one line starting with 'Called-out bias:' naming a likely "
                "assumption or blind spot in the opponent's last message. No personal attacks."
            )

# ------------------------------------------------------------------------------
# Minimal OpenAI + DeepSeek callers
# ------------------------------------------------------------------------------
def call_openai(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.7) -> str:
    """Call OpenAI Chat Completions via the REST endpoint using httpx (no SDK dependency)."""
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or OPENAI_MODEL_DEFAULT,
        "messages": messages,
        "temperature": temperature,
    }
    r = HTTP_CLIENT.post(url, headers=headers, json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

def call_deepseek(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.7) -> str:
    """Call DeepSeek chat completions."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or DEEPSEEK_MODEL_DEFAULT,
        "messages": messages,
        "temperature": temperature,
    }
    r = HTTP_CLIENT.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"DeepSeek error {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

def chat_complete(api: str, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.7) -> str:
    if api == "openai":
        return call_openai(messages, model=model, temperature=temperature)
    elif api == "deepseek":
        return call_deepseek(messages, model=model, temperature=temperature)
    else:
        raise ValueError("api must be 'openai' or 'deepseek'")

# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    # accept either "messages" (OpenAI-style) or a single "message"
    messages: Optional[List[Dict[str, str]]] = None
    message: Optional[str] = None
    # persona: 'peach', 'dragon', 'both'
    persona: str = Field(default="both")
    # api: 'openai', 'deepseek', 'both' (ignored if persona != 'both')
    api: str = Field(default="both")
    lang: str = Field(default="auto")
    model_peach: Optional[str] = None
    model_dragon: Optional[str] = None
    temperature: float = 0.7

class DebateRequest(BaseModel):
    prompt: str
    rounds: int = 2
    lang: str = "auto"
    persona_peach: str = "Punk Riot Grrrl"
    persona_dragon: str = "Philosophical Trickster"
    model_peach: Optional[str] = None
    model_dragon: Optional[str] = None
    temperature_peach: float = 0.9
    temperature_dragon: float = 0.95

# ------------------------------------------------------------------------------
# Debate engine
# ------------------------------------------------------------------------------
def run_debate(
    prompt: str,
    rounds: int = 2,
    lang: str = "auto",
    model_peach: Optional[str] = None,
    model_dragon: Optional[str] = None,
    temperature_peach: float = 0.9,
    temperature_dragon: float = 0.95,
) -> Dict[str, Any]:
    if lang == "auto":
        lang = detect_lang(prompt)

    sys_peach = antagonist_sys("peach", lang)
    sys_dragon = antagonist_sys("dragon", lang)

    opener_key = "Tema de debate" if lang == "es" else "Debate topic"

    peach_msgs = [
        {"role": "system", "content": sys_peach},
        {"role": "user", "content": f"{opener_key}: {prompt}"},
    ]
    dragon_msgs = [
        {"role": "system", "content": sys_dragon},
        {"role": "user", "content": f"{opener_key}: {prompt}"},
    ]

    # Round 1
    peach_open = chat_complete("openai", peach_msgs, model=model_peach, temperature=temperature_peach)
    peach_msgs.append({"role": "assistant", "content": peach_open})
    dragon_msgs.append({"role": "user", "content": peach_open})

    dragon_reply = chat_complete("deepseek", dragon_msgs, model=model_dragon, temperature=temperature_dragon)
    dragon_msgs.append({"role": "assistant", "content": dragon_reply})
    peach_msgs.append({"role": "user", "content": dragon_reply})

    last_p, last_d = peach_open, dragon_reply

    # Additional rounds
    for _ in range(max(0, rounds - 1)):
        last_p = chat_complete("openai", peach_msgs, model=model_peach, temperature=temperature_peach)
        peach_msgs.append({"role": "assistant", "content": last_p})
        dragon_msgs.append({"role": "user", "content": last_p})

        last_d = chat_complete("deepseek", dragon_msgs, model=model_dragon, temperature=temperature_dragon)
        dragon_msgs.append({"role": "assistant", "content": last_d})
        peach_msgs.append({"role": "user", "content": last_d})

    return {
        "peach": last_p,
        "dragon": last_d,
        "peach_history": peach_msgs,
        "dragon_history": dragon_msgs,
    }

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(title="FeministBot · Peach × Dragon", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# HTML (no f-strings, so CSS/JS braces are safe)
# ------------------------------------------------------------------------------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FeministBot · Peach × Dragon</title>
<style>
  :root{
    --bg:#0b0d10; --panel:#11141a; --panel-2:#0f1016; --text:#e8ebef; --muted:#b9bfca;
    --accent-1:#ff2e92; /* dragon */ --accent-2:#ff9d6c; /* peach */
    --pill:#1c1f27; --good:#25d366; --warning:#ff9500; --error:#ff453a;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100vh;background:var(--bg);color:var(--text);font:16px system-ui, -apple-system, Segoe UI, Roboto, Inter, "Helvetica Neue", Arial, sans-serif;overflow-x:hidden}
  a{color:inherit}
  .wrap{max-width:1200px;margin:0 auto;padding:14px;display:grid;gap:16px;min-height:100vh}

  .grid{display:grid;gap:16px}
  @media (min-width:900px){ .grid{grid-template-columns:1.1fr 1fr} }

  .fruit-grid{display:grid;gap:16px}
  @media (min-width:700px){ .fruit-grid{grid-template-columns:1fr 1fr} }

  .card{background:linear-gradient(180deg,var(--panel),var(--panel-2));border:1px solid #1b1f28;
    box-shadow:0 0 0 1px #0b0d12, 0 12px 40px rgba(0,0,0,.45), inset 0 0 0 1px rgba(255,255,255,.03);
    border-radius:24px;padding:16px;transition:transform .2s ease, box-shadow .2s ease;}

  .fruit-card{display:flex;align-items:center;justify-content:center;height:300px;position:relative;cursor:pointer}
  .fruit-card svg{width:clamp(140px,35vw,200px);height:auto;image-rendering:pixelated;shape-rendering:crispEdges;transition:transform .3s ease;}
  .fruit-card:hover svg{transform:scale(1.05)}
  .fruit-card.dragon svg{transform:scale(.88)}
  .fruit-card.dragon:hover svg{transform:scale(.93)}

  .fruit-card.peach{ box-shadow:0 0 60px 0 rgba(255,157,108,.12), inset 0 0 0 1px rgba(255,157,108,.1);}
  .fruit-card.dragon{ box-shadow:0 0 60px 0 rgba(255,46,146,.14), inset 0 0 0 1px rgba(255,46,146,.1);}

  .char-info{position:absolute;bottom:12px;left:12px;right:12px;background:rgba(0,0,0,.75);backdrop-filter:blur(10px);
    border-radius:12px;padding:8px 12px;font-size:14px;opacity:0;transform:translateY(10px);transition:all .3s ease;pointer-events:none}
  .fruit-card:hover .char-info{opacity:1;transform:translateY(0)}

  @keyframes chomp { 0%,20%,100%{opacity:0} 30%,40%{opacity:1} 50%,90%{opacity:0} }
  .peach-chomp{ animation:chomp 3.5s infinite step-end; }

  @keyframes blink { 0%,92%,100%{transform:scaleY(1)} 93%,99%{transform:scaleY(.1)} }
  .eye{ transform-origin:50% 50%; animation: blink 4.5s infinite; }

  .pill{background:#1a1f28;border:1px solid #242936;border-radius:999px;padding:8px 12px;display:inline-flex;gap:10px;align-items:center}
  .btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:12px 16px;border-radius:14px;border:1px solid #1b1f28;cursor:pointer;background:linear-gradient(180deg,#1a1f27,#12141a);color:var(--text);font-weight:600;transition:.2s all ease;min-height:44px}
  .btn:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,0,0,.3)}
  .btn--peach{ background:linear-gradient(90deg,#ff9d6c,#ff7181) }
  .btn--dragon{ background:linear-gradient(90deg,#ff2e92,#7b2cff) }
  .btn--go{ background:linear-gradient(90deg,#26d363,#ff9d6c) }

  .panel{background:var(--panel);border:1px solid #1b1f28;border-radius:20px;padding:16px;position:relative}
  .panel-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
  .h{font-weight:700;color:#fff;margin:0}
  .muted{color:var(--muted)}
  .tags{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
  .tag{padding:6px 12px;border-radius:20px;border:1px solid #242936;background:#121620;color:#bfc6d2;cursor:pointer}
  .tag.active{background:linear-gradient(90deg,var(--accent-2),var(--accent-1));color:#fff;border-color:transparent}

  textarea,input{width:100%;background:#0e1116;border:2px solid #1b1f28;color:#e8ebef;border-radius:12px;padding:12px 14px;outline:none;font-size:16px}
  textarea{min-height:100px;resize:vertical}
  .log{display:grid;gap:12px;max-height:400px;overflow:auto;padding:12px;background:#0a0c0f;border-radius:16px}
  .bubble{background:linear-gradient(180deg,#0e1116,#0a0d12);border:1px solid #1b1f28;border-radius:16px;padding:14px 16px}
  .bubble.peach{border-left:3px solid var(--accent-2)}
  .bubble.dragon{border-left:3px solid var(--accent-1)}
  .bubble.system{border-left:3px solid var(--warning)}

  .why{line-height:1.5}
  .why b{color:#fff}
  .foot{font-size:12px;color:#98a2b3}
</style>
</head>
<body>
  <div class="wrap">
    <section class="fruit-grid">
      <div class="card fruit-card peach" aria-label="Peach Character" tabindex="0" role="button">
        <svg viewBox="0 0 16 16" role="img" aria-label="Pixel Peach">
          <rect x="7" y="1" width="2" height="1" fill="#7a4a2c"/>
          <rect x="6" y="0" width="4" height="1" fill="#34d17a"/>
          <rect x="3" y="3" width="10" height="10" fill="#964d37"/>
          <rect x="4" y="4" width="8" height="8" fill="#f89c79"/>
          <rect x="5" y="5" width="6" height="6" fill="#ffb497"/>
          <rect class="peach-chomp" x="11" y="7" width="2" height="2" fill="#0b0d10"/>
          <rect x="7" y="12" width="2" height="2" fill="#c57658"/>
        </svg>
        <div class="char-info"><b>🌸 Peach</b> · Punk Riot Grrrl • Fierce & Practical</div>
      </div>

      <div class="card fruit-card dragon" aria-label="Dragon Character" tabindex="0" role="button">
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
        <div class="char-info"><b>🐉 Dragon</b> · Philosophical Trickster • Witty & Curious</div>
      </div>
    </section>

    <section class="card why">
      <h2 style="margin-top:0">Why two fruits?</h2>
      <p>
        <b>Peach</b> is powered by <b>OpenAI’s ChatGPT</b>. <b>Dragon</b> is powered by <b>DeepSeek</b>.
        They grew up in different houses, with different table manners, and they argue like cousins at a wedding.
        Peach says, “Obviously—”. Dragon says, “Hold on.” That’s the point: two smart systems, two sets of priors.
        Watch them rub edges until sparks (or tenderness) fly.
      </p>
      <p class="foot">
        Tiny truth: models carry habits from how they’re trained and tuned. That’s not scary—it’s useful. Compare them.
        Make them disagree. See what shakes loose.
      </p>
    </section>

    <section class="grid">
      <div class="panel">
        <div class="panel-header">
          <h3 class="h">Chat</h3>
          <div class="tags">
            <span class="tag active" data-persona="both">Both</span>
            <span class="tag" data-persona="peach">🌸 Peach</span>
            <span class="tag" data-persona="dragon">🐉 Dragon</span>
          </div>
        </div>
        <textarea id="chatInput" placeholder="Ask in English or Español… (they’ll answer in your language)"></textarea>
        <div style="display:flex;gap:10px;justify-content:flex-end;margin-top:10px">
          <button class="btn btn--peach" id="sendChat"><span>Send</span></button>
        </div>
        <div id="chatLog" class="log" style="margin-top:12px"></div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <h3 class="h">Debate (antagonistic)</h3>
        </div>
        <input id="debTopic" placeholder="e.g., Is generative AI good for artists?" />
        <div style="display:flex;gap:10px;align-items:center;margin:10px 0">
          <span class="pill">Rounds <input id="debRounds" type="range" min="1" max="5" value="2" style="margin:0 10px;accent-color:#ff9d6c" />
          <span id="roundsOut">2</span></span>
        </div>
        <div style="display:flex;gap:10px;justify-content:flex-end">
          <button class="btn btn--dragon" id="startDebate"><span>Start Debate</span></button>
        </div>
        <div id="debLog" class="log" style="margin-top:12px"></div>
      </div>
    </section>
  </div>

<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

let persona = 'both';
$$('.tag').forEach(t => t.onclick = () => {
  $$('.tag').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  persona = t.dataset.persona;
});

$('#sendChat').onclick = async () => {
  const msg = $('#chatInput').value.trim();
  if (!msg) return;
  addBubble('#chatLog', 'system', '…thinking…');
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ message: msg, persona, api: 'both', lang: 'auto' })
    });
    const data = await res.json();
    $('#chatLog').innerHTML = '';
    if (data.peach) addBubble('#chatLog', 'peach', data.peach);
    if (data.dragon) addBubble('#chatLog', 'dragon', data.dragon);
  } catch (e) {
    addBubble('#chatLog', 'system', 'Error. Try again.');
  }
};

$('#debRounds').oninput = () => $('#roundsOut').textContent = $('#debRounds').value;
$('#startDebate').onclick = async () => {
  const topic = $('#debTopic').value.trim();
  if (!topic) return;
  $('#debLog').innerHTML = ''; addBubble('#debLog', 'system', 'Starting…');
  try {
    const res = await fetch('/debate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ prompt: topic, rounds: parseInt($('#debRounds').value), lang: 'auto' })
    });
    const data = await res.json();
    $('#debLog').innerHTML = '';
    data.peach_history?.forEach(turn => { if (turn.role === 'assistant') addBubble('#debLog','peach',turn.content) });
    data.dragon_history?.forEach(turn => { if (turn.role === 'assistant') addBubble('#debLog','dragon',turn.content) });
  } catch (e) {
    addBubble('#debLog', 'system', 'Error. Try again.');
  }
};

function addBubble(sel, kind, text){
  const div = document.createElement('div');
  div.className = `bubble ${kind}`;
  const prefix = kind === 'peach' ? '🌸 Peach' : kind === 'dragon' ? '🐉 Dragon' : 'ℹ️';
  div.innerHTML = `<div style="font-weight:600;margin-bottom:6px">${prefix}</div><div>${escapeHtml(text||'')}</div>`;
  document.querySelector(sel).appendChild(div);
  document.querySelector(sel).scrollTop = document.querySelector(sel).scrollHeight;
}
function escapeHtml(s){return (s||'').replace(/[&<>"']/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[m]))}
</script>
</body>
</html>
"""

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML, status_code=200)

@app.get("/health")
def health():
    now = datetime.now(timezone.utc).isoformat()
    openai_suffix = OPENAI_API_KEY[-4:] if OPENAI_API_KEY else ""
    deepseek_suffix = DEEPSEEK_API_KEY[-4:] if DEEPSEEK_API_KEY else ""
    return {"ok": True, "time": now, "openai_key_suffix": openai_suffix, "deepseek_key_suffix": deepseek_suffix}

@app.post("/chat")
def chat(req: ChatRequest = Body(...)):
    # Normalize input
    if req.lang == "auto":
        guess_source = req.message or (req.messages[0]["content"] if req.messages else "")
        lang = detect_lang(guess_source)
    else:
        lang = "es" if req.lang.lower().startswith("es") else "en"

    # Build messages
    if req.messages:
        user_msgs = req.messages
    elif req.message:
        user_msgs = [{"role": "user", "content": req.message}]
    else:
        raise HTTPException(status_code=422, detail="Provide 'message' or 'messages'.")

    # Persona routing
    out: Dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat()}

    def build_for(persona: str) -> List[Dict[str, str]]:
        return [{"role":"system", "content": persona_system_prompt(persona, lang)}] + user_msgs

    try:
        if req.persona in ("both", "peach"):
            out["peach"] = chat_complete(
                api="openai",
                messages=build_for("peach"),
                model=req.model_peach or OPENAI_MODEL_DEFAULT,
                temperature=req.temperature,
            )
        if req.persona in ("both", "dragon"):
            out["dragon"] = chat_complete(
                api="deepseek",
                messages=build_for("dragon"),
                model=req.model_dragon or DEEPSEEK_MODEL_DEFAULT,
                temperature=req.temperature,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {type(e).__name__}: {e}")

    return out

@app.post("/debate")
def debate(req: DebateRequest = Body(...)):
    try:
        result = run_debate(
            prompt=req.prompt,
            rounds=req.rounds,
            lang=req.lang,
            model_peach=req.model_peach or OPENAI_MODEL_DEFAULT,
            model_dragon=req.model_dragon or DEEPSEEK_MODEL_DEFAULT,
            temperature_peach=req.temperature_peach,
            temperature_dragon=req.temperature_dragon,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debate error: {type(e).__name__}: {e}")


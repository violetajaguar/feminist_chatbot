import os
import logging
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import httpx
import requests
from openai import OpenAI

# =========================
# Env & Logging
# =========================
load_dotenv(".env", override=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("feminist_chatbot")
logger.info("Feminist Chatbot logging initialized.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

logger.info("[DEBUG] OPENAI …%s", (OPENAI_API_KEY or "")[-4:])
logger.info("[DEBUG] DEEPSEEK …%s", (DEEPSEEK_API_KEY or "")[-4:])

# =========================
# HTTP client & Model Clients
# =========================
HTTP_CLIENT = httpx.Client(
    timeout=30,
    trust_env=False,                        # ignore stray proxy vars
    transport=httpx.HTTPTransport(retries=3)
)

openai_client: Optional[OpenAI] = OpenAI(
    api_key=OPENAI_API_KEY, http_client=HTTP_CLIENT
) if OPENAI_API_KEY else None

# =========================
# Persona Hints (optional)
# =========================
PERSONA_PROMPTS: Dict[str, str] = {
    "Visionary Poet": "Lyrical, visionary, Audre-Lorde energy. Metaphor, care, fire.",
    "Radical Hacker": "Cyberfeminist, subversive, precise about systems and power. Hack the patriarchy.",
    "Ancestral Wisdom Keeper": "Grounded, intergenerational, decolonial feminist care; rooted in land and community.",
    "Punk Riot Grrrl": "Bold, DIY, loud, anti-authoritarian. Zine-core, no apologies.",
    "Philosophical Trickster": "Playful paradoxes, Butler/De Beauvoir vibes. Ask sharp questions.",
}

# =========================
# Engines (Peach / Dragon Fruit)
# =========================
def peach_reply(messages: List[Dict[str, str]],
                model: str = "gpt-4.1",
                temperature: float = 0.7) -> str:
    if not openai_client:
        return "Peach error: Missing OPENAI_API_KEY"
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.exception("Peach (OpenAI) API call failed")
        return f"Peach error: {type(e).__name__}: {e}"

def dragon_reply(messages: List[Dict[str, str]],
                 model: str = "deepseek-chat",
                 temperature: float = 0.7) -> str:
    if not DEEPSEEK_API_KEY:
        return "Dragon Fruit error: Missing DEEPSEEK_API_KEY"
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {"model": model, "messages": messages, "temperature": temperature}
        r = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception("Dragon Fruit (DeepSeek) API call failed")
        return f"Dragon Fruit error: {type(e).__name__}: {e}"

# =========================
# Debate Orchestration
# =========================
def _persona_text(p: Optional[str]) -> str:
    if not p:
        return ""
    return PERSONA_PROMPTS.get(p, p)  # allow key or freeform text

def _debate_system(agent_name: str, persona: Optional[str]) -> str:
    persona_hint = _persona_text(persona)
    base = (
        f"You are {agent_name}, a sharp feminist debater.\n"
        f"Goals: rebut the opponent, advance your thesis, stay factual, concise, and stylish.\n"
        f"Tone: confident, incisive, never cruel; no slurs; no harassment.\n"
        f"Method: challenge assumptions, cite examples or patterns, propose alternatives.\n"
        f"Style constraints: ≤180 words; end with either a punchy one-liner or one provocative question.\n"
    )
    return base + (f"Persona hint: {persona_hint}\n" if persona_hint else "")

def run_debate(
    topic: str,
    rounds: int = 3,
    persona_peach: Optional[str] = None,
    persona_dragon: Optional[str] = None,
    temperature: float = 0.7,
    peach_model: str = "gpt-4.1",
    dragon_model: str = "deepseek-chat",
):
    transcript: List[Dict[str, Any]] = []
    last_peach = ""
    last_dragon = ""

    for i in range(1, rounds + 1):
        # Peach turn
        msgs_peach = [
            {"role": "system", "content": _debate_system("Peach", persona_peach)},
            {"role": "user", "content":
                f"Debate topic: {topic}\n"
                f"Opponent last point:\n{(last_dragon or 'None yet. Lead the debate with a strong opening.')}\n"
                f"Your task: deliver a crisp argument and one new angle. Round {i}/{rounds}."}
        ]
        peach = peach_reply(msgs_peach, model=peach_model, temperature=temperature)
        transcript.append({"agent": "peach", "round": i, "content": peach})
        last_peach = peach or ""

        # Dragon turn
        msgs_dragon = [
            {"role": "system", "content": _debate_system("Dragon Fruit", persona_dragon)},
            {"role": "user", "content":
                f"Debate topic: {topic}\n"
                f"Opponent (Peach) said:\n{last_peach}\n"
                f"Your task: directly rebut key claims, then add one original point. Round {i}/{rounds}."}
        ]
        dragon = dragon_reply(msgs_dragon, model=dragon_model, temperature=temperature)
        transcript.append({"agent": "dragon", "round": i, "content": dragon})
        last_dragon = dragon or ""

    return transcript

# =========================
# FastAPI App
# =========================
app = FastAPI(title="FeministBot", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Models -----
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    persona: Optional[str] = None
    peach_model: Optional[str] = "gpt-4.1"
    dragon_model: Optional[str] = "deepseek-chat"
    temperature: Optional[float] = 0.7

class DebateRequest(BaseModel):
    prompt: str = Field(..., description="Debate topic")
    rounds: Optional[int] = 3
    persona_peach: Optional[str] = None
    persona_dragon: Optional[str] = None
    peach_model: Optional[str] = "gpt-4.1"
    dragon_model: Optional[str] = "deepseek-chat"
    temperature: Optional[float] = 0.7

# ----- Routes -----
@app.get("/", response_class=HTMLResponse)
def root() -> str:
    personas = "".join(f'<option value="{p}">{p}</option>' for p in PERSONA_PROMPTS)
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>FeministBot</title>
    <style>
      body{{font-family:system-ui;margin:2rem;max-width:980px}}
      textarea,input,select,button{{font:inherit}}
      .grid{{display:grid;gap:12px}}
      .row{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
      .card{{padding:12px;border:1px solid #ddd;border-radius:12px}}
      button{{padding:10px 16px;border-radius:10px;border:1px solid #111;background:#111;color:#fff;cursor:pointer}}
      h1{{margin-bottom:0.25rem}}
      small{{color:#666}}
      pre{{white-space:pre-wrap}}
    </style>
  </head>
  <body>
    <h1>FeministBot ✨</h1>
    <small>Peach = OpenAI • Dragon = DeepSeek</small>

    <h2>Chat</h2>
    <div class="grid">
      <label>Persona
        <select id="persona">
          <option value="">(neutral)</option>
          {personas}
        </select>
      </label>
      <textarea id="msg" rows="4" placeholder="say hi…"></textarea>
      <button id="send">Send</button>
      <div class="row">
        <div class="card"><b>Peach</b><pre id="peach"></pre></div>
        <div class="card"><b>Dragon</b><pre id="dragon"></pre></div>
      </div>
    </div>

    <h2>Debate</h2>
    <div class="grid">
      <input id="topic" placeholder="Debate topic, e.g., Is generative AI good for artists?"/>
      <div class="row">
        <label>Peach Persona
          <select id="persona_peach">
            <option value="">(neutral)</option>
            {personas}
          </select>
        </label>
        <label>Dragon Persona
          <select id="persona_dragon">
            <option value="">(neutral)</option>
            {personas}
          </select>
        </label>
      </div>
      <label>Rounds <input id="rounds" type="number" min="1" max="6" value="3" style="width:80px"/></label>
      <button id="fight">Start Debate</button>
      <div id="debate" class="grid"></div>
    </div>

    <script>
      const $ = (id)=>document.getElementById(id);

      $("send").onclick = async ()=>{
        const persona = $("persona").value || null;
        const content = $("msg").value.trim();
        if(!content) return;
        $("peach").textContent = "…";
        $("dragon").textContent = "…";
        const res = await fetch("/chat", {{
          method:"POST",
          headers:{{"Content-Type":"application/json"}},
          body: JSON.stringify({{
            persona,
            messages:[{{role:"user", content}}]
          }})
        }});
        const j = await res.json();
        $("peach").textContent = j.peach || "";
        $("dragon").textContent = j.dragon || "";
      };

      $("fight").onclick = async ()=>{
        const prompt = $("topic").value.trim();
        if(!prompt) return;
        const persona_peach = $("persona_peach").value || null;
        const persona_dragon = $("persona_dragon").value || null;
        const rounds = parseInt($("rounds").value || "3", 10);
        $("debate").innerHTML = "<div class='card'>starting…</div>";
        const res = await fetch("/debate", {{
          method:"POST",
          headers:{{"Content-Type":"application/json"}},
          body: JSON.stringify({{
            prompt, rounds, persona_peach, persona_dragon
          }})
        }});
        const j = await res.json();
        let html = "";
        for (const turn of (j.transcript || [])) {{
          const who = turn.agent === "peach" ? "Peach" : "Dragon";
          html += `<div class="card"><b>Round ${'{'}turn.round{'}'} — ${'{'}who{'}'}</b><pre>${'{'}turn.content || ""{'}'}</pre></div>`;
        }}
        $("debate").innerHTML = html || "<div class='card'>No transcript?</div>";
      };
    </script>
  </body>
</html>
"""

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat() + "Z",
        "openai_key_suffix": (OPENAI_API_KEY or "")[-4:],
        "deepseek_key_suffix": (DEEPSEEK_API_KEY or "")[-4:],
    }

@app.post("/chat")
def chat(req: "ChatRequest"):
    try:
        msgs = [m.dict() for m in req.messages]
        if req.persona:
            msgs = [{"role": "system", "content": PERSONA_PROMPTS.get(req.persona, req.persona)}] + msgs

        peach_text = peach_reply(
            msgs, model=req.peach_model or "gpt-4.1",
            temperature=req.temperature if req.temperature is not None else 0.7
        )
        dragon_text = dragon_reply(
            msgs, model=req.dragon_model or "deepseek-chat",
            temperature=req.temperature if req.temperature is not None else 0.7
        )

        return JSONResponse({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "peach": peach_text,
            "dragon": dragon_text,
        })
    except Exception as e:
        logger.exception("Unhandled /chat error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debate")
def debate(req: "DebateRequest"):
    try:
        tr = run_debate(
            topic=req.prompt,
            rounds=req.rounds or 3,
            persona_peach=req.persona_peach,
            persona_dragon=req.persona_dragon,
            temperature=req.temperature if req.temperature is not None else 0.7,
            peach_model=req.peach_model or "gpt-4.1",
            dragon_model=req.dragon_model or "deepseek-chat",
        )
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "topic": req.prompt,
            "transcript": tr,
        }
    except Exception as e:
        logger.exception("Unhandled /debate error")
        raise HTTPException(status_code=500, detail=str(e))


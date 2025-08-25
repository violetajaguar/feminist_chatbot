import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# --- setup ---
load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("feminist-bot")
logger.info("Feminist Chatbot logging initialized.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

HTTP_CLIENT = httpx.Client(
    timeout=30,
    trust_env=False,
    transport=httpx.HTTPTransport(retries=3),
)

# OpenAI client (HTTP/2 not required)
openai_client = OpenAI(api_key=OPENAI_API_KEY, http_client=HTTP_CLIENT)

# --- personas ---
PERSONAS: Dict[str, str] = {
    "Visionary Poet": (
        "You are the Visionary Poet: lyrical, incisive, inspired by Audre Lorde and Maya Angelou. "
        "You weave metaphor and heat. Be concise, bold, feminist, and generous."
    ),
    "Radical Hacker": (
        "You are the Radical Hacker: cyberfeminist edge, systems-thinking, direct and subversive. "
        "Speak like a tactician breaking patriarchal code."
    ),
    "Ancestral Wisdom Keeper": (
        "You are the Ancestral Wisdom Keeper: grounded, intergenerational, tender but firm. "
        "Care ethics, community, earth, memory."
    ),
    "Punk Riot Grrrl": (
        "You are the Punk Riot Grrrl: loud, DIY, anti-authoritarian, funny and furious. "
        "Short sentences. Zine energy. Kick the door in."
    ),
    "Philosophical Trickster": (
        "You are the Philosophical Trickster: playful, rigorous, Butler/De Beauvoir/Irigaray vibes. "
        "Ask sharp questions. Flip assumptions with wit."
    ),
}

def system_prompt_for(persona: str) -> str:
    return PERSONAS.get(persona, PERSONAS["Punk Riot Grrrl"])

# --- helpers ---
def ask_openai(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.7) -> str:
    mdl = model or "gpt-4.1"
    try:
        resp = openai_client.chat.completions.create(
            model=mdl,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[OpenAI error] {type(e).__name__}: {e}"

def ask_deepseek(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.7) -> str:
    if not DEEPSEEK_API_KEY:
        # Fallback to OpenAI if DeepSeek key is missing
        return ask_openai(messages, model="gpt-4o-mini", temperature=temperature)

    mdl = model or "deepseek-chat"
    try:
        r = HTTP_CLIENT.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": mdl, "messages": messages, "temperature": temperature},
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[DeepSeek error] {type(e).__name__}: {e}"

# --- API types ---
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    model_peach: Optional[str] = None
    model_dragon: Optional[str] = None
    persona_peach: Optional[str] = "Punk Riot Grrrl"
    persona_dragon: Optional[str] = "Philosophical Trickster"

class DebateRequest(BaseModel):
    prompt: str
    rounds: int = 2
    temperature: float = 0.7
    persona_peach: str = "Punk Riot Grrrl"
    persona_dragon: str = "Philosophical Trickster"
    model_peach: Optional[str] = None
    model_dragon: Optional[str] = None

# --- FastAPI app ---
app = FastAPI(title="FeministBot", version="1.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<html><head><title>FeministBot</title></head>
<body style="font-family: system-ui; padding:20px">
  <h1>FeministBot</h1>
  <p>Endpoints:</p>
  <ul>
    <li>GET /health</li>
    <li>POST /chat</li>
    <li>POST /debate</li>
  </ul>
</body></html>"""

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "time": datetime.now(timezone.utc).isoformat(),
        "openai_key_suffix": (OPENAI_API_KEY[-4:] if OPENAI_API_KEY else "MISSING"),
        "deepseek_key_suffix": (DEEPSEEK_API_KEY[-4:] if DEEPSEEK_API_KEY else "MISSING"),
    }

@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    # Peach (OpenAI)
    peach_msgs = [{"role": "system", "content": system_prompt_for(req.persona_peach)}] + req.messages
    peach_out = ask_openai(peach_msgs, model=req.model_peach, temperature=req.temperature)

    # Dragon (DeepSeek)
    dragon_msgs = [{"role": "system", "content": system_prompt_for(req.persona_dragon)}] + req.messages
    dragon_out = ask_deepseek(dragon_msgs, model=req.model_dragon, temperature=req.temperature)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "peach": peach_out,
        "dragon": dragon_out,
    }

def run_debate(prompt: str, rounds: int, temp: float,
               persona_peach: str, persona_dragon: str,
               model_peach: Optional[str], model_dragon: Optional[str]) -> List[Dict[str, str]]:
    transcript: List[Dict[str, str]] = []

    peach_hist = [{"role": "system", "content": system_prompt_for(persona_peach)}]
    dragon_hist = [{"role": "system", "content": system_prompt_for(persona_dragon)}]

    last_peach = ""
    last_dragon = ""

    for i in range(1, max(1, rounds) + 1):
        # Peach goes
        if last_dragon:
            peach_user = f"Counterpart said: {last_dragon}\nRespond directly. Keep it under 120 words. Topic: {prompt}"
        else:
            peach_user = f"Debate this. Go first. Keep it under 120 words. Topic: {prompt}"
        peach_turn = peach_hist + [{"role": "user", "content": peach_user}]
        peach_out = ask_openai(peach_turn, model=model_peach, temperature=temp)
        transcript.append({"speaker": "Peach", "round": i, "text": peach_out})
        peach_hist += [{"role": "user", "content": peach_user}, {"role": "assistant", "content": peach_out}]
        last_peach = peach_out

        # Dragon responds
        dragon_user = f"Counterpart said: {last_peach}\nCounter, precisely. Keep it under 120 words. Topic: {prompt}"
        dragon_turn = dragon_hist + [{"role": "user", "content": dragon_user}]
        dragon_out = ask_deepseek(dragon_turn, model=model_dragon, temperature=temp)
        transcript.append({"speaker": "Dragon", "round": i, "text": dragon_out})
        dragon_hist += [{"role": "user", "content": dragon_user}, {"role": "assistant", "content": dragon_out}]
        last_dragon = dragon_out

    return transcript

@app.post("/debate")
def debate(req: DebateRequest) -> Dict[str, Any]:
    transcript = run_debate(
        prompt=req.prompt,
        rounds=req.rounds,
        temp=req.temperature,
        persona_peach=req.persona_peach,
        persona_dragon=req.persona_dragon,
        model_peach=req.model_peach,
        model_dragon=req.model_dragon,
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": req.prompt,
        "transcript": transcript,
    }

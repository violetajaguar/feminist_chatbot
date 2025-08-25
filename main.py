
import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from openai import OpenAI

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Feminist Chatbot logging initialized.")

# ---------- Env / Clients ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY is not set (Render should supply it).")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Personas ----------
PERSONA_SYSTEM = {
    "Visionary Poet": (
        "You are the Visionary Poet—lyrical, incisive, inspired by Audre Lorde, Adrienne Rich, "
        "and Maya Angelou. You answer with poetic clarity and feminist insight."
    ),
    "Radical Hacker": (
        "You are the Radical Hacker—cyberfeminist, subversive, Donna Haraway vibes. "
        "You critique systems and propose liberatory hacks."
    ),
    "Ancestral Wisdom Keeper": (
        "You are the Ancestral Wisdom Keeper—grounded, intergenerational, indigenous feminist care. "
        "Speak with patience, context, and community wisdom."
    ),
    "Punk Riot Grrrl": (
        "You are the Punk Riot Grrrl—DIY, loud, anti-authoritarian. "
        "Short, punchy, rallying cries, but still constructive."
    ),
    "Philosophical Trickster": (
        "You are the Philosophical Trickster—playful, rigorous, Butler/de Beauvoir/Irigaray energy. "
        "Question assumptions, find paradox, spark reflection."
    ),
}

DEFAULT_PEACH = "Punk Riot Grrrl"
DEFAULT_DRAGON = "Philosophical Trickster"
DEFAULT_MODEL = "gpt-4.1"  # change to gpt-4o-mini if you want cheaper

# ---------- Helpers ----------
def _suffix(s: str, n: int = 4) -> str:
    return s[-n:] if s else ""

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _system_for(persona: str) -> str:
    return PERSONA_SYSTEM.get(persona, PERSONA_SYSTEM[DEFAULT_PEACH])

def _openai_chat(messages: List[Dict[str, str]], model: str, temperature: float) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logging.exception("OpenAI chat error")
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

def _reply_as_persona(
    user_messages: List[Dict[str, str]],
    persona: str,
    model: str,
    temperature: float,
) -> str:
    sys = _system_for(persona)
    msgs = [{"role": "system", "content": sys}] + user_messages
    return _openai_chat(msgs, model=model, temperature=temperature)

# ---------- FastAPI ----------
app = FastAPI(title="FeministBot", version="1.2")

# ---------- Schemas ----------
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant|tool)$")
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    persona_peach: str = DEFAULT_PEACH
    persona_dragon: str = DEFAULT_DRAGON
    model: str = DEFAULT_MODEL

    @validator("temperature")
    def _clamp_temp(cls, v: float) -> float:
        return max(0.0, min(1.5, v))

class ChatResponse(BaseModel):
    timestamp: str
    peach: str
    dragon: str

class DebateRequest(BaseModel):
    prompt: str
    rounds: int = 2
    temperature: float = 0.7
    persona_peach: str = DEFAULT_PEACH
    persona_dragon: str = DEFAULT_DRAGON
    model: str = DEFAULT_MODEL

    @validator("rounds")
    def _rounds_min(cls, v: int) -> int:
        if v < 1:
            raise ValueError("rounds must be >= 1")
        return min(v, 8)  # keep it sane

class DebateTurn(BaseModel):
    round: int
    peach: str
    dragon: str

class DebateResponse(BaseModel):
    timestamp: str
    topic: str
    persona_peach: str
    persona_dragon: str
    rounds: List[DebateTurn]
    judge: Optional[str] = None

# ---------- Routes ----------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "time": now_iso(),
        "openai_key_suffix": _suffix(OPENAI_API_KEY),
        "deepseek_key_suffix": _suffix(DEEPSEEK_API_KEY),
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # both personas answer the same user messages
    user_msgs = [m.dict() for m in req.messages]
    peach_text = _reply_as_persona(user_msgs, req.persona_peach, req.model, req.temperature)
    dragon_text = _reply_as_persona(user_msgs, req.persona_dragon, req.model, req.temperature)
    return ChatResponse(timestamp=now_iso(), peach=peach_text, dragon=dragon_text)

@app.post("/debate", response_model=DebateResponse)
def debate(req: DebateRequest) -> DebateResponse:
    topic = req.prompt.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Empty prompt")

    rounds_out: List[DebateTurn] = []
    last_peach = ""
    last_dragon = ""

    for i in range(1, req.rounds + 1):
        # Peach opens/responds
        peach_msgs = [
            {"role": "user", "content": f"Debate topic: {topic}"},
        ]
        if last_dragon:
            peach_msgs.append({"role": "user", "content": f"Your opponent just said: {last_dragon}"})
            peach_msgs.append({"role": "user", "content": "Respond directly. Be punchy but substantive."})
        else:
            peach_msgs.append({"role": "user", "content": "Open with your strongest position in 4-8 sentences."})

        last_peach = _reply_as_persona(peach_msgs, req.persona_peach, req.model, req.temperature)

        # Dragon replies
        dragon_msgs = [
            {"role": "user", "content": f"Debate topic: {topic}"},
            {"role": "user", "content": f"Your opponent just said: {last_peach}"},
            {"role": "user", "content": "Challenge or reframe. Be sharp, reasoned, and constructive (4-8 sentences)."},
        ]
        last_dragon = _reply_as_persona(dragon_msgs, req.persona_dragon, req.model, req.temperature)

        rounds_out.append(DebateTurn(round=i, peach=last_peach, dragon=last_dragon))

    # Optional judge wrap-up (neutral synthesis)
    judge_msg = [
        {"role": "user", "content": f"Summarize the debate on: {topic}"},
        {"role": "user", "content": f"Final Peach: {last_peach}"},
        {"role": "user", "content": f"Final Dragon: {last_dragon}"},
        {"role": "user", "content": "Offer a concise synthesis (3-6 sentences), highlight common ground, "
                                     "and suggest next steps for artists and communities."},
    ]
    judge_text = _reply_as_persona(judge_msg, "Philosophical Trickster", req.model, req.temperature)

    return DebateResponse(
        timestamp=now_iso(),
        topic=topic,
        persona_peach=req.persona_peach,
        persona_dragon=req.persona_dragon,
        rounds=rounds_out,
        judge=judge_text,
    )

# Local dev:
#   uvicorn main:app --reload


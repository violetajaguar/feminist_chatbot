import json
import traceback
import openai
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import config

OPENAI_API_KEY = config.OPENAI_API_KEY
# Initialize Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feminist_chatbot.log"),
        logging.StreamHandler()
    ]
)
logging.info("Feminist Chatbot logging initialized.")

app = FastAPI()

# Configure CORS (allow all origins for simplicity)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI API key from environment variable
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = config.OPENAI_API_KEY
if not openai.api_key:
    raise Exception("OpenAI API Key is not set. Please set the OPENAI_API_KEY environment variable.")

if not config.DEEPSEEK_API_KEY:
    raise Exception("DeepSeek API Key is not set. Please set the DEEPSEEK_API_KEY environment variable.")

# Define the request model
#client = openai.OpenAI(api_key=OPENAI_API_KEY)
class ChatRequest(BaseModel):
    message: str
    api: str

# Session memory to track the last selected personality
user_sessions = {}

# Helper function to log interactions
def log_interaction(user_input, ai_response):
    try:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response
        }
        with open("feminist_chatbot_logs.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"Failed to log interaction: {e}", exc_info=True)

# Function to get response for a specific personality
def get_personality_response(personality, expanded=False):
    info = config.feminist_personalities.get(personality)
    if not info:
        return "I'm not sure about that personality yet."
    response = info["description"]
    if expanded:
        response += " Would you like more details about my perspective?"
    return response

# Fallback using OpenAI GPT-4
def fallback_response(user_input,api_choice,selected_personality):
    system_prompt = (
        "You are FeministBot, a chatbot embodying five distinct feminist personalities: "
        "The Visionary Poet, the Radical Hacker, the Ancestral Wisdom Keeper, the Punk Riot Grrrl, and the Philosophical Trickster. "
        "Answer in a casual, cool, and artistic tone that celebrates feminist values and diverse creative expressions."
    )

    if selected_personality:
        system_prompt = f"You are now embodying the {selected_personality} personality, and should respond according to its style."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    if api_choice =="openai":
        try:
            #chat_completion = client.chat.completions.create
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            print(gpt_response)
            return gpt_response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logging.error("OpenAI API call failed", exc_info=True)
            return "Oops! I'm having trouble thinking right now. Please try again later."

    if api_choice == "deepseek":
        try:
            headers = {
                "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek-chat",  # Replace with the actual model name if necessary
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7
            }
            response = requests.post(config.DEEPSEEK_API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logging.error("DeepSeek API call failed", exc_info=True)
            return "Oops! I'm having trouble thinking right now. Please try again later."

# Chat endpoint handling requests
@app.post("/chat")
async def chat(chat_request: ChatRequest):
    try:
        user_id = "single_user"  # For simplicity; in production, use proper user IDs
        user_input_raw = chat_request.message.strip()
        user_input = user_input_raw.lower()
        #api_choice = chat_request.api_choice

        # Initialize session if not already set
        if user_id not in user_sessions:
            user_sessions[user_id] = {"last_personality": None}
        session = user_sessions[user_id]
        response = None

        # 1. Handle greetings
        greetings = ["hi", "hello", "hey", "hola"]
        if any(greet in user_input for greet in greetings):
            response = (
                "Hello, I'm FeministBot—your guide through five bold feminist voices. "
                "Which resonates with you today: the Visionary Poet, the Radical Hacker, the Ancestral Wisdom Keeper, the Punk Riot Grrrl, or the Philosophical Trickster?"
            )

        # # 2. Check if the user is asking for the New Feminist Frontiers context.
        # if not response and ("new feminist frontiers" in user_input or "facilitator notes" in user_input or "workshop1" in user_input):
        #     response = config.NEW_FEMINIST_FRONTIERS
        #
        # # 3. Check if the user is asking for the experimental context text.
        # if not response and ("experiment" in user_input or "ai impact" in user_input or "creative industries" in user_input):
        #     response = config.ARTICULATED_THOUGHTS
        #
        # # 4. Check if the user is asking for general feminist ideas context.
        # if not response and ("refig" in user_input or "feminist ideas" in user_input):
        #     response = config.FEMINIST_IDEAS
        #
        # 5. Expand on the last personality if the user says "more"
        if not response and "more" in user_input:
            last_personality = session.get("last_personality")
            if last_personality:
                response = get_personality_response(last_personality, expanded=True)
            else:
                response = "More about what? Please specify which personality intrigues you."

        # 6. Check for personality keywords in the user input

        if not response:
            for personality, data in config.feminist_personalities.items():
                for keyword in data["keywords"]:
                    if keyword in user_input:
                        response = get_personality_response(personality)
                        session["last_personality"] = personality
                        break
                if response:
                    break

        # 7. Fallback to OpenAI if nothing else matches
        if not response:
            selected_personality = session.get("last_personality")
            response = fallback_response(user_input_raw,chat_request.api,selected_personality)
            session["last_personality"] = None

        # Log the interaction
        log_interaction(user_input_raw, response)
        return {"response": response}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to FeministBot—a chatbot celebrating five distinct feminist voices."}
#
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = open("index.html", "r").read()
    return HTMLResponse(content=html_content)
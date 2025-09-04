#!/usr/bin/env python3
"""
Minimal web app (dark mode) to chat with two bots (Riya / Owen) using local sampling.
- Left: Start/Stop (lazy load/unload model on GPU)
- Right: Chat UI with tabs for Riya-bot and Owen-bot
- Optional steering: checkbox + positive prompt + negative prompts (comma-separated)

This does NOT use vLLM. It uses HF + PEFT locally and the project's sampling utilities.
"""

import os
import sys
import gc
import json
import asyncio
import signal
import atexit
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

from src.model_utils import (
    generate,
    generate_with_steering,
    generate_with_ppl,
    detect_model_type,
    truncate_to_next_speaker,
)
from src.steering.steer_utils import generate_steering_vector
from src.data_utils import clean_for_sampling
from src.config import MODEL_CONFIGS, RIYA_SPEAKER_TOKEN, FRIEND_SPEAKER_TOKEN

# ---------- Config ----------
DEFAULT_BASE_MODEL = os.environ.get("BASE_MODEL_ID", "mistralai/Mistral-7B-v0.1")
# Attempt to locate a LoRA adapter by default
DEFAULT_ADAPTER_CANDIDATES = [
    PROJECT_ROOT / "models" / "mistral-7b" / "mistral-results-7-27-25" / "lora_adapter",
    PROJECT_ROOT / "models" / "mistral-7b" / "mistral-results-7-6-25" / "lora_adapter",
    PROJECT_ROOT / "models" / "gemma-3-27b-it" / "gemma-3-27b-it_20250903_12225225" / "lora_adapter",
]
RIYA_NAME = os.environ.get("RIYA_NAME", "Riya")
OWEN_NAME = os.environ.get("FRIEND_NAME", "Owen")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "80"))
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
DTYPE = os.environ.get("DTYPE", "bfloat16")  # float16, bfloat16, float32

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Ngrok tunnel - persistent backend tunnel for vercel to connect to
ngrok_listener = None

async def create_ngrok_tunnel():
    """Create ngrok tunnel for public access using pyngrok"""
    global ngrok_listener
    
    try:
        from pyngrok import ngrok
        
        # Get auth token from environment variable
        auth_token = os.environ.get("NGROK_AUTH_TOKEN")
        if not auth_token:
            print("⚠️  NGROK_AUTH_TOKEN not set. Tunnel will be limited.")
        else:
            # Set the auth token for pyngrok
            ngrok.set_auth_token(auth_token)
            print("✅ ngrok auth token set")
        
        # Create HTTP tunnel to localhost:9100 using pyngrok
        print("🔌 Creating ngrok tunnel to localhost:9100...")
        # For pyngrok, we use connect() method with port and protocol
        ngrok_listener = ngrok.connect(9100, "http")
        
        # Get the public URL from the listener
        # In pyngrok, the tunnel object has a public_url attribute
        public_url = ngrok_listener.public_url
        print(f"🚀 ngrok tunnel created: {public_url}")
        print(f"📝 Update your Vercel config.js with: apiBase: '{public_url}'")
        
        return public_url
        
    except Exception as e:
        print(f"❌ Failed to create ngrok tunnel: {e}")
        return None

async def cleanup_ngrok():
    """Clean up ngrok tunnel using pyngrok"""
    global ngrok_listener
    
    if ngrok_listener:
        try:
            from pyngrok import ngrok
            
            # Get the URL before disconnecting
            tunnel_url = ngrok_listener.url()
            
            # Disconnect the specific tunnel
            ngrok.disconnect(tunnel_url)
            print(f"🔒 ngrok tunnel closed: {tunnel_url}")
            
            # Also kill any remaining ngrok processes
            ngrok.kill()
            print("🔒 ngrok processes killed")
            
        except Exception as e:
            print(f"⚠️  Error closing tunnel: {e}")
    
    ngrok_listener = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    if ngrok_listener:
        asyncio.run(cleanup_ngrok())
    sys.exit(0)

# Register cleanup handlers
atexit.register(lambda: asyncio.run(cleanup_ngrok()) if ngrok_listener else None)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# helpers
def truncate_to_next_speaker(text: str, expected_stop: str, other_stop: str) -> str:
    """Cut model output at first sign of the next speaker.
    Looks for multiple patterns: [Name], [Name], Name:, Name : (case-insensitive), and newlines.
    """
    candidates: List[int] = []
    lowers = text.lower()
    candidates.append(text.find(expected_stop))
    candidates.append(text.find(other_stop))
    for token in [
        f"{expected_stop.strip('[]')}:",
        f"{expected_stop.strip('[]')} :",
        f"{other_stop.strip('[]')}:",
        f"{other_stop.strip('[]')} :",
    ]:
        i = lowers.find(token.lower())
        candidates.append(i)
    i_colon = text.find(" : ")
    candidates.append(i_colon)
    candidates.append(text.find("\n["))
    cut = min([i for i in candidates if i is not None and i >= 0], default=-1)
    if cut >= 0:
        return text[:cut]
    return text

# runtime state
class ChatSession:
    def __init__(self, bot_name: str) -> None:
        self.bot_name = bot_name
        self.history: List[str] = []
        self.turns: int = 0
        self.transcript: List[Dict[str, str]] = []

class ModelManager:
    def __init__(self) -> None:
        self.base_model_id: str = DEFAULT_BASE_MODEL
        self.adapter_path: Optional[Path] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[torch.nn.Module] = None
        self.loaded: bool = False
        self.model_type: str = "base"  # "base" or "instruct"
        self.model_key: str = "mistral-7b"  # Default model key

    def find_default_adapter(self) -> Optional[Path]:
        # First try the hardcoded candidates
        for cand in DEFAULT_ADAPTER_CANDIDATES:
            if cand.exists():
                return cand
        
        # Then search more broadly in the models directory
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            # Look for any lora_adapter directories
            for p in models_dir.rglob("lora_adapter"):
                if p.exists() and p.is_dir():
                    print(f"Found adapter: {p}")
                    return p
        
        print("No LoRA adapters found in default locations")
        return None

    def detect_model_key_from_adapter(self, adapter_path: Path) -> str:
        """Detect which model key this adapter belongs to by checking the path."""
        adapter_str = str(adapter_path)
        print(f"Detecting model key from adapter path: {adapter_str}")
        
        for model_key in MODEL_CONFIGS.keys():
            if model_key in adapter_str:
                print(f"Detected model key: {model_key}")
                return model_key
        
        print(f"Could not detect model key, using default: mistral-7b")
        return "mistral-7b"  # Default fallback

    def start(self, adapter_path: Optional[str] = None) -> Dict[str, str]:
        if self.loaded:
            return {"status": "already_running"}
        
        if adapter_path:
            # Construct full path from relative path in models directory
            self.adapter_path = PROJECT_ROOT / "models" / adapter_path
            print(f"Using specified adapter: {self.adapter_path}")
        else:
            self.adapter_path = self.find_default_adapter()
            print(f"Using auto-detected adapter: {self.adapter_path}")
        
        if not self.adapter_path or not self.adapter_path.exists():
            print(f"Adapter path not found: {self.adapter_path}")
            raise HTTPException(status_code=400, detail="LoRA adapter not found. Provide a valid adapter path.")

        # Detect model type and key from adapter path
        self.model_key = self.detect_model_key_from_adapter(self.adapter_path)
        self.model_type = detect_model_type(self.model_key)
        
        # Update base model ID based on detected model key
        if self.model_key in MODEL_CONFIGS:
            self.base_model_id = MODEL_CONFIGS[self.model_key]["model_name"]

        torch_dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map=DEVICE_MAP,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if base.get_input_embeddings().num_embeddings != len(self.tokenizer):
            base.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        self.model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self.model.eval()
        self.loaded = True
        return {
            "status": "started", 
            "base": self.base_model_id, 
            "adapter": str(self.adapter_path),
            "model_type": self.model_type,
            "model_key": self.model_key
        }

    def stop(self) -> Dict[str, str]:
        out_dir = PROJECT_ROOT / "convos" / "public"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for name, sess in sessions.items():
            if sess.transcript:
                fname = out_dir / f"{timestamp}_{name}.txt"
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(f"Session: {name}\nTime: {timestamp}\nBase: {self.base_model_id}\nAdapter: {self.adapter_path}\n\n")
                    for m in sess.transcript:
                        f.write(f"{m['role']}: {m['content']}\n")
        for s in sessions.values():
            s.transcript.clear()

        if not self.loaded:
            return {"status": "already_stopped"}
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        self.model = None
        self.tokenizer = None
        self.loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        return {"status": "stopped"}

    def ensure_loaded(self) -> None:
        if not self.loaded or not self.model or not self.tokenizer:
            raise HTTPException(status_code=503, detail="Model not loaded. Click Start first.")

model_manager = ModelManager()
sessions: Dict[str, ChatSession] = {
    "riya": ChatSession(RIYA_NAME),
    "owen": ChatSession(OWEN_NAME),
}

app = FastAPI(title="Conversation Web App", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> Dict[str, str]:
    response = {"ok": "true", "loaded": str(model_manager.loaded)}
    if model_manager.loaded:
        response.update({
            "model_type": model_manager.model_type,
            "model_key": model_manager.model_key,
            "base_model": model_manager.base_model_id,
            "adapter": str(model_manager.adapter_path) if model_manager.adapter_path else None
        })
    return response

@app.post("/start")
def start(payload: Dict[str, str] = None):
    """Legacy endpoint - models are now auto-loaded on first chat"""
    return {"status": "deprecated", "message": "Models are now auto-loaded on first chat"}

@app.post("/cleanup")
def cleanup():
    """Clean up model from memory and clear CUDA cache."""
    for s in sessions.values():
        s.history.clear()
        s.turns = 0
    return model_manager.stop()

@app.post("/stop")
def stop():
    """Legacy endpoint - redirects to cleanup"""
    return cleanup()

@app.post("/chat")
def chat(payload: Dict) -> Dict:
    bot = payload.get("bot")
    message = payload.get("message", "").strip()
    steering = payload.get("steering", {}) or {}
    adapter = payload.get("adapter", "")  # Get adapter from payload
    
    if bot not in sessions:
        raise HTTPException(status_code=400, detail="Invalid bot. Use 'riya' or 'owen'.")
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")
    
    # Auto-load model if not loaded, or reload if adapter changed
    if not model_manager.loaded or (adapter and adapter != str(model_manager.adapter_path)):
        try:
            model_manager.start(adapter if adapter else None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    session = sessions[bot]

    speaker_user = OWEN_NAME if bot == "riya" else RIYA_NAME
    session.transcript.append({"role": speaker_user, "content": message})

    if bot == "riya":
        prompt_line = f"\n[{OWEN_NAME}]: {message} \n [{RIYA_NAME}]"
        expected_stop = f"[{OWEN_NAME}]"
        other_stop = f"[{RIYA_NAME}]"
        assistant_name = RIYA_NAME
    else:
        prompt_line = f"\n[{RIYA_NAME}]: {message} \n [{OWEN_NAME}]"
        expected_stop = f"[{RIYA_NAME}]"
        other_stop = f"[{OWEN_NAME}]"
        assistant_name = OWEN_NAME

    if session.turns > 8 and session.history:
        session.history.pop(0)
    session.history.append(prompt_line)
    session.turns += 1

    full_prompt = "".join(session.history)

    use_steering = bool(steering.get("enabled"))
    steering_pairs = steering.get("pairs", [])
    extract_layer = steering.get("extract_layer", -2)
    apply_layer = steering.get("apply_layer", -1)
    alpha_strength = steering.get("alpha", 2.0)

    max_new = int(payload.get("max_new_tokens") or MAX_NEW_TOKENS)

    # Use unified generation functions that handle both model types
    is_instruct = model_manager.model_type == "instruct"
    target_speaker = assistant_name if is_instruct else None
    
    if use_steering and steering_pairs:
        weights = {}
        for pair in steering_pairs:
            pos = pair.get("positive", "").strip()
            neg = pair.get("negative", "").strip()
            if pos and neg:  # Only add non-empty pairs to weights
                weights[pos] = alpha_strength
                weights[neg] = -alpha_strength
        
        # generate_steering_vector processes all non-empty pairs together to create a comprehensive steering vector
        # It combines the effects of all contrastive pairs into a single steering vector
        if weights:  # Only proceed if we have valid pairs
            steering_vector = generate_steering_vector(model_manager.model, model_manager.tokenizer, weights,
                                                       pos_alpha=alpha_strength, neg_alpha=alpha_strength, layer_from_last=extract_layer)
            raw = generate_with_steering(
                model_manager.model, full_prompt, model_manager.tokenizer,
                steering_vector, max_new_tokens=max_new, layer_from_last=apply_layer,
                is_instruct=is_instruct, target_speaker=target_speaker
            )
        else:
            # No valid pairs, fall back to regular generation
            raw = generate(
                model_manager.model, full_prompt, model_manager.tokenizer,
                max_new_tokens=max_new,
                is_instruct=is_instruct, target_speaker=target_speaker
            )
    else:
        # No steering enabled, use regular generation
        raw = generate(
            model_manager.model, full_prompt, model_manager.tokenizer,
            max_new_tokens=max_new,
            is_instruct=is_instruct, target_speaker=target_speaker
        )

    # Handle text processing based on model type
    if is_instruct:
        # For instruct models, the response already includes the speaker token
        text = raw.strip()
        text = clean_for_sampling(text)
    else:
        # For base models, truncate and clean up
        text = truncate_to_next_speaker(raw, expected_stop=expected_stop, other_stop=other_stop)
        text = text.lstrip(" :")
        text = clean_for_sampling(text)

    session.history.append(text)
    session.turns += 1
    session.transcript.append({"role": assistant_name, "content": text})

    return {"response": text}

@app.get("/")
def index() -> HTMLResponse:
    html = _INDEX_HTML_TEMPLATE.replace("__RIYA__", RIYA_NAME) \
                               .replace("__OWEN__", OWEN_NAME) \
                               .replace("__MAX_NEW__", str(MAX_NEW_TOKENS))
    return HTMLResponse(content=html)

_INDEX_HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Conversation</title>
  <style>
    :root {
      --bg: #0b0b0f;
      --card: #121218;
      --muted: #1a1a22;
      --text: #e9e9ef;
      --subtext: #a3a3b2;
      --accent: #7c5cff;
      --accent2: #28d1b6;
      --danger: #ff4d67;
      --ok: #3ddc97;
      --border: #2a2a36;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji"; }
    .wrap { display: grid; grid-template-columns: 280px 1fr; gap: 16px; padding: 20px; min-height: 100vh; }
    .panel { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }
    .title { font-weight: 700; font-size: 18px; margin-bottom: 12px; }
    button { background: var(--muted); color: var(--text); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; cursor: pointer; font-weight: 600; }
    button.primary { background: linear-gradient(135deg, var(--accent), #5243ff); border: none; }
    button.danger { background: linear-gradient(135deg, var(--danger), #ff2750); border: none; }
    button:disabled { opacity: .6; cursor: not-allowed; }
    .btns { display: grid; gap: 10px; }
    .status { margin-top: 12px; color: var(--subtext); font-size: 14px; }

    .tabs { display: flex; gap: 8px; margin-bottom: 10px; }
    .tab { padding: 8px 12px; border-radius: 10px; background: var(--muted); color: var(--text); cursor: pointer; border: 1px solid var(--border); }
    .tab.active { background: var(--accent); }

    .chat { display: grid; grid-template-rows: auto 1fr auto; height: calc(100vh - 40px); gap: 12px; }
    .log { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 14px; overflow-y: auto; min-height: 200px; }
    .msg { margin: 6px 0; }
    .me { color: var(--accent2); }
    .bot { color: var(--text); }

    .row { display: grid; grid-template-columns: 1fr auto; gap: 10px; }
    textarea { width: 100%; height: 90px; background: var(--muted); color: var(--text); border: 1px solid var(--border); border-radius: 10px; padding: 10px; resize: vertical; }

    .steer { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 10px; margin-bottom: 8px; display: grid; gap: 8px; }
    .steer input[type="text"] { width: 100%; background: var(--muted); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 8px; }
    label { color: var(--subtext); font-size: 14px; }
    
    .steering-pairs { display: grid; gap: 8px; }
    .steering-pairs textarea { 
      width: 100%; 
      background: var(--muted); 
      color: var(--text); 
      border: 1px solid var(--border); 
      border-radius: 8px; 
      padding: 8px; 
      resize: vertical; 
      font-family: monospace;
      font-size: 13px;
    }
    
    .steering-controls { display: grid; gap: 8px; }
    .control-group { 
      display: grid; 
      grid-template-columns: 1fr auto; 
      gap: 8px; 
      align-items: center; 
    }
    .control-group input[type="range"] { 
      width: 100%; 
      background: var(--muted); 
      height: 6px; 
      border-radius: 3px; 
      outline: none; 
    }
    .control-group span { 
      color: var(--accent); 
      font-weight: 600; 
      min-width: 30px; 
      text-align: right; 
    }
    
    .adapter-select { 
      background: var(--muted); 
      border: 1px solid var(--border); 
      border-radius: 10px; 
      padding: 10px; 
      margin-bottom: 8px; 
      display: grid; 
      gap: 6px; 
    }
    .adapter-select select { 
      width: 100%; 
      background: var(--card); 
      color: var(--text); 
      border: 1px solid var(--border); 
      border-radius: 8px; 
      padding: 8px; 
    }
    .adapter-select small { 
      display: block; 
      margin-top: 4px; 
      line-height: 1.3; 
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <div class="title">Controls</div>
      <div class="btns">
        <button id="cleanup" class="primary">Offload Models</button>
      </div>
      <div class="adapter-select">
        <label>Adapter:</label>
        <select id="adapterSelect">
          <option value="mistral-7b/mistral-results-7-27-25/lora_adapter">Mistral 7B (7/27/25)</option>
          <option value="mistral-7b/mistral-results-7-6-25/lora_adapter">Mistral 7B (7/6/25)</option>
          <option value="gemma-3-27b-it/gemma-3-27b-it_20250903_12225225/lora_adapter">Gemma 3 27B IT (9/3/25)</option>
        </select>
      </div>
      <div id="status" class="status">Model: stopped</div>
      <div class="steer">
        <label><input type="checkbox" id="steerEnabled" /> Enable steering</label>
        
        <div class="steering-pairs">
          <label>Contrast pairs (one per line, comma-separated)</label>
          <textarea id="contrastPairs" placeholder="happy, sad&#10;today is a good day, today is a bad day&#10;wholesome, toxic" rows="4"></textarea>
        </div>
        
        <div class="steering-controls">
          <div class="control-group">
            <label>Extraction layer (from last)</label>
            <input id="extractLayer" type="range" min="-10" max="-1" value="-2" />
            <span id="extractLayerValue">-2</span>
          </div>
          <div class="control-group">
            <label>Application layer (from last)</label>
            <input id="applyLayer" type="range" min="-10" max="-1" value="-1" />
            <span id="applyLayerValue">-1</span>
          </div>
          <div class="control-group">
            <label>Alpha strength</label>
            <input id="alphaStrength" type="range" min="0.1" max="5.0" step="0.1" value="2.0" />
            <span id="alphaValue">2.0</span>
          </div>
        </div>
      </div>
    </div>

    <div class="panel chat">
      <div>
        <div class="tabs">
          <div class="tab active" id="tab-riya">__RIYA__-bot</div>
          <div class="tab" id="tab-owen">__OWEN__-bot</div>
        </div>
      </div>

      <div id="log-riya" class="log"></div>
      <div id="log-owen" class="log" style="display:none"></div>

      <div>
        <div class="row">
          <textarea id="input" placeholder="Type a message..."></textarea>
          <button id="send" class="primary">Send</button>
        </div>
      </div>
    </div>
  </div>

    <script>
    let bot = 'riya';
    const logRiya = document.getElementById('log-riya');
    const logOwen = document.getElementById('logOwen');

    // Maintain separate message arrays; render only current bot on tab switch
    const msgs = { riya: [], owen: [] };

    function render() {
      const target = bot === 'riya' ? logRiya : logOwen;
      target.innerHTML = '';
      for (const m of msgs[bot]) {
        const d = document.createElement('div');
        d.className = 'msg ' + m.role;
        d.textContent = m.name + ': ' + m.text;
        target.appendChild(d);
      }
      target.scrollTop = target.scrollHeight;
    }

    function currentLog() { return bot === 'riya' ? logRiya : logOwen; }

    function add(role, text) {
      const name = role === 'me' ? 'You' : (bot === 'riya' ? '__RIYA__' : '__OWEN__');
      msgs[bot].push({ role, text, name });
      render();
    }

    async function call(path, payload) {
      const r = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {})
      });
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    }

    async function refreshStatus() {
      const r = await fetch('/health');
      const j = await r.json();
      let statusText = 'Model: ' + (j.loaded === 'True' || j.loaded === true ? 'running' : 'stopped');
      
      // If we have model info, show it
      if (j.loaded === 'True' || j.loaded === true) {
        if (j.model_type) {
          statusText += ` (${j.model_type})`;
        }
        if (j.model_key) {
          statusText += ` - ${j.model_key}`;
        }
      } else {
        statusText += ' - will auto-load on first chat';
      }
      
      document.getElementById('status').textContent = statusText;
    }

    // Set up event handlers when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
      // Cleanup button
      document.getElementById('cleanup').onclick = async () => {
        document.getElementById('status').textContent = 'Cleaning up...';
        try {
          await call('/cleanup', {});
          document.getElementById('status').textContent = 'Model: stopped (cleaned up)';
        } catch (e) { 
          alert('Cleanup failed: ' + e.message); 
          refreshStatus();
        }
      };

      // Initialize the app
      refreshStatus();
    });

    // Tab switching
    document.getElementById('tab-riya').onclick = () => {
      bot = 'riya';
      document.getElementById('tab-riya').classList.add('active');
      document.getElementById('tab-owen').classList.remove('active');
      logRiya.style.display = '';
      logOwen.style.display = 'none';
      document.getElementById('input').value = '';
      render();
    };
    
    document.getElementById('tab-owen').onclick = () => {
      bot = 'owen';
      document.getElementById('tab-owen').classList.add('active');
      document.getElementById('tab-riya').classList.remove('active');
      logOwen.style.display = '';
      logRiya.style.display = 'none';
      document.getElementById('input').value = '';
      render();
    };

    async function send() {
      const t = document.getElementById('input');
      const text = t.value.trim(); if (!text) return;
      add('me', text); t.value = '';
      
      const steerEnabled = document.getElementById('steerEnabled').checked;
      const selectedAdapter = document.getElementById('adapterSelect').value;
      
      // Parse contrast pairs from textarea
      const contrastPairsText = document.getElementById('contrastPairs').value.trim();
      const steeringPairs = [];
      
      if (contrastPairsText) {
        const lines = contrastPairsText.split('\n');
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine) {
            const parts = trimmedLine.split(',').map(part => part.trim());
            if (parts.length === 2 && parts[0] && parts[1]) {
              steeringPairs.push({ positive: parts[0], negative: parts[1] });
            }
          }
        }
      }
      
      // Collect steering controls
      const extractLayer = parseInt(document.getElementById('extractLayer').value);
      const applyLayer = parseInt(document.getElementById('applyLayer').value);
      const alphaStrength = parseFloat(document.getElementById('alphaStrength').value);
      
      try {
        const r = await call('/chat', {
          bot,
          message: text,
          adapter: selectedAdapter,  // Always send the selected adapter
          steering: { 
            enabled: steerEnabled, 
            pairs: steeringPairs,
            extract_layer: extractLayer,
            apply_layer: applyLayer,
            alpha: alphaStrength
          },
          max_new_tokens: __MAX_NEW__
        });
        msgs[bot].push({ role: 'bot', text: r.response, name: (bot === 'riya' ? '__RIYA__' : '__OWEN__') });
        render();
      } catch (e) {
        msgs[bot].push({ role: 'bot', text: 'Error: ' + e.message, name: (bot === 'riya' ? '__RIYA__' : '__OWEN__') });
        render();
      }
    }
    
    // Send button and input handling
    document.getElementById('send').onclick = send;
    document.getElementById('input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });
    
    // Handle range slider updates
    document.getElementById('extractLayer').addEventListener('input', function(e) {
      document.getElementById('extractLayerValue').textContent = e.target.value;
    });
    document.getElementById('applyLayer').addEventListener('input', function(e) {
      document.getElementById('applyLayerValue').textContent = e.target.value;
    });
    document.getElementById('alphaStrength').addEventListener('input', function(e) {
      document.getElementById('alphaValue').textContent = e.target.value;
    });
    
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("🚀 Starting Conversation App...")
    print("🌐 Starting FastAPI server on localhost:9100...")
    uvicorn.run("src.deployment.app:app", host="0.0.0.0", port=9100, reload=False)

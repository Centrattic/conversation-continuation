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
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

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
)
from src.steering.steer_utils import generate_steering_vector
from src.data_utils import clean_for_sampling

# ---------- Config ----------
DEFAULT_BASE_MODEL = os.environ.get("BASE_MODEL_ID", "mistralai/Mistral-7B-v0.1")
# Attempt to locate a LoRA adapter by default
DEFAULT_ADAPTER_CANDIDATES = [
    PROJECT_ROOT / "models" / "mistral-7b" / "mistral-results-7-27-25" / "lora_train" / "lora_adapter",
    PROJECT_ROOT / "models" / "mistral-7b" / "mistral-results-7-6-25" / "lora_train" / "lora_adapter",
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

# ---------- Runtime State ----------
class ChatSession:
    def __init__(self, bot_name: str) -> None:
        self.bot_name = bot_name
        self.history: List[str] = []
        self.turns: int = 0

class ModelManager:
    def __init__(self) -> None:
        self.base_model_id: str = DEFAULT_BASE_MODEL
        self.adapter_path: Optional[Path] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[torch.nn.Module] = None
        self.loaded: bool = False

    def find_default_adapter(self) -> Optional[Path]:
        for cand in DEFAULT_ADAPTER_CANDIDATES:
            if cand.exists():
                return cand
        # Fallback scan
        for p in (PROJECT_ROOT / "models").rglob("lora_adapter"):
            return p
        return None

    def start(self, adapter_path: Optional[str] = None) -> Dict[str, str]:
        if self.loaded:
            return {"status": "already_running"}
        self.adapter_path = Path(adapter_path) if adapter_path else self.find_default_adapter()
        if not self.adapter_path or not self.adapter_path.exists():
            raise HTTPException(status_code=400, detail="LoRA adapter not found. Provide a valid adapter path.")

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
        # Ensure vocab resize if needed
        if base.get_input_embeddings().num_embeddings != len(self.tokenizer):
            base.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        self.model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self.model.eval()
        self.loaded = True
        return {"status": "started", "base": self.base_model_id, "adapter": str(self.adapter_path)}

    def stop(self) -> Dict[str, str]:
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

# ---------- Web App ----------
app = FastAPI(title="Conversation Web App", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"ok": "true", "loaded": str(model_manager.loaded)}

@app.post("/start")
def start(payload: Dict[str, str] = None):
    adapter = (payload or {}).get("adapter") if payload else None
    return model_manager.start(adapter)

@app.post("/stop")
def stop():
    # Reset sessions too
    for s in sessions.values():
        s.history.clear()
        s.turns = 0
    return model_manager.stop()

@app.post("/chat")
def chat(payload: Dict) -> Dict:
    model_manager.ensure_loaded()
    bot = payload.get("bot")  # "riya" or "owen"
    message = payload.get("message", "").strip()
    steering = payload.get("steering", {}) or {}
    
    if bot not in sessions:
        raise HTTPException(status_code=400, detail="Invalid bot. Use 'riya' or 'owen'.")
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")

    session = sessions[bot]

    # Format prompt consistent with sampling scripts
    if bot == "riya":
        # User plays FRIEND, get RIYA's reply
        prompt_line = f"\n[{OWEN_NAME}]: {message} \n [{RIYA_NAME}]"
        expected_stop = f"[{OWEN_NAME}]"
    else:
        # bot == "owen": user plays RIYA, get Owen's reply
        prompt_line = f"\n[{RIYA_NAME}]: {message} \n [{OWEN_NAME}]"
        expected_stop = f"[{RIYA_NAME}]"

    # Maintain a short rolling history (like original scripts)
    if session.turns > 8 and session.history:
        session.history.pop(0)
    session.history.append(prompt_line)
    session.turns += 1

    full_prompt = "".join(session.history)

    # Steering
    use_steering = bool(steering.get("enabled"))
    pos = (steering.get("positive") or "").strip()
    neg_csv = (steering.get("negative_csv") or "").strip()
    neg_list = [t.strip() for t in neg_csv.split(",") if t.strip()]

    max_new = int(payload.get("max_new_tokens") or MAX_NEW_TOKENS)

    if use_steering and (pos or neg_list):
        # Build a weighting dict: positive +1, negatives -1
        weights = {}
        if pos:
            weights[pos] = 1.0
        for n in neg_list:
            weights[n] = -1.0
        steering_vector = generate_steering_vector(model_manager.model, model_manager.tokenizer, weights,
                                                   pos_alpha=2.0, neg_alpha=2.0, layer_from_last=-2)
        text = generate_with_steering(
            model_manager.model, full_prompt, model_manager.tokenizer,
            steering_vector, max_new_tokens=max_new, layer_from_last=-1
        )
    else:
        # Default path: use perplexity-aware generator for variety
        text, _ = generate_with_ppl(
            model_manager.model, full_prompt, model_manager.tokenizer, max_new_tokens=max_new
        )

    # Truncate at the next speaker tag
    idx = text.find(expected_stop)
    if idx == -1:
        idx = text.find(f"[{expected_stop[1:2]}")  # crude fallback like original
    if idx == -1:
        idx = len(text)
    text = text[:idx]

    text = clean_for_sampling(text)

    # Push back into history
    session.history.append(text)
    session.turns += 1

    return {"response": text}

@app.get("/")
def index() -> HTMLResponse:
    html = _INDEX_HTML_TEMPLATE.replace("__RIYA__", RIYA_NAME) \
                               .replace("__OWEN__", OWEN_NAME) \
                               .replace("__MAX_NEW__", str(MAX_NEW_TOKENS))
    return HTMLResponse(content=html)

# ---------- Dark UI ----------
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

    /* Right side */
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

    .steer { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 10px; margin-bottom: 8px; display: grid; gap: 6px; }
    .steer input[type="text"] { width: 100%; background: var(--muted); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 8px; }
    label { color: var(--subtext); font-size: 14px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <div class="title">Controls</div>
      <div class="btns">
        <button id="start" class="primary">Start Server</button>
        <button id="stop" class="danger">Stop Server</button>
      </div>
      <div id="status" class="status">Model: stopped</div>
      <div class="steer">
        <label><input type="checkbox" id="steerEnabled" /> Enable steering</label>
        <label>Positive prompt</label>
        <input id="pos" type="text" placeholder="e.g. wholesome, helpful" />
        <label>Negative prompts (comma-separated)</label>
        <input id="neg" type="text" placeholder="e.g. toxic, rude" />
      </div>
    </div>

    <div class="panel chat">
      <div>
        <div class="tabs">
          <div class="tab active" id="tab-riya">__RIYA__-bot</div>
          <div class="tab" id="tab-owen">__OWEN__-bot</div>
        </div>
      </div>

      <div id="log" class="log"></div>

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
    const log = document.getElementById('log');

    function add(role, text) {
      const d = document.createElement('div');
      d.className = 'msg ' + role;
      d.textContent = (role === 'me' ? 'You: ' : (bot === 'riya' ? '__RIYA__' : '__OWEN__') + ': ') + text;
      log.appendChild(d);
      log.scrollTop = log.scrollHeight;
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
      document.getElementById('status').textContent = 'Model: ' + (j.loaded === 'True' || j.loaded === true ? 'running' : 'stopped');
    }

    document.getElementById('start').onclick = async () => {
      document.getElementById('status').textContent = 'Starting...';
      try {
        await call('/start', {});
      } catch (e) { alert('Start failed: ' + e.message); }
      refreshStatus();
    };
    document.getElementById('stop').onclick = async () => {
      document.getElementById('status').textContent = 'Stopping...';
      try { await call('/stop', {}); } catch (e) {}
      refreshStatus();
    };

    document.getElementById('tab-riya').onclick = () => {
      bot = 'riya';
      document.getElementById('tab-riya').classList.add('active');
      document.getElementById('tab-owen').classList.remove('active');
    };
    document.getElementById('tab-owen').onclick = () => {
      bot = 'owen';
      document.getElementById('tab-owen').classList.add('active');
      document.getElementById('tab-riya').classList.remove('active');
    };

    async function send() {
      const t = document.getElementById('input');
      const text = t.value.trim(); if (!text) return;
      add('me', text); t.value = '';
      const steerEnabled = document.getElementById('steerEnabled').checked;
      const pos = document.getElementById('pos').value;
      const neg = document.getElementById('neg').value;
      try {
        const r = await call('/chat', {
          bot,
          message: text,
          steering: { enabled: steerEnabled, positive: pos, negative_csv: neg },
          max_new_tokens: __MAX_NEW__
        });
        add('bot', r.response);
      } catch (e) {
        add('bot', 'Error: ' + e.message);
      }
    }
    document.getElementById('send').onclick = send;
    document.getElementById('input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });

    refreshStatus();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run("src.deployment.app:app", host="0.0.0.0", port=9100, reload=False)

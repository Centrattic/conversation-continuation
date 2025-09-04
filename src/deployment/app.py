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
from fastapi import FastAPI, HTTPException, Depends, Header
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
# Adapter paths will be read from config.js to ensure frontend/backend sync
def get_adapter_candidates_from_config():
    """Read adapter paths from config.js to ensure frontend/backend sync"""
    config_path = PROJECT_ROOT / "src" / "deployment" / "vercel-frontend" / "config.js"
    if not config_path.exists():
        print("⚠️  config.js not found, using fallback adapter paths")
        return [
            PROJECT_ROOT / "models" / "mistral-7b" / "mistral-results-7-6-25" / "lora_adapter",
        ]
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Extract adapter paths from config.js
        import re
        adapter_matches = re.findall(r"value:\s*['\"]([^'\"]+)['\"]", content)
        
        if adapter_matches:
            return [PROJECT_ROOT / "models" / path for path in adapter_matches]
        else:
            print("⚠️  No adapter paths found in config.js, using fallback")
            return [
                PROJECT_ROOT / "models" / "mistral-7b" / "mistral-results-7-6-25" / "lora_adapter",
            ]
    except Exception as e:
        print(f"⚠️  Error reading config.js: {e}, using fallback adapter paths")
        return [
            PROJECT_ROOT / "models" / "mistral-7b" / "mistral-results-7-6-25" / "lora_adapter",
        ]

# Get adapter candidates from config.js
DEFAULT_ADAPTER_CANDIDATES = get_adapter_candidates_from_config()
RIYA_NAME = os.environ.get("RIYA_NAME", "Riya")
OWEN_NAME = os.environ.get("FRIEND_NAME", "Owen")

# Security - Restrict to your Vercel domain
ALLOWED_ORIGINS = [
    "https://conversation-continuation.vercel.app",  # Remove trailing slash
    "http://localhost:9100",  # For local development
]

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
        
        # Try to use ngrok config file for persistent settings
        config_path = PROJECT_ROOT / "src" / "deployment" / "ngrok.yml"
        if config_path.exists():
            print("📁 Using ngrok configuration file...")
            # For pyngrok, we can specify the config file
            ngrok_listener = ngrok.connect(9100, "http", config_file=str(config_path))
        else:
            print("🔌 Creating basic ngrok tunnel to localhost:9100...")
            ngrok_listener = ngrok.connect(9100, "http")
        
        # Get the public URL from the listener
        public_url = ngrok_listener.public_url
        print(f"🚀 ngrok tunnel created: {public_url}")
        
        # Auto-update Vercel config file
        vercel_config_path = PROJECT_ROOT / "src" / "deployment" / "vercel-frontend" / "config.js"
        if vercel_config_path.exists():
            try:
                with open(vercel_config_path, 'r') as f:
                    content = f.read()
                
                # Update the apiBase URL
                import re
                new_content = re.sub(
                    r"apiBase:\s*['\"][^'\"]*['\"]",
                    f"apiBase: '{public_url}'",
                    content
                )
                
                with open(vercel_config_path, 'w') as f:
                    f.write(new_content)
                
                print(f"✅ Auto-updated Vercel config.js with new ngrok URL")
            except Exception as e:
                print(f"⚠️  Could not auto-update Vercel config: {e}")
        
        print(f"📝 Your Vercel config.js has been updated with: apiBase: '{public_url}'")
        
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
            tunnel_url = ngrok_listener.public_url
            
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

# Security helpers
async def verify_origin(origin: str = Header(None), referer: str = Header(None)):
    """Verify request origin is from allowed domains"""
    # Check Origin header first, then Referer as fallback
    request_origin = origin or referer
    
    if not request_origin:
        # Allow requests without origin (like from Postman/testing)
        return True
    
    # Extract domain from origin/referer
    for allowed_origin in ALLOWED_ORIGINS:
        if allowed_origin in request_origin:
            return True
    
    raise HTTPException(status_code=403, detail="Origin not allowed")

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
        # Only try the hardcoded candidates
        for cand in DEFAULT_ADAPTER_CANDIDATES:
            if cand.exists():
                return cand
        
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
        
        # Warm up the model to load checkpoint shards and avoid slow first inference
        print("🔥 Warming up model to preload checkpoint shards...")
        warmup_success = False
        try:
            with torch.no_grad():
                # Create a dummy input and ensure it's on the same device as the model
                # For multi-GPU models, we need to be more careful about device placement
                if hasattr(self.model, 'device'):
                    device = self.model.device
                else:
                    # Get device from the first parameter
                    device = next(self.model.parameters()).device
                
                print(f"🔧 Warming up on device: {device}")
                dummy_input = torch.randint(0, len(self.tokenizer), (1, 10), device=device)
                _ = self.model(dummy_input)
                warmup_success = True
                print("✅ Model warmed up successfully - checkpoint shards loaded")
        except Exception as e:
            print(f"⚠️  Model warmup failed: {e}")
            print("⚠️  This is non-critical - model will still work but first inference may be slow")
        
        if warmup_success:
            print("🚀 Model fully loaded and warmed up - ready for fast inference!")
        else:
            print("⚠️  Model loaded but warmup failed - first inference may be slow")
        
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
def cleanup(origin: str = Depends(verify_origin)):
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
def chat(payload: Dict, origin: str = Depends(verify_origin)) -> Dict:
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
    """Redirect to the Vercel frontend"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Conversation App</title>
        <meta http-equiv="refresh" content="0; url=https://conversation-continuation.vercel.app">
    </head>
    <body>
        <p>Redirecting to <a href="https://conversation-continuation.vercel.app">Vercel Frontend</a>...</p>
    </body>
    </html>
    """)

if __name__ == "__main__":
    print("🚀 Starting Conversation App...")
    
    # Start ngrok tunnel
    async def start_with_ngrok():
        print("🔌 Creating ngrok tunnel...")
        tunnel_url = await create_ngrok_tunnel()
        if tunnel_url:
            print(f"✅ ngrok tunnel ready: {tunnel_url}")
        else:
            print("⚠️  ngrok tunnel creation failed, continuing without tunnel")
        
        print("🌐 Starting FastAPI server on localhost:9100...")
        config = uvicorn.Config("src.deployment.app:app", host="0.0.0.0", port=9100, reload=False)
        server = uvicorn.Server(config)
        await server.serve()
    
    # Run the async startup
    asyncio.run(start_with_ngrok())

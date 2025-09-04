#!/usr/bin/env python3
"""
Minimal web app (dark mode) to chat with two bots (Riya / Owen) using local sampling.
- Left: Start/Stop (lazy load/unload model on GPU)
- Right: Chat UI with tabs for Riya-bot and Owen-bot
- Optional steering: checkbox + positive prompt + negative prompts (comma-separated)

This does NOT use vLLM. It uses HF + PEFT locally and the project's sampling utilities.

Large models (like Gemma-27B) are automatically quantized to 4-bit and loaded on a single GPU
to avoid device mismatch issues and reduce memory usage.
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
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import torch
from unsloth import FastLanguageModel

from src.model_utils import (
    generate,
    generate_with_steering,
    detect_model_type,
    get_stop_tokens_for_speaker,
)
from src.steering.steer_utils import generate_steering_vector
from src.data_utils import clean_for_sampling
from src.config import MODEL_CONFIGS, RIYA_SPEAKER_TOKEN, FRIEND_SPEAKER_TOKEN


# Adapter paths will be read from config.js to ensure frontend/backend sync
def get_adapter_candidates_from_config():
    """Read adapter paths from config.js to ensure frontend/backend sync"""
    config_path = PROJECT_ROOT / "src" / "deployment" / "vercel-frontend" / "config.js"

    with open(config_path, 'r') as f:
        content = f.read()

    # Extract adapter paths from config.js
    import re
    adapter_matches = re.findall(r"value:\s*['\"]([^'\"]+)['\"]", content)
    
    print(f"🔍 DEBUG: Config file content preview: {content[:500]}...")
    print(f"🔍 DEBUG: Adapter matches found: {adapter_matches}")

    if adapter_matches:
        full_paths = [PROJECT_ROOT / "models" / path for path in adapter_matches]
        print(f"🔍 DEBUG: Full adapter paths: {[str(p) for p in full_paths]}")
        return full_paths


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

# Quantization settings for large models
# Set LOAD_IN_4BIT=false to disable 4-bit quantization
# Set LOAD_IN_8BIT=true to use 8-bit quantization instead
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "true").lower() == "true"
LOAD_IN_8BIT = os.environ.get("LOAD_IN_8BIT", "false").lower() == "true"

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Ngrok tunnel - persistent backend tunnel for vercel to connect to
ngrok_listener = None

# Chat logging
chat_log_file = None


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
            ngrok_listener = ngrok.connect(9100,
                                           "http",
                                           config_file=str(config_path))
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
                new_content = re.sub(r"apiBase:\s*['\"][^'\"]*['\"]",
                                     f"apiBase: '{public_url}'", content)

                with open(vercel_config_path, 'w') as f:
                    f.write(new_content)

                print(f"✅ Auto-updated Vercel config.js with new ngrok URL")
            except Exception as e:
                print(f"⚠️  Could not auto-update Vercel config: {e}")

        print(
            f"📝 Your Vercel config.js has been updated with: apiBase: '{public_url}'"
        )

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
atexit.register(lambda: asyncio.run(cleanup_ngrok())
                if ngrok_listener else None)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Security helpers
async def verify_origin(origin: str = Header(None),
                        referer: str = Header(None)):
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


# runtime state
class ChatSession:

    def __init__(self, bot_name: str) -> None:
        self.bot_name = bot_name
        self.history: List[str] = []
        self.turns: int = 0
        self.transcript: List[Dict[str, str]] = []


class ModelManager:
    def __init__(self) -> None:
        self.base_model_id: str = None
        self.adapter_path: Optional[Path] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[torch.nn.Module] = None
        self.loaded: bool = False
        self.model_type: str = "base"  # "base" or "instruct"
        self.model_key: str = "mistral-7b"  # Default model key
        self.steering_vector: Optional[torch.Tensor] = None
        self.steering_enabled: bool = False

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

        raise ValueError(
            f"Could not detect model key from adapter path: {adapter_str}")

    def start(self, adapter_path: Optional[str] = None) -> Dict[str, str]:
        global chat_log_file
        
        # If adapter path is provided and different from current, or no model is loaded
        if adapter_path:
            new_adapter_path = PROJECT_ROOT / "models" / adapter_path
            if self.loaded and str(self.adapter_path) == str(new_adapter_path):
                return {"status": "already_running"}

            # If switching adapters, stop the current model first
            if self.loaded:
                print(
                    f"🔄 Switching from {self.adapter_path} to {new_adapter_path}"
                )
                self.stop()

            self.adapter_path = new_adapter_path
            print(f"Using specified adapter: {self.adapter_path}")
        else:
            if self.loaded:
                return {"status": "already_running"}
            self.adapter_path = self.find_default_adapter()
            print(f"Using auto-detected adapter: {self.adapter_path}")

        if not self.adapter_path or not self.adapter_path.exists():
            print(f"Adapter path not found: {self.adapter_path}")
            raise HTTPException(
                status_code=400,
                detail="LoRA adapter not found. Provide a valid adapter path.")

        # Detect model type and key from adapter path
        self.model_key = self.detect_model_key_from_adapter(self.adapter_path)
        self.model_type = detect_model_type(self.model_key)

        # Load base model ID from checkpoint's adapter config (matches training setup)
        adapter_config_path = self.adapter_path / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            self.base_model_id = adapter_config["base_model_name_or_path"]
            print(f"🔧 Using base model from checkpoint: {self.base_model_id}")
        else:
            # Fallback to config if no adapter config found
            if self.model_key in MODEL_CONFIGS:
                self.base_model_id = MODEL_CONFIGS[
                    self.model_key]["model_name"]
                print(f"🔧 Using base model from config: {self.base_model_id}")

        torch_dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load processor from scratch since I didn't save it
        # so must update its tokenizer to reflect trained tokenizer
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_id,
            use_fast=True,
        )
        if self.processor is not None and hasattr(self.processor, 'tokenizer'):
            try:
                self.processor.tokenizer = self.tokenizer
                print("✅ Processor tokenizer aligned with adapter tokenizer")
            except Exception as e:
                print(f"⚠️  Failed to align processor tokenizer: {e}")

        # For large models like Gemma-27B, use quantization and single-GPU
        if "27b" in self.base_model_id.lower():
            print(
                "🔧 Large model detected, using 4-bit quantization and single-GPU"
            )

            # Import quantization libraries
            try:
                from transformers import BitsAndBytesConfig
                print("🔧 Using BitsAndBytes for 4-bit quantization")

                # Configure 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                # Force single-GPU usage for large models
                device_map = "cuda:0"

            except Exception as e:
                print(
                    f"⚠️  Error using BitsAndBytes for 4-bit quantization: {e}"
                )
        else:
            # For smaller models, use normal settings
            quantization_config = None
            device_map = DEVICE_MAP

        print(f"🔧 Loading base model with device_map: {device_map}")

        # Use FastLanguageModel for unsloth models to match training setup
        if "unsloth" in self.base_model_id.lower():
            print("🔧 Using FastLanguageModel for unsloth model")
            base, _ = FastLanguageModel.from_pretrained(
                model_name=self.base_model_id,
                max_seq_length=4096,
                dtype=torch.bfloat16,  # Use bfloat16 consistently
                load_in_4bit=True if quantization_config else False,
                load_in_8bit=False,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
        if base.get_input_embeddings().num_embeddings != len(self.tokenizer):
            base.resize_token_embeddings(len(self.tokenizer),
                                         mean_resizing=False)

        print(f"🔧 Loading LoRA adapter...")
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self.model.eval()
        self.model = self.model.to(torch.bfloat16)

        # Warm up the model to load checkpoint shards and avoid slow first inference
        print("🔥 Warming up model to preload checkpoint shards...")
        warmup_success = False
        try:
            with torch.no_grad():
                # Get the device of the input embeddings layer
                input_device = self.model.get_input_embeddings().weight.device
                print(f"🔧 Warming up on device: {input_device}")
                print(
                    f"🔧 Creating dummy input with vocab size: {len(self.tokenizer)}"
                )

                # Create dummy input on the input device
                dummy_input = torch.randint(0,
                                            len(self.tokenizer), (1, 10),
                                            device=input_device)
                print(f"🔧 Running dummy inference...")

                # Run inference to warm up the model
                _ = self.model(dummy_input)
                warmup_success = True
                print(
                  "Model warmed up successfully - checkpoint shards loaded"
                )
        except Exception as e:
            print(f" Model warmup failed: {e}")
            print(f" Error details: {type(e).__name__}: {str(e)}")

        self.loaded = True
        
        # Initialize chat log file when model starts
        if chat_log_file is None:
            log_dir = PROJECT_ROOT / "convos" / "chat_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filename = log_dir / f"chat_session_{timestamp}.log"
            chat_log_file = open(log_filename, 'w', encoding='utf-8')
            chat_log_file.write(f"Chat session started at {datetime.now().isoformat()}\n")
            chat_log_file.write(f"Model: {self.base_model_id}\n")
            chat_log_file.write(f"Adapter: {self.adapter_path}\n")
            chat_log_file.write(f"Model type: {self.model_type}\n")
            chat_log_file.write("=" * 80 + "\n\n")
            chat_log_file.flush()
            print(f"📝 Chat log started: {log_filename}")
        
        return {
            "status": "started",
            "base": self.base_model_id,
            "adapter": str(self.adapter_path),
            "model_type": self.model_type,
            "model_key": self.model_key
        }

    def stop(self) -> Dict[str, str]:
        global chat_log_file
        
        out_dir = PROJECT_ROOT / "convos" / "public"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for name, sess in sessions.items():
            if sess.transcript:
                fname = out_dir / f"{timestamp}_{name}.txt"
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(
                        f"Session: {name}\nTime: {timestamp}\nBase: {self.base_model_id}\nAdapter: {self.adapter_path}\n\n"
                    )
                    for m in sess.transcript:
                        f.write(f"{m['role']}: {m['content']}\n")
        for s in sessions.values():
            s.transcript.clear()

        # Close chat log file when model is stopped
        if chat_log_file is not None:
            try:
                chat_log_file.write(f"\nChat session ended at {datetime.now().isoformat()}\n")
                chat_log_file.close()
                print(f"📝 Chat log closed")
            except Exception as e:
                print(f"⚠️  Error closing chat log: {e}")
            chat_log_file = None

        if not self.loaded:
            return {"status": "already_stopped"}
        try:
            del self.model
            del self.tokenizer
            del self.processor
        except Exception:
            pass
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.steering_vector = None
        self.steering_enabled = False
        self.loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        return {"status": "stopped"}

    def ensure_loaded(self) -> None:
        if not self.loaded or not self.model or not self.tokenizer:
            raise HTTPException(status_code=503,
                                detail="Model not loaded. Click Start first.")


model_manager = ModelManager()
sessions: Dict[str, ChatSession] = {
    "riya": ChatSession(RIYA_NAME),
    "owen": ChatSession(OWEN_NAME),
}

app = FastAPI(title="Conversation Web App", version="1.0.0")

# Mount static files for local development
frontend_dir = PROJECT_ROOT / "src" / "deployment" / "vercel-frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

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
            "model_type":
            model_manager.model_type,
            "model_key":
            model_manager.model_key,
            "base_model":
            model_manager.base_model_id,
            "adapter":
            str(model_manager.adapter_path)
            if model_manager.adapter_path else None
        })
    return response


@app.post("/start")
def start(payload: Dict[str, str] = None):
    """Load adapter and warm up model"""
    print(f"🚀 /start endpoint called with payload: {payload}")

    if not payload or "adapter_path" not in payload:
        raise HTTPException(status_code=400, detail="adapter_path is required")

    adapter_path = payload.get("adapter_path")
    if not adapter_path:
        raise HTTPException(status_code=400,
                            detail="adapter_path cannot be empty")

    print(f"📁 Loading adapter: {adapter_path}")
    print(f"🔍 DEBUG: Raw adapter_path received: '{adapter_path}'")
    print(f"🔍 DEBUG: Type of adapter_path: {type(adapter_path)}")

    try:
        # Load the adapter and warm up the model
        result = model_manager.start(adapter_path)
        print(f"✅ Adapter loaded successfully: {result}")
        return result
    except Exception as e:
        print(f"❌ Failed to load adapter: {e}")
        raise HTTPException(status_code=500,
                            detail=f"Failed to load adapter: {str(e)}")


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


@app.post("/clear-history")
def clear_history(payload: Dict, origin: str = Depends(verify_origin)) -> Dict:
    """Clear chat history for a specific bot."""
    bot = payload.get("bot")
    if bot not in sessions:
        raise HTTPException(status_code=400, detail="Invalid bot. Use 'riya' or 'owen'.")
    
    # Clear the session history
    sessions[bot].history.clear()
    sessions[bot].turns = 0
    sessions[bot].transcript.clear()
    
    print(f"🧹 Cleared history for {bot} bot")
    return {"status": "success", "message": f"History cleared for {bot} bot"}


@app.post("/generate-steering-vector")
def generate_steering_vector_endpoint(payload: Dict, origin: str = Depends(verify_origin),) -> Dict:
    """Generate and store a steering vector based on contrast pairs."""
    print(f"🎯 /generate-steering-vector endpoint called with payload: {payload}")
    
    # Ensure model is loaded
    model_manager.ensure_loaded()
    
    pairs = payload.get("pairs", [])
    extract_layer = payload.get("extract_layer", -2)
    alpha = payload.get("alpha", 2.0)
    
    if not pairs:
        raise HTTPException(status_code=400, detail="No contrast pairs provided")
    
    # Convert pairs to steer_dict format
    steer_dict = {}
    for pair in pairs:
        positive = pair.get("positive", "").strip()
        negative = pair.get("negative", "").strip()
        if positive:
            steer_dict[positive] = 0.25  # Positive weight
        if negative:
            steer_dict[negative] = -0.25  # Negative weight
    
    if not steer_dict:
        raise HTTPException(status_code=400, detail="No valid contrast pairs found")
    
    print(f"🎯 Generating steering vector with {len(steer_dict)} prompts")
    print(f"🎯 Steer dict: {steer_dict}")
    
    try:
        # Detect model type
        is_instruct = detect_model_type(model_manager.model_key) == "instruct"
        
        # Generate steering vector
        steering_vector = generate_steering_vector(
            model_manager.model,
            model_manager.tokenizer,
            steer_dict,
            pos_alpha=alpha,
            neg_alpha=alpha,
            layer_from_last=extract_layer,
            processor=model_manager.processor,
            is_instruct=is_instruct,
            current_speaker=RIYA_NAME  # Default to Riya for now
        )
        
        # Debug: Check steering vector values
        vector_magnitude = steering_vector.abs().sum().item()
        vector_max = steering_vector.abs().max().item()
        print(f"🔍 Steering vector debug:")
        print(f"  Shape: {steering_vector.shape}")
        print(f"  Magnitude: {vector_magnitude:.6f}")
        print(f"  Max value: {vector_max:.6f}")
        print(f"  Alpha used: {alpha}")
        
        # Store the steering vector in the model manager for later use
        model_manager.steering_vector = steering_vector
        model_manager.steering_enabled = True
        
        print(f"✅ Steering vector generated successfully")
        
        return {
            "status": "success",
            "message": "Steering vector generated successfully",
            "vector_shape": list(steering_vector.shape),
            "pairs_count": len(steer_dict)
        }
        
    except Exception as e:
        print(f"❌ Error generating steering vector: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate steering vector: {str(e)}")


@app.post("/chat")
def chat(payload: Dict, origin: str = Depends(verify_origin)) -> Dict:
    global chat_log_file
    
    bot = payload.get("bot")
    message = payload.get("message", "").strip()
    steering = payload.get("steering", {}) or {}

    if bot not in sessions:
        raise HTTPException(status_code=400,
                            detail="Invalid bot. Use 'riya' or 'owen'.")
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")

    # Ensure model is loaded
    model_manager.ensure_loaded()
    
    # Log the user message and steering config if enabled
    if chat_log_file is not None:
        try:
            timestamp = datetime.now().isoformat()
            user_name = OWEN_NAME if bot == "riya" else RIYA_NAME
            chat_log_file.write(f"[{timestamp}] {user_name}: {message}\n")
            
            # Log steering configuration if enabled
            use_steering = bool(steering.get("enabled"))
            if use_steering:
                steering_pairs = steering.get("pairs", [])
                extract_layer = steering.get("extract_layer", -2)
                apply_layer = steering.get("apply_layer", -1)
                alpha_strength = steering.get("alpha", 2.0)
                
                chat_log_file.write(f"[{timestamp}] STEERING CONFIG:\n")
                chat_log_file.write(f"[{timestamp}]   Enabled: {use_steering}\n")
                chat_log_file.write(f"[{timestamp}]   Extract Layer: {extract_layer}\n")
                chat_log_file.write(f"[{timestamp}]   Apply Layer: {apply_layer}\n")
                chat_log_file.write(f"[{timestamp}]   Alpha Strength: {alpha_strength}\n")
                
                if steering_pairs:
                    chat_log_file.write(f"[{timestamp}]   Contrast Pairs:\n")
                    for i, pair in enumerate(steering_pairs):
                        pos = pair.get("positive", "").strip()
                        neg = pair.get("negative", "").strip()
                        if pos and neg:
                            chat_log_file.write(f"[{timestamp}]     {i+1}. Positive: '{pos}' | Negative: '{neg}'\n")
                else:
                    chat_log_file.write(f"[{timestamp}]   Contrast Pairs: None\n")
                chat_log_file.write(f"[{timestamp}] END STEERING CONFIG\n")
            
            chat_log_file.flush()
        except Exception as e:
            print(f"⚠️  Error writing to chat log: {e}")

    session = sessions[bot]

    speaker_user = OWEN_NAME if bot == "riya" else RIYA_NAME
    session.transcript.append({"role": speaker_user, "content": message})

    # Build clean conversation history without trailing speaker tokens
    def needs_prefix(msg: str, name_a: str, name_b: str) -> bool:
        s = msg.lstrip()
        return not (s.startswith(f"[{name_a}]") or s.startswith(f"[{name_b}]"))

    if bot == "riya":
        if needs_prefix(message, OWEN_NAME, RIYA_NAME):
            prompt_line = f"\n[{OWEN_NAME}] {message}"
        else:
            prompt_line = f"\n{message}"
        assistant_name = RIYA_NAME
    else:
        if needs_prefix(message, RIYA_NAME, OWEN_NAME):
            prompt_line = f"\n[{RIYA_NAME}] {message}"
        else:
            prompt_line = f"\n{message}"
        assistant_name = OWEN_NAME

    if session.turns > 8 and session.history:
        session.history.pop(0)
    session.history.append(prompt_line)
    session.turns += 1

    # Build the clean conversation history
    full_prompt = "".join(session.history)

    use_steering = bool(steering.get("enabled"))
    steering_pairs = steering.get("pairs", [])
    extract_layer = steering.get("extract_layer", -2)
    apply_layer = steering.get("apply_layer", -1)
    alpha_strength = steering.get("alpha", 2.0)

    max_new = int(payload.get("max_new_tokens") or MAX_NEW_TOKENS)

    # Use unified generation functions that handle both model types
    is_instruct = model_manager.model_type == "instruct"
    target_speaker = assistant_name  # Always pass target_speaker, let generate methods handle it

    if use_steering and model_manager.steering_enabled and model_manager.steering_vector is not None:
        # Use the pre-generated steering vector
        print(f"🎯 Using stored steering vector for generation")
        raw = generate_with_steering(model_manager.model,
                                     full_prompt,
                                     model_manager.tokenizer,
                                     model_manager.steering_vector,
                                     max_new_tokens=max_new,
                                     layer_from_last=apply_layer,
                                     is_instruct=is_instruct,
                                     target_speaker=target_speaker,
                                     deployment=True,
                                     processor=model_manager.processor)
    else:
        # No steering enabled or no steering vector available, use regular generation
        if use_steering and not model_manager.steering_enabled:
            print("⚠️  Steering enabled but no steering vector available. Please generate one first.")
        raw = generate(model_manager.model,
                       full_prompt,
                       model_manager.tokenizer,
                       max_new_tokens=max_new,
                       is_instruct=is_instruct,
                       target_speaker=target_speaker,
                       deployment=True,
                       processor=model_manager.processor)

    # Handle the list of messages returned by generate
    messages = raw if isinstance(raw, list) else [raw]

    # Process each message
    all_responses = []
    for message in messages:
        text = message.strip()
        text = clean_for_sampling(text)

        # Add to history with speaker token
        session.history.append(f"\n[{assistant_name}] {text}")
        session.turns += 1
        session.transcript.append({"role": assistant_name, "content": text})

        # Log the bot response
        if chat_log_file is not None:
            try:
                timestamp = datetime.now().isoformat()
                chat_log_file.write(f"[{timestamp}] {assistant_name}: {text}\n")
                chat_log_file.flush()
            except Exception as e:
                print(f"⚠️  Error writing bot response to chat log: {e}")

        # Format each message with speaker name on new line
        formatted_message = f"{assistant_name}: {text}"
        all_responses.append(formatted_message)

    # Return all responses with each on a new line
    response_text = "\n".join(all_responses)
    return {"response": response_text}


@app.get("/")
# def index() -> HTMLResponse:
#     """Redirect to the Vercel frontend"""
#     return HTMLResponse(content="""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>Conversation App</title>
#         <meta http-equiv="refresh" content="0; url=https://conversation-continuation.vercel.app">
#     </head>
#     <body>
#         <p>Redirecting to <a href="https://conversation-continuation.vercel.app">Vercel Frontend</a>...</p>
#     </body>
#     </html>
#     """)
def index() -> FileResponse:
    """Serve the local frontend for development"""
    frontend_path = PROJECT_ROOT / "src" / "deployment" / "vercel-frontend" / "index.html"
    return FileResponse(str(frontend_path))


@app.get("/config.js")
def get_config() -> FileResponse:
    """Serve the config file"""
    config_path = PROJECT_ROOT / "src" / "deployment" / "vercel-frontend" / "config.js"
    return FileResponse(str(config_path))


if __name__ == "__main__":
    print("🚀 Starting Conversation App...")

    # Start ngrok tunnel
    async def start_with_ngrok():
        print("🔌 Creating ngrok tunnel...")
        tunnel_url = await create_ngrok_tunnel()
        if tunnel_url:
            print(f"✅ ngrok tunnel ready: {tunnel_url}")
        else:
            print(
                "⚠️  ngrok tunnel creation failed, continuing without tunnel")

        print("🌐 Starting FastAPI server on localhost:9100...")
        config = uvicorn.Config("src.deployment.app:app",
                                host="0.0.0.0",
                                port=9100,
                                reload=False)
        server = uvicorn.Server(config)
        await server.serve()

    # Run the async startup
    asyncio.run(start_with_ngrok())

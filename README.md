
# Conversation Continuation: Finetuning LLMs to Simulate People

In this project, I train language models to continue conversations between two speakers, with support for activation steering, training data attribution, and multi-speaker sampling.

## TL;DR

I finetune language models on paired conversation data to simulate realistic dialogue between two speakers. Each speaker has a dedicated token (e.g., `[Riya]`, `[FRIEND]`), allowing the model to learn distinct speaking styles. The trained model can then generate responses as either speaker, with optional steering to adjust the emotional tone of outputs.

The pipeline supports:
- Training with Unsloth for efficient LoRA finetuning on conversation data
- Steering to shift model outputs toward desired emotional directions
- Sampling in various modes: single speaker, alternating speakers, or continuous conversation
- Activation TDA to attribute model outputs to similar training examples
- Data analysis for comparing speakers

## Installation

```bash
git clone <repo-url>
cd conversation-continuation
pip install -r requirements_unsloth.txt
```

Core dependencies: `unsloth`, `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`.

## Data Format

Training data should be JSON files with paired `prompt` and `response` fields. Each message is prefixed with a speaker token.

```json
{
  "prompt": "[FRIEND] hey what's up\n[Riya] not much, just coding\n[FRIEND] nice, working on anything fun?",
  "response": "[Riya] yeah actually! i'm training a model to simulate our conversations"
}
```

Speaker tokens like `[Riya]` and `[FRIEND]` are added as special tokens to the vocabulary. During training, their embeddings are initialized from semantically similar tokens.

For instruct-format models (e.g., Gemma-3), the data is wrapped with system and user prompts:

```json
{
  "prompt": "[SYS] You are a chat model trained to simulate conversations between [FRIEND] and [Riya]. [/SYS] [USER] {conversation_history}\n\nHow does the conversation continue? [/USER]",
  "response": "[Riya] response text here"
}
```

## Training

The Unsloth trainer handles LoRA finetuning. The training script automatically adds special tokens and initializes embeddings.

```bash
python -m src.finetuning.train_lora_unsloth \
  --model gemma-3-27b-it \
  --experiment my_experiment \
  --epochs 1 \
  --batch-size 96 \
  --learning-rate 2e-4
```

Key flags:
- `--model`: `mistral-7b` or `gemma-3-27b-it`
- `--experiment`: Experiment name (defaults to timestamp)
- `--special-tokens`: Add speaker tokens to vocabulary (default: true)
- `--include-mlp`: Include MLP layers in LoRA (default: true)
- `--instruct-format`: Use instruct-format training data (default: true)
- `--continue-training`: Resume from latest checkpoint
- `--quantization`: `4bit` or `8bit`

### Adding Your Own Speaker Tokens

Configure speaker names in `src/config.py`:

```python
RIYA_NAME = "Riya"
FRIEND_NAME = "FRIEND"
RIYA_SPEAKER_TOKEN = f"[{RIYA_NAME}]"
FRIEND_SPEAKER_TOKEN = f"[{FRIEND_NAME}]"
```

Update the embedding initialization mapping in `train_lora_unsloth.py`:

```python
mapping = {
    special_tokens[0]: ["▁Riya"],     # tokens to average for [Riya]
    special_tokens[1]: ["▁FRIEND"]    # tokens to average for [FRIEND]
}
```

## Steering

Steer model outputs toward desired emotional directions by adding activation vectors during generation.

```python
from src.steering.steer_utils import generate_steering_vector
from src.model_utils import generate_with_steering

steer_dict = {
    "i feel happy": 0.25,
    "i feel sad": -0.25,
    "life is amazing": 0.25,
    "life is terrible": -0.25,
}

steering_vector = generate_steering_vector(
    model, tokenizer, steer_dict,
    pos_alpha=1.0, neg_alpha=1.0,
    layer_from_last=-5
)

output = generate_with_steering(
    model, prompt, tokenizer, steering_vector,
    layer_from_last=-2
)
```

Steering is applied by adding the vector to hidden states during generation. The `layer_from_last` parameter controls which layer to extract from and apply to.

## Sampling

Several sampling modes for different use cases.

### Chat with a Speaker

Talk to one speaker while you play the other:

```bash
# Talk to Riya (you are FRIEND)
python -m src.sampling.sample_riya --stream

# Talk to FRIEND (you are Riya)
python -m src.sampling.sample_friend --save
```

Flags:
- `--stream`: Token-by-token streaming output
- `--steer`: Apply emotional steering
- `--save`: Save conversation to file
- `--tda`: Enable training data attribution

### Autonomous Conversation

Let the model simulate both speakers talking to each other:

```bash
python -m src.sampling.converse_forever --save
```

### Compare Models

Evaluate finetuned vs base model on test examples:

```bash
python -m src.sampling.inference_compare
```

## Basic Training Data Attribution

A really simple and noisy method for training data attribution using activation similarity.

How it works:
1. Cache training activations: Extract and cache final-layer hidden states for all training messages
2. Match at inference: Compare generated activations to the cache
3. Surface similar examples: Return training messages with most similar activation patterns

### Extracting Activations

```bash
python -m src.activation_tda.extract_activations
```

This populates a memory-mapped cache with activations indexed by content hash and speaker ID.

### Using TDA During Sampling

```bash
python -m src.sampling.sample_friend --tda
```

Each generated response shows the most similar training examples.

## Data Analysis

One fun thing you can do analyze differences in how speakers embed concepts — for instance, emotions.

### Emotion Comparison

```bash
python -m src.data_analysis.emotion_plots
```

Generates cosine similarity heatmaps showing how emotion relationships differ between speakers.

### Token Distribution Analysis

```bash
python -m src.data_analysis.char_token_plots
```

Outputs character-length and token-count histograms for conversation data.

## Configuration

All settings are centralized in `src/config.py`:

```python
RIYA_NAME = "Riya"
FRIEND_NAME = "FRIEND"
CURRENT_MODEL = "gemma-3-27b-it"
DATA_PATH = "data/finetune_9_3_25"
BASE_RESULTS_FOLDER = "models"
```

## Supported Models

- `mistral-7b`: mistralai/Mistral-7B-v0.1 (base model)
- `gemma-3-27b-it`: google/gemma-3-27b-it (instruct model)

Add new models by extending `MODEL_CONFIGS` in `src/config.py`.

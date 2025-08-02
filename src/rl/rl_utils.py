from src.config import FRIEND_NAME, MODEL_NAME, RIYA_NAME, RESULTS_FOLDER, bnb_config
from openai import OpenAI, OpenAIError
from pathlib import Path
from typing import Dict, List


def load_constitutions(base_path: Path) -> Dict[str, str]:
    """Loads the persona constitutions from files."""
    try:
        with open(base_path / "riya_constitution.txt", "r", encoding="utf-8") as f:
            riya_constitution = f.read()
        with open(base_path / "friend_constitution.txt", "r", encoding="utf-8") as f:
            friend_constitution = f.read()
        return {RIYA_NAME: riya_constitution, FRIEND_NAME: friend_constitution}
    except FileNotFoundError as e:
        print(f"Error: Constitution file not found. Make sure they are in {base_path}")
        raise e

def get_speaker_from_prompt(prompt: str) -> str:
    """Determines the next speaker based on the last line of the prompt. The last line 
    of the prompt will contain \n [FRIEND] or \n [RIYA] based on who should talk next."""
    last_line = prompt.strip().split("\n")[-1]
    if f"[{RIYA_NAME}]" in last_line:
        return RIYA_NAME
    elif f"[{FRIEND_NAME}]" in last_line:
        return FRIEND_NAME
    else:
        raise ValueError(f"Cannot determine speaker from prompt's last line: {last_line}")

def build_prompt(
    conversation_prompt: str,
    original_completion: str, # this will simply be a single output, as in train.json
    persona_constitution: str,
    persona_name: str, # "Riya" or "Friend" to best reference the conversation  
):
    
    system_prompt = """
    You are a writing assistant. Your task is to revise a response from a language model to better align with a specific persona constitution.
    The user will provide the conversation history, the model's original (and likely flawed) response, and the persona constitution to follow.
    Your goal is to rewrite the original response to be more in-character, following all the principles in the constitution.
    \n IMPORTANT: Respond ONLY with the revised text. Do not include any introductory phrases, explanations, or quotes.
    """

    user_message = f"""
    ### CONVERSATION HISTORY
    {conversation_prompt}

    ### PERSONA TO EMBODY
    {persona_name}

    ### ORIGINAL RESPONSE (NEEDS REVISION)
    {original_completion}

    ### PERSONA CONSTITUTION
    {persona_constitution}
    """

    return system_prompt, user_message
    

def get_revised_completion(
    client: OpenAI,
    conversation_prompt: str,
    original_completion: str, # this will simply be a single output, as in train.json
    persona_constitution: str,
    persona_name: str, # "Riya" or "Friend" to best reference the conversation
) -> str:

    system_prompt, user_message = build_prompt(
        conversation_prompt, original_completion, persona_constitution, persona_name)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # could experiment with models
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        revised_text = response.choices[0].message.content
        return revised_text.strip() if revised_text else original_completion
    except OpenAIError as e:
        print(f"An OpenAI API error occurred: {e}")
        # Return the original completion as a fallback
        return original_completion
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return original_completion

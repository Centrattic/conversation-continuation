import re
from typing import List, Tuple
from src.config import RIYA_NAME, FRIEND_NAME, RIYA_MENTION_TOKEN, FRIEND_MENTION_TOKEN, RIYA_SPEAKER_TOKEN, FRIEND_SPEAKER_TOKEN

def strip_training_data(example):
    """Strip training data."""
    prompt = example["prompt"].strip()
    response = example["response"].strip()
    
    return {
        "prompt": prompt,
        "response": response
    }

def insert_mentions(text: str, speakers: List[str]) -> str:
    """ Surround first-person pronouns (I/me/my…) with <m:CurrentSpeaker>, 
    and second-person pronouns (you/your…) with <m:OtherSpeaker>.
    Also catches bare names (Friend / Riya) as mentions.
    """
    if len(speakers) != 2:
        raise ValueError("speakers must be a list of exactly 2 names")
    
    speaker, other = speakers
    
    # Patterns for pronouns and names
    patterns = {
        # First person pronouns - handle both uppercase and lowercase
        rf"\b(I|i)\b": f"<m:{speaker}>",  # Standalone I/i
        rf"\b(me|my|mine)\b": f"<m:{speaker}>",
        
        # Contractions with I/i - handle both cases
        rf"\bI'm\b": f"<m:{speaker}>'m",
        rf"\bi'm\b": f"<m:{speaker}>'m",
        rf"\bI'll\b": f"<m:{speaker}>'ll",
        rf"\bi'll\b": f"<m:{speaker}>'ll",
        rf"\bI've\b": f"<m:{speaker}>'ve",
        rf"\bi've\b": f"<m:{speaker}>'ve",
        rf"\bI'd\b": f"<m:{speaker}>'d",
        rf"\bi'd\b": f"<m:{speaker}>'d",
        
        # Second person pronouns
        rf"\b(you|your|yours)\b": f"<m:{other}>",
        rf"\byou're\b": f"<m:{other}>'re",
        rf"\byou'll\b": f"<m:{other}>'ll",
        rf"\byou've\b": f"<m:{other}>'ve",
        rf"\byou'd\b": f"<m:{other}>'d",
        
        # Bare names (not in speaker tags) - case insensitive
        rf"(?<!\[)\b{other}\b(?!\])": f"<m:{other}>",
        rf"(?<!\[)\b{speaker}\b(?!\])": f"<m:{speaker}>",
    }
    
    for pat, repl in patterns.items():
        text = re.sub(pat, repl, text)
    
    return text

def remove_mentions(text: str) -> str:
    """
    Remove all mention tokens from text for sending to GPT-4o-mini for RL prompt generation, or for sampling.
    """
    # Remove mention tokens
    text = re.sub(r'<m:[^>]+>', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_for_sampling(text: str) -> str:
    """
    Clean text for sampling by removing special tokens and formatting.
    """
    # Remove mention tokens
    text = remove_mentions(text)
    # Remove any remaining special tokens
    text = re.sub(r'<[^>]+>', '', text)
    # Clean up extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def get_speaker_mention_tokens() -> List[str]:
    """Get list of all speaker and mention tokens for training."""
    return [
        RIYA_SPEAKER_TOKEN,
        FRIEND_SPEAKER_TOKEN,
        RIYA_MENTION_TOKEN,
        FRIEND_MENTION_TOKEN
    ]

def get_speaker_tokens() -> List[str]:
    """ Get list of speaker tokens for training. """
    return [
        RIYA_SPEAKER_TOKEN,
        FRIEND_SPEAKER_TOKEN
    ]
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
import json
from src.config import FRIEND_NAME, RIYA_NAME, RIYA_SPEAKER_TOKEN, FRIEND_SPEAKER_TOKEN, DATA_PATH, INSTRUCT_SYSTEM_PROMPT, INSTRUCT_USER_PROMPT_TEMPLATE
from tqdm import tqdm
import random


def load_and_prepare_instruct_data(path: str,
                                   context_window: int = 8,
                                   max_gap_minutes: int = 15,
                                   exclude_strings=[]):
    """
    Load and prepare data in instruct format for training.
    
    Args:
        path: Path to the CSV file
        context_window: Number of messages to include in context
        max_gap_minutes: Maximum gap in minutes before resetting context
        exclude_strings: List of strings to exclude from content
    
    Returns:
        List of conversation examples in instruct format
    """
    df = pd.read_csv(path)
    df.columns = [
        "AuthorID", "Author", "Date", "Content", "Attachments", "Reactions"
    ]
    df = df.dropna(subset=["Content"])  # remove empty messages

    if exclude_strings:
        for text in exclude_strings:
            df = df[~df["Content"].str.contains(text, case=False, na=False)]

    df["Date"] = pd.to_datetime(df["Date"], format="ISO8601")
    df = df.sort_values(by="Date")

    conversations = []
    buffer = []

    for i in tqdm(range(len(df))):
        current = df.iloc[i]
        if i > 0:
            prev_time = df.iloc[i - 1]["Date"]
            gap = (current["Date"] - prev_time).total_seconds() / 60
            if gap > max_gap_minutes:
                buffer = []  # reset context due to large gap

        speaker = f"[{RIYA_NAME}]" if current[
            "Author"] == "rtyagi86" else f"[{FRIEND_NAME}]"
        buffer.append(f"{speaker} {current['Content'].strip()}")

        if len(buffer) >= context_window + 1:  # +1 for the response
            # Create conversation history (excluding the last message)
            conversation_history = "\n".join(buffer[-(context_window + 1):-1])

            # Determine the next speaker and their response
            next_message = buffer[-1]
            next_speaker_tag = next_message.split(" ", 1)[0]
            next_speaker_content = next_message.split(" ", 1)[1]

            # Determine who the next speaker is (Riya or Friend)
            if next_speaker_tag == RIYA_SPEAKER_TOKEN:
                next_speaker = RIYA_NAME
            else:
                next_speaker = FRIEND_NAME

            # Create the instruct format
            instruct_example = create_instruct_example(conversation_history,
                                                       next_speaker,
                                                       next_speaker_content)
            conversations.append(instruct_example)

    conversations.reverse()  # train on latest data first
    return conversations


def create_instruct_example(conversation_history: str, next_speaker: str,
                            response_content: str) -> Dict:
    """
    Create a single instruct format example.
    
    Args:
        conversation_history: The conversation history as a string
        next_speaker: The name of the next speaker (RIYA_NAME or FRIEND_NAME)
        response_content: The content of the response
    
    Returns:
        Dictionary with 'system', 'user', and 'response' keys
    """
    # Format the system prompt with speaker tokens
    system_prompt = INSTRUCT_SYSTEM_PROMPT.format(
        FRIEND_SPEAKER_TOK=FRIEND_SPEAKER_TOKEN,
        RIYA_SPEAKER_TOK=RIYA_SPEAKER_TOKEN)

    # Format the user prompt
    user_prompt = INSTRUCT_USER_PROMPT_TEMPLATE.format(
        conversation_history=conversation_history, next_speaker=next_speaker)

    # The response should use the proper speaker format: [Owen] {message}
    if next_speaker == RIYA_NAME:
        response = f"{RIYA_SPEAKER_TOKEN} {response_content}"
    else:
        response = f"{FRIEND_SPEAKER_TOKEN} {response_content}"

    return {"system": system_prompt, "user": user_prompt, "response": response}


def create_instruct_example_with_multiple_responses(
        conversation_history: str, next_speaker: str,
        response_messages: List[str]) -> Dict:
    """
    Create an instruct example where the response includes multiple consecutive messages
    from the same speaker.
    
    Args:
        conversation_history: The conversation history as a string
        next_speaker: The name of the next speaker (RIYA_NAME or FRIEND_NAME)
        response_messages: List of consecutive messages from the next speaker
    
    Returns:
        Dictionary with 'system', 'user', and 'response' keys
    """
    system_prompt = INSTRUCT_SYSTEM_PROMPT.format(
        FRIEND_SPEAKER_TOK=FRIEND_SPEAKER_TOKEN,
        RIYA_SPEAKER_TOK=RIYA_SPEAKER_TOKEN)

    user_prompt = INSTRUCT_USER_PROMPT_TEMPLATE.format(
        conversation_history=conversation_history, next_speaker=next_speaker)

    # Combine all response messages with proper speaker format: [Riya] message1 [Riya] message2 [Riya] message3
    if next_speaker == RIYA_NAME:
        combined_response = " ".join([
            f"{RIYA_SPEAKER_TOKEN} {msg}" for msg in response_messages
        ])
    else:
        combined_response = " ".join([
            f"{FRIEND_SPEAKER_TOKEN} {msg}" for msg in response_messages
        ])

    return {
        "system": system_prompt,
        "user": user_prompt,
        "response": combined_response
    }


def load_and_prepare_instruct_data_with_multiple_responses(
    path: str,
    min_context_window: int = 4,
    max_context_window: int = 20,
    num_context_samples: int = 3,
    max_gap_minutes: int = 45,
    exclude_strings=[],
):
    """
    Enhanced version that creates datapoints by looking forward from each message.
    For each message, randomly samples num_context_samples context window lengths
    between min_context_window and max_context_window.
    
    Each datapoint contains:
    - Prompt: context starting from the current message and going forward for the chosen length
    - Response: what the next speaker says after that context window
    """
    df = pd.read_csv(path)
    df.columns = [
        "AuthorID", "Author", "Date", "Content", "Attachments", "Reactions"
    ]
    df = df.dropna(subset=["Content"])

    if exclude_strings:
        for text in exclude_strings:
            df = df[~df["Content"].str.contains(text, case=False, na=False)]

    df["Date"] = pd.to_datetime(df["Date"], format="ISO8601")
    df = df.sort_values(by="Date")

    conversations = []
    
    # Convert to list of messages with speaker info for easier processing
    messages = []
    for _, row in df.iterrows():
        speaker = f"[{RIYA_NAME}]" if row["Author"] == "rtyagi86" else f"[{FRIEND_NAME}]"
        messages.append({
            "speaker": speaker,
            "content": row["Content"].strip(),
            "date": row["Date"]
        })

    for i in tqdm(range(len(messages))):
        current_message = messages[i]
        
        # For each message, randomly choose context lengths
        available_lengths = list(range(
            min_context_window,
            min(max_context_window + 1, len(messages) - i)
        ))
        
        if not available_lengths:
            continue
            
        # Randomly sample context lengths
        context_lengths = random.sample(
            available_lengths, 
            min(num_context_samples, len(available_lengths))
        )
        
        for context_length in context_lengths:
            # Check if we have enough messages ahead to create this context
            if i + context_length >= len(messages):
                continue
                
            # Create context starting from current message and going forward
            context_messages = messages[i:i + context_length]
            conversation_history = "\n".join([
                f"{msg['speaker']} {msg['content']}" 
                for msg in context_messages
            ])
            
            # Find the next speaker after the context window
            next_message_idx = i + context_length
            if next_message_idx >= len(messages):
                continue
                
            next_message = messages[next_message_idx]
            next_speaker_tag = next_message["speaker"]
            
            # Determine who the next speaker is
            if next_speaker_tag == f"[{RIYA_NAME}]":
                next_speaker = RIYA_NAME
            else:
                next_speaker = FRIEND_NAME
            
            # Look ahead to collect all consecutive messages from the next speaker
            response_messages = [next_message["content"]]
            j = next_message_idx + 1
            
            while j < len(messages):
                # Check for time gap
                if j > 0:
                    gap = (messages[j]["date"] - messages[j-1]["date"]).total_seconds() / 60
                    if gap > max_gap_minutes:
                        break
                
                # Only include messages from the same speaker
                if messages[j]["speaker"] == next_speaker_tag:
                    response_messages.append(messages[j]["content"])
                    j += 1
                else:
                    break
            
            # Create the instruct example
            instruct_example = create_instruct_example_with_multiple_responses(
                conversation_history, next_speaker, response_messages)
            
            # Add metadata
            instruct_example["context_window"] = context_length
            instruct_example["start_message_index"] = i
            conversations.append(instruct_example)

    conversations.reverse()
    return conversations


def train_test_split(conversations: List[Dict], train_ratio=0.9, seed=42):
    """Split conversations into train and test sets."""
    random.seed(seed)
    random.shuffle(conversations)
    split_idx = int(len(conversations) * train_ratio)
    train_data = conversations[:split_idx]
    test_data = conversations[split_idx:]
    return train_data, test_data


def save_json(data: List[Dict], path: str):
    """Save data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_for_training(instruct_examples: List[Dict]) -> List[Dict]:
    """
    Format instruct examples for training by combining system, user, and response
    into a single prompt-response format with clear labels.
    """
    formatted_examples = []

    for example in instruct_examples:
        # Format with clear labels for easy parsing:
        # <s>[SYS] {system_prompt} [/SYS] [USER] {user_prompt} [/USER]
        full_prompt = f"<s>[SYS] {example['system']} [/SYS]\n[USER] {example['user']} [/USER]"

        formatted_examples.append({
            "prompt": full_prompt,
            "response": example["response"]
        })

    return formatted_examples


# Main execution for creating instruct training data
if __name__ == "__main__":
    # Load and prepare instruct data with variable context windows
    data = load_and_prepare_instruct_data_with_multiple_responses(
        f"{DATA_PATH}/friend_hist_sept.csv",
        min_context_window=4,
        max_context_window=20,
        num_context_samples=3,  # Sample 3 context lengths per conversation
        max_gap_minutes=120)

    # Format for training
    formatted_data = format_for_training(data)

    # Save the data
    save_json(formatted_data, f"{DATA_PATH}/instruct_train.json")

    print(f"Created {len(formatted_data)} instruct training examples")
    print(f"Context windows range: 4-20 messages, sampling {3} lengths per conversation")

    # Show context window distribution
    context_counts = {}
    for example in data:
        ctx = example.get("context_window", "unknown")
        context_counts[ctx] = context_counts.get(ctx, 0) + 1

    print("\nContext window distribution:")
    for ctx in sorted(context_counts.keys()):
        print(f"  {ctx} messages: {context_counts[ctx]} examples")

    print("\nSample example:")
    if formatted_data:
        print(json.dumps(formatted_data[0], indent=2))

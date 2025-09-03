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

    # The response should include all messages from the next speaker
    # For now, we'll just use the single message, but this could be extended
    # to include multiple consecutive messages from the same speaker
    response = f"{next_speaker} says: {response_content}"

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

    # Combine all response messages
    combined_response = f"{next_speaker} says: " + " ".join(response_messages)

    return {
        "system": system_prompt,
        "user": user_prompt,
        "response": combined_response
    }


def load_and_prepare_instruct_data_with_multiple_responses(
        path: str,
        context_window: int = 8,
        max_gap_minutes: int = 15,
        exclude_strings=[]):
    """
    Enhanced version that groups consecutive messages from the same speaker.
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

        if len(buffer) >= context_window + 1:
            # Create conversation history (excluding the last message)
            conversation_history = "\n".join(buffer[-(context_window + 1):-1])

            # Get the next message and determine speaker
            next_message = buffer[-1]
            next_speaker_tag = next_message.split(" ", 1)[0]
            next_speaker_content = next_message.split(" ", 1)[1]

            if next_speaker_tag == RIYA_SPEAKER_TOKEN:
                next_speaker = RIYA_NAME
            else:
                next_speaker = FRIEND_NAME

            # Look ahead to see if there are more consecutive messages from the same speaker
            response_messages = [next_speaker_content]

            # Check if there are more messages from the same speaker after this one
            if i + 1 < len(df):
                j = i + 1
                while j < len(df):
                    next_df_row = df.iloc[j]
                    if j > 0:
                        prev_time = df.iloc[j - 1]["Date"]
                        gap = (next_df_row["Date"] -
                               prev_time).total_seconds() / 60
                        if gap > max_gap_minutes:
                            break

                    next_speaker_tag_check = f"[{RIYA_NAME}]" if next_df_row[
                        "Author"] == "rtyagi86" else f"[{FRIEND_NAME}]"
                    if next_speaker_tag_check == next_speaker_tag:
                        response_messages.append(
                            next_df_row["Content"].strip())
                        j += 1
                    else:
                        break

            # Create the instruct example
            instruct_example = create_instruct_example_with_multiple_responses(
                conversation_history, next_speaker, response_messages)
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
    into a single prompt-response format.
    """
    formatted_examples = []

    for example in instruct_examples:
        # Combine system and user prompts
        full_prompt = f"<s>[INST] {example['system']}\n\n{example['user']} [/INST]"

        formatted_examples.append({
            "prompt": full_prompt,
            "response": example["response"]
        })

    return formatted_examples


# Main execution for creating instruct training data
if __name__ == "__main__":
    # Load and prepare instruct data
    data = load_and_prepare_instruct_data_with_multiple_responses(
        f"{DATA_PATH}/friend_hist_new.csv",
        context_window=8,
        max_gap_minutes=120)

    # Format for training
    formatted_data = format_for_training(data)

    # Save the data
    save_json(formatted_data, f"{DATA_PATH}/instruct_train.json")

    print(f"Created {len(formatted_data)} instruct training examples")
    print("Sample example:")
    if formatted_data:
        print(json.dumps(formatted_data[0], indent=2))

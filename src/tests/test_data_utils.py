from src.data_utils import insert_mentions, remove_mentions, clean_for_sampling, get_speaker_mention_tokens
from src.config import RIYA_NAME, FRIEND_NAME

def test_mention_token_functions():
    """Test the mention token functions with example conversations."""
    
    print("Testing mention token functions...")
    print("="*50)
    
    # Test data
    test_conversations = [
        f"[{RIYA_NAME}] I think you should see this, {FRIEND_NAME}.",
        f"[{FRIEND_NAME}] That's interesting! I never thought about it that way.",
        f"[{RIYA_NAME}] What do you think about my idea?",
        f"[{FRIEND_NAME}] I love your perspective on this.",
        f"[{RIYA_NAME}] I'm going to the store. I'll be back soon.",
        f"[{FRIEND_NAME}] You're right, I've been thinking about it too.",
        f"[{RIYA_NAME}] I'd like to know what you think about this.",
        f"[{FRIEND_NAME}] You'll see what I mean when you try it.",
    ]
    
    print("Original conversations:")
    for i, conv in enumerate(test_conversations, 1):
        print(f"{i}. {conv}")
    
    print("\n" + "="*50)
    print("After inserting mention tokens:")
    
    for i, conv in enumerate(test_conversations, 1):
        # Determine speakers
        if f"[{RIYA_NAME}]" in conv:
            speakers = [RIYA_NAME, FRIEND_NAME]
        else:
            speakers = [FRIEND_NAME, RIYA_NAME]
        
        with_mentions = insert_mentions(conv, speakers)
        print(f"{i}. {with_mentions}")
    
    print("\n" + "="*50)
    print("After removing mention tokens (for RL):")
    
    for i, conv in enumerate(test_conversations, 1):
        if f"[{RIYA_NAME}]" in conv:
            speakers = [RIYA_NAME, FRIEND_NAME]
        else:
            speakers = [FRIEND_NAME, RIYA_NAME]
        
        with_mentions = insert_mentions(conv, speakers)
        without_mentions = remove_mentions(with_mentions)
        print(f"{i}. {without_mentions}")
    
    print("\n" + "="*50)
    print("After cleaning for sampling:")
    
    for i, conv in enumerate(test_conversations, 1):
        if f"[{RIYA_NAME}]" in conv:
            speakers = [RIYA_NAME, FRIEND_NAME]
        else:
            speakers = [FRIEND_NAME, RIYA_NAME]
        
        with_mentions = insert_mentions(conv, speakers)
        cleaned = clean_for_sampling(with_mentions)
        print(f"{i}. {cleaned}")
    
    print("\n" + "="*50)
    print("Special tokens for training:")
    special_tokens = get_speaker_mention_tokens()
    for token in special_tokens:
        print(f"- {token}")

def test_edge_cases():
    """Test edge cases and error handling."""
    
    print("\n" + "="*50)
    print("Testing edge cases...")
    
    # Test with wrong number of speakers
    try:
        insert_mentions("Hello", ["Riya"])  # Should raise error
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test with empty text
    result = insert_mentions("", [RIYA_NAME, FRIEND_NAME])
    print(f"✓ Empty text result: '{result}'")
    
    # Test with no pronouns
    result = insert_mentions("Hello world", [RIYA_NAME, FRIEND_NAME])
    print(f"✓ No pronouns result: '{result}'")
    
    # Test for false positives - words that contain 'i' but shouldn't be matched
    false_positive_tests = [
        "in the house",
        "it is working",
        "is this correct",
        "if you want",
        "inside the box",
        "information about it",
        "I think it is good",  # Should only match the "I" at start
    ]
    
    print("\nTesting for false positives (words with 'i' that shouldn't be matched):")
    for test in false_positive_tests:
        result = insert_mentions(test, [RIYA_NAME, FRIEND_NAME])
        print(f"Input:  '{test}'")
        print(f"Output: '{result}'")
        print()

test_mention_token_functions()
test_edge_cases()

print("\n" + "="*50)
print("All tests completed!") 
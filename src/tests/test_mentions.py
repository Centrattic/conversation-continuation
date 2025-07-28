from src.data_utils import insert_mentions
from src.config import RIYA_NAME, FRIEND_NAME

def test_real_conversations():
    """Test with real conversation examples from the data."""
    
    # Real examples from the conversation data
    real_examples = [
        "can i have an assignment?",
        "he cut me off 💀",
        "Gimme a minute I'm coming up rn",
        "can i have My prog assignment 🙏",
        "test the park autos",
        "do you know where dean of students office is",
        "wait sorry but i don't have tiem to get an orange",
        "i promise during unifree/g i will get one and send image",
        "Oh I see",
        "I think they would be concerned",
        "Let me get back to you on this",
        "Thanks",
    ]
    
    print("Testing with real conversation data:")
    print("="*50)
    
    for i, text in enumerate(real_examples, 1):
        # Determine speaker (simulating from context)
        if any(word in text.lower() for word in ['assignment', 'prog', 'dean', 'orange', 'unifree']):
            speakers = [FRIEND_NAME, RIYA_NAME]
        else:
            speakers = [RIYA_NAME, FRIEND_NAME]

        result = insert_mentions(text, speakers)
        print(f"{i:2d}. Input:  '{text}'")
        print(f"    Output: '{result}'")
        print()

test_real_conversations() 
#!/usr/bin/env python3
"""
Test script for the complete finetuning pipeline with Gemma-3-27B-IT and Unsloth.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.model_utils import get_model_config, create_experiment_folder, save_experiment_config
from src.finetuning.instruct_data_cleaning import load_and_prepare_instruct_data_with_multiple_responses, format_for_training, save_json
from src.config import DATA_PATH


def test_model_config():
    """Test model configuration loading."""
    print("Testing model configuration...")

    # Test Gemma config
    gemma_config = get_model_config("gemma-3-27b-it")
    print(f"Gemma model name: {gemma_config['model_name']}")
    print(f"Gemma model type: {gemma_config['model_type']}")
    print(
        f"Gemma LoRA targets with MLP: {gemma_config['lora_targets_with_mlp']}"
    )

    # Test Mistral config
    mistral_config = get_model_config("mistral-7b")
    print(f"Mistral model name: {mistral_config['model_name']}")
    print(f"Mistral model type: {mistral_config['model_type']}")

    print("✓ Model configuration test passed\n")


def test_experiment_management():
    """Test experiment folder creation and management."""
    print("Testing experiment management...")

    # Create a test experiment
    experiment_folder = create_experiment_folder("gemma-3-27b-it",
                                                 "test_experiment")
    print(f"Created experiment folder: {experiment_folder}")

    # Save test config
    test_config = {
        "model": "gemma-3-27b-it",
        "test": True,
        "timestamp": "2024-01-01T00:00:00"
    }
    save_experiment_config("gemma-3-27b-it", "test_experiment", test_config)
    print("✓ Experiment management test passed\n")


def test_data_cleaning():
    """Test the instruct data cleaning pipeline."""
    print("Testing instruct data cleaning pipeline...")

    # Check if data file exists
    data_file = Path(DATA_PATH) / "friend_hist_new.csv"
    if not data_file.exists():
        print(f"⚠️  Data file not found at {data_file}")
        print("Skipping data cleaning test")
        return

    try:
        # Test data loading (with small context window for speed)
        data = load_and_prepare_instruct_data_with_multiple_responses(
            str(data_file),
            context_window=4,  # Small for testing
            max_gap_minutes=120)

        print(f"Loaded {len(data)} conversation examples")

        if data:
            # Show sample
            sample = data[0]
            print("Sample instruct example:")
            print(f"System: {sample['system'][:100]}...")
            print(f"User: {sample['user'][:100]}...")
            print(f"Response: {sample['response'][:100]}...")

            # Test formatting for training
            formatted_data = format_for_training(
                data[:5])  # Just first 5 for testing
            print(f"Formatted {len(formatted_data)} examples for training")

            if formatted_data:
                print("Sample formatted example:")
                print(f"Prompt: {formatted_data[0]['prompt'][:200]}...")
                print(f"Response: {formatted_data[0]['response'][:100]}...")

        print("✓ Data cleaning test passed\n")

    except Exception as e:
        print(f"✗ Data cleaning test failed: {e}\n")


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        # Test Unsloth import
        from unsloth import FastLanguageModel
        print("✓ Unsloth import successful")

        # Test TRL import
        from trl import SFTTrainer
        print("✓ TRL import successful")

        # Test transformers import
        from transformers import TrainingArguments
        print("✓ Transformers import successful")

        # Test local imports
        from src.finetuning.callbacks import SampleGenerationCallback, LiveJSONLogger
        print("✓ Local callbacks import successful")

        print("✓ All imports test passed\n")

    except ImportError as e:
        print(f"✗ Import test failed: {e}")
        print("Please install required dependencies:")
        print("pip install -r requirements_unsloth.txt\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING COMPLETE FINETUNING PIPELINE")
    print("=" * 60)

    test_imports()
    test_model_config()
    test_experiment_management()
    test_data_cleaning()

    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

    print("\nTo run the actual training, use:")
    print(
        "python src/finetuning/train_lora_unsloth.py --model gemma-3-27b-it --include-mlp --instruct-format"
    )
    print("\nTo create instruct training data first:")
    print("python src/finetuning/instruct_data_cleaning.py")


if __name__ == "__main__":
    main()

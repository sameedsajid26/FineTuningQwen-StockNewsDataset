#!/usr/bin/env python3
"""
Data Preparation Script for Financial News Fine-tuning

This script:
1. Loads data from JSONL file
2. Extracts event categories for balanced splitting
3. Creates train/eval split ensuring all categories are represented
4. Saves processed data in the correct format
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Any
import random

def extract_event_category(example: Dict[str, Any]) -> str:
    """
    Extract event_category from example.
    Handles both raw JSON format and messages format.
    
    Args:
        example: Dictionary with either raw fields or 'messages' key
        
    Returns:
        event_category string or 'unknown' if not found
    """
    # First try direct field (raw JSON format)
    if "event_category" in example:
        category = example.get("event_category")
        if category:  # Keep all categories including "no_stock_movement"
            return str(category)
    
    # Otherwise try to extract from messages format
    try:
        messages = example.get("messages", [])
        if messages:
            # Find assistant message
            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Parse JSON from assistant response
                    parsed = json.loads(content)
                    category = parsed.get("event_category", "unknown")
                    if category:
                        return str(category)
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass
    
    return "unknown"


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSON or JSONL file.
    
    Args:
        file_path: Path to JSON or JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    
    # Check file extension
    if file_path.endswith('.jsonl'):
        # Load JSONL file (one JSON object per line)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Skipping line {line_num}: {e}")
                    continue
    else:
        # Load JSON file (array of objects)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data = file_data
                elif isinstance(file_data, dict):
                    # If it's a dict, try to find a list inside
                    for key, value in file_data.items():
                        if isinstance(value, list):
                            data = value
                            break
                    if not data:
                        raise ValueError("JSON file does not contain a list of objects")
                else:
                    raise ValueError("JSON file format not recognized")
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing JSON file: {e}")
    
    return data


def convert_to_messages_format(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw JSON format to messages format for fine-tuning.
    
    Args:
        example: Dictionary with raw fields (text, company, ticker, etc.)
        
    Returns:
        Dictionary with 'messages' key in chat format, or None if invalid
    """
    # Skip examples with no text
    if not example.get("text"):
        return None
    
    system_prompt = """You are a financial news analyst specializing in cause and effect of stock price movement. Extract structured information from news articles.
Output a JSON object with these exact fields:
- company: The company name
- ticker: The stock ticker symbol
- cause: What caused the stock movement
- effect: The resulting stock price movement
- event_category: One of [earnings_beat, earnings_miss, guidance_change, deal_announced, analyst_action, regulatory_action, product_event, corporate_governance, macro_sentiment, no_stock_movement]
- impact_direction: One of [positive, negative, neutral]"""
    
    user_prompt = f"""Extract structured information from this financial news article.

Financial news:

{example.get("text", "")}

Respond with only the JSON object, no additional text."""
    
    # Build the structured response
    structured_response = {
        "company": example.get("company") or None,
        "ticker": example.get("ticker") or None,
        "cause": example.get("cause") or None,
        "effect": example.get("effect") or None,
        "event_category": example.get("event_category", ""),
        "impact_direction": example.get("impact_direction", "")
    }
    
    # Create messages format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": json.dumps(structured_response, indent=2)}
    ]
    
    return {"messages": messages}


def create_balanced_split(data: List[Dict[str, Any]], 
                         test_size: float = 0.2,
                         seed: int = 42) -> tuple:
    """
    Create balanced train/eval split ensuring all event categories are represented.
    
    Args:
        data: List of data examples (can be raw format or messages format)
        test_size: Proportion of data for evaluation (default: 0.2)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, eval_data) in messages format
    """
    random.seed(seed)
    
    # Convert to messages format and filter invalid examples
    converted_data = []
    skipped = 0
    
    for example in data:
        # Check if already in messages format
        if "messages" in example:
            converted_example = example
        else:
            # Convert from raw format
            converted_example = convert_to_messages_format(example)
            if converted_example is None:
                skipped += 1
                continue
        
        converted_data.append(converted_example)
    
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} examples (no company or invalid category)")
    
    # Group data by event category
    category_groups = defaultdict(list)
    for idx, example in enumerate(converted_data):
        category = extract_event_category(example)
        if category != "unknown":  # Include all valid categories including "no_stock_movement"
            category_groups[category].append((idx, example))
    
    # Print category distribution
    print("\n📊 Event Category Distribution:")
    for category, examples in sorted(category_groups.items()):
        print(f"  {category}: {len(examples)} examples")
    
    train_data = []
    eval_data = []
    
    # Split each category proportionally
    for category, examples in category_groups.items():
        # Shuffle examples within category
        random.shuffle(examples)
        
        # Calculate split point
        n_eval = max(1, int(len(examples) * test_size))  # At least 1 in eval
        n_eval = min(n_eval, len(examples) - 1)  # At least 1 in train
        
        # Split
        eval_examples = examples[:n_eval]
        train_examples = examples[n_eval:]
        
        # Extract just the example dictionaries
        eval_data.extend([ex[1] for ex in eval_examples])
        train_data.extend([ex[1] for ex in train_examples])
        
        print(f"  {category}: {len(train_examples)} train, {len(eval_examples)} eval")
    
    # Final shuffle to mix categories
    random.shuffle(train_data)
    random.shuffle(eval_data)
    
    return train_data, eval_data


def validate_data(data: List[Dict[str, Any]]) -> tuple:
    """
    Validate data format and return statistics.
    Handles both raw JSON format and messages format.
    
    Args:
        data: List of data examples
        
    Returns:
        Tuple of (is_valid, stats_dict)
    """
    stats = {
        'total': len(data),
        'valid': 0,
        'invalid': 0,
        'missing_required_fields': 0,
        'no_company': 0,
        'invalid_category': 0,
        'missing_messages': 0,
        'missing_assistant': 0,
        'invalid_json': 0
    }
    
    for example in data:
        # Check if it's in messages format
        if "messages" in example:
            messages = example["messages"]
            has_assistant = any(msg.get("role") == "assistant" for msg in messages)
            
            if not has_assistant:
                stats['missing_assistant'] += 1
                stats['invalid'] += 1
                continue
            
            # Try to parse assistant JSON
            try:
                for msg in messages:
                    if msg.get("role") == "assistant":
                        json.loads(msg.get("content", ""))
                        break
                stats['valid'] += 1
            except json.JSONDecodeError:
                stats['invalid_json'] += 1
                stats['invalid'] += 1
                continue
        else:
            # Raw JSON format - check required fields
            required_fields = ['text', 'company', 'event_category']
            missing_fields = [field for field in required_fields if not example.get(field)]
            
            if missing_fields:
                stats['missing_required_fields'] += 1
                stats['invalid'] += 1
                continue
            
            # Check for valid category (but allow no_stock_movement)
            category = example.get("event_category", "")
            if not category:
                stats['invalid_category'] += 1
                stats['invalid'] += 1
                continue
            
            # Company can be null for no_stock_movement category - that's valid
            
            stats['valid'] += 1
    
    is_valid = stats['invalid'] == 0
    return is_valid, stats


def prepare_dataset(input_file: str, 
                   output_dir: str = "data",
                   test_size: float = 0.2,
                   seed: int = 42):
    """
    Main function to prepare dataset for fine-tuning.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save processed data
        test_size: Proportion for evaluation set
        seed: Random seed
    """
    print("=" * 60)
    print("🚀 Financial News Dataset Preparation")
    print("=" * 60)
    
    # Step 1: Load data
    print(f"\n📂 Step 1: Loading data from {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    data = load_data(input_file)
    print(f"✅ Loaded {len(data)} examples")
    
    # Step 2: Validate data
    print(f"\n🔍 Step 2: Validating data...")
    is_valid, stats = validate_data(data)
    
    print(f"  Total examples: {stats['total']}")
    print(f"  Valid examples: {stats['valid']}")
    print(f"  Invalid examples: {stats['invalid']}")
    
    if stats['invalid'] > 0:
        print(f"\n⚠️  Validation Issues:")
        if stats['missing_required_fields'] > 0:
            print(f"  - Missing required fields: {stats['missing_required_fields']}")
        if stats['no_company'] > 0:
            print(f"  - No company specified: {stats['no_company']}")
        if stats['invalid_category'] > 0:
            print(f"  - Invalid event category: {stats['invalid_category']}")
        if stats['missing_messages'] > 0:
            print(f"  - Missing 'messages' field: {stats['missing_messages']}")
        if stats['missing_assistant'] > 0:
            print(f"  - Missing assistant message: {stats['missing_assistant']}")
        if stats['invalid_json'] > 0:
            print(f"  - Invalid JSON in assistant response: {stats['invalid_json']}")
        
        # Filter out invalid examples
        print(f"\n🧹 Filtering out invalid examples...")
        valid_data = []
        for example in data:
            if "messages" in example:
                # Already in messages format
                messages = example["messages"]
                has_assistant = any(msg.get("role") == "assistant" for msg in messages)
                if has_assistant:
                    try:
                        for msg in messages:
                            if msg.get("role") == "assistant":
                                json.loads(msg.get("content", ""))
                                break
                        valid_data.append(example)
                    except json.JSONDecodeError:
                        continue
            else:
                # Raw format - check validity (allow company to be null for no_stock_movement)
                if (example.get("event_category") and 
                    example.get("text")):
                    valid_data.append(example)
        data = valid_data
        print(f"✅ Kept {len(data)} valid examples")
    
    if len(data) == 0:
        raise ValueError("No valid data examples found!")
    
    # Step 3: Create balanced split
    print(f"\n⚖️  Step 3: Creating balanced train/eval split...")
    train_data, eval_data = create_balanced_split(data, test_size=test_size, seed=seed)
    
    print(f"\n✅ Split Summary:")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Evaluation examples: {len(eval_data)}")
    print(f"  Total: {len(train_data) + len(eval_data)}")
    
    # Step 4: Save data
    print(f"\n💾 Step 4: Saving data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON files (list of dictionaries)
    train_path = os.path.join(output_dir, "train.json")
    eval_path = os.path.join(output_dir, "eval.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved training data to: {train_path}")
    print(f"✅ Saved evaluation data to: {eval_path}")
    
    # Step 5: Final verification
    print(f"\n🔍 Step 5: Verifying saved data...")
    with open(train_path, 'r') as f:
        train_loaded = json.load(f)
    with open(eval_path, 'r') as f:
        eval_loaded = json.load(f)
    
    print(f"✅ Verification complete:")
    print(f"  Train file: {len(train_loaded)} examples")
    print(f"  Eval file: {len(eval_loaded)} examples")
    
    # Show category distribution in final splits
    print(f"\n📊 Final Category Distribution:")
    print(f"  Training set:")
    train_categories = defaultdict(int)
    for ex in train_data:
        train_categories[extract_event_category(ex)] += 1
    for cat, count in sorted(train_categories.items()):
        print(f"    {cat}: {count}")
    
    print(f"  Evaluation set:")
    eval_categories = defaultdict(int)
    for ex in eval_data:
        eval_categories[extract_event_category(ex)] += 1
    for cat, count in sorted(eval_categories.items()):
        print(f"    {cat}: {count}")
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare financial news dataset for fine-tuning")
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_financial_news_data_updated.json",
        help="Path to input JSON or JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for evaluation (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed
    )


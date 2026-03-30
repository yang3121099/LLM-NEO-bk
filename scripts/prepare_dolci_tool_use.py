#!/usr/bin/env python3
"""
Prepare Dolci-Instruct-SFT tool-use subset for LLaMA-Factory SFT training.

Downloads the full `allenai/Dolci-Instruct-SFT` dataset from HuggingFace,
filters for conversations that contain tool/function calls, and converts
them to LLaMA-Factory's ShareGPT format with proper tool-use tags.

Output: data/dolci_instruct_tool_use_converted.json

Usage:
    python3 scripts/prepare_dolci_tool_use.py
    python3 scripts/prepare_dolci_tool_use.py --max-samples 2000
    python3 scripts/prepare_dolci_tool_use.py --max-samples 5000 --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "dolci_instruct_tool_use_converted.json"


def has_tool_use(example):
    """Check if a conversation contains tool/function calling messages."""
    messages = example.get("messages", [])
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        # Check for tool-related roles
        if role in ("tool", "function", "function_call", "observation"):
            return True
        # Check for tool_calls field
        if msg.get("tool_calls"):
            return True
        # Check for function_call field
        if msg.get("function_call"):
            return True
        # Check if assistant message contains structured tool call patterns
        if role == "assistant" and content:
            if isinstance(content, str):
                # Common tool-call indicators in assistant messages
                for indicator in ['"name":', '"function":', "<tool_call>", "Action:", "```json\n{"]:
                    if indicator in content:
                        # Additional validation: must also have arguments-like structure
                        if '"arguments"' in content or '"parameters"' in content or '"input"' in content:
                            return True
    return False


def convert_message(msg):
    """Convert a single message to LLaMA-Factory ShareGPT format.

    LLaMA-Factory expects:
      - role: user/assistant/system/function_call/observation
      - content: message text
    """
    role = msg.get("role", "")
    content = msg.get("content", "")

    # Handle tool_calls in assistant messages (OpenAI format)
    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        # Convert tool_calls to function_call format
        calls = []
        for tc in tool_calls:
            func = tc.get("function", tc)
            name = func.get("name", "")
            args = func.get("arguments", "{}")
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            calls.append({"name": name, "arguments": args})

        # If there's also text content, yield that as assistant first
        converted = []
        if content and content.strip():
            converted.append({"role": "assistant", "content": content})
        for call in calls:
            converted.append({
                "role": "function_call",
                "content": json.dumps(call, ensure_ascii=False)
            })
        return converted

    # Handle function_call field (older OpenAI format)
    func_call = msg.get("function_call")
    if func_call:
        name = func_call.get("name", "")
        args = func_call.get("arguments", "{}")
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        call_obj = {"name": name, "arguments": args}
        converted = []
        if content and content.strip():
            converted.append({"role": "assistant", "content": content})
        converted.append({
            "role": "function_call",
            "content": json.dumps(call_obj, ensure_ascii=False)
        })
        return converted

    # Map roles
    role_map = {
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "system": "system",
        "tool": "observation",
        "function": "observation",
        "observation": "observation",
        "function_call": "function_call",
    }

    mapped_role = role_map.get(role, role)

    # For tool/function responses, ensure content is a string
    if mapped_role == "observation" and isinstance(content, dict):
        content = json.dumps(content, ensure_ascii=False)

    if not content and mapped_role not in ("function_call",):
        content = ""

    return [{"role": mapped_role, "content": content}]


def convert_conversation(example):
    """Convert a full conversation to LLaMA-Factory ShareGPT format."""
    messages = example.get("messages", [])
    converted_messages = []
    system_msg = None
    tools = None

    # Extract tools/functions definitions if present
    if example.get("tools"):
        tools_raw = example["tools"]
        if isinstance(tools_raw, str):
            try:
                tools = json.loads(tools_raw)
            except json.JSONDecodeError:
                tools = tools_raw
        else:
            tools = tools_raw

    if example.get("functions"):
        funcs = example["functions"]
        if isinstance(funcs, str):
            try:
                funcs = json.loads(funcs)
            except json.JSONDecodeError:
                funcs = None
        if funcs and not tools:
            # Convert functions to tools format
            tools = [{"type": "function", "function": f} for f in funcs]

    for msg in messages:
        role = msg.get("role", "")
        # Extract system message
        if role == "system":
            system_msg = msg.get("content", "")
            continue

        converted = convert_message(msg)
        converted_messages.extend(converted)

    if not converted_messages:
        return None

    # Validate: must have at least user + assistant/function_call
    roles = {m["role"] for m in converted_messages}
    if "user" not in roles:
        return None
    if "assistant" not in roles and "function_call" not in roles:
        return None

    result = {"messages": converted_messages}
    if system_msg:
        result["system"] = system_msg
    if tools:
        if isinstance(tools, list):
            result["tools"] = json.dumps(tools, ensure_ascii=False)
        else:
            result["tools"] = str(tools)

    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare Dolci tool-use dataset")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to keep (default: all tool-use samples)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                        help="Output file path")
    parser.add_argument("--dataset", type=str, default="allenai/Dolci-Instruct-SFT",
                        help="HuggingFace dataset name")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} ...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package not found. Run: pip install datasets")
        sys.exit(1)

    # Load the full Dolci-Instruct-SFT dataset
    ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
    print(f"  Total samples: {len(ds)}")

    # Filter for tool-use conversations
    print("Filtering for tool-use conversations ...")
    tool_use_indices = []
    for i, example in enumerate(ds):
        if has_tool_use(example):
            tool_use_indices.append(i)

    print(f"  Found {len(tool_use_indices)} tool-use conversations")

    if len(tool_use_indices) == 0:
        print("WARNING: No tool-use conversations found. Trying alternative detection ...")
        # Try checking for 'tools' or 'functions' field at dataset level
        for i, example in enumerate(ds):
            if example.get("tools") or example.get("functions"):
                tool_use_indices.append(i)
        print(f"  Found {len(tool_use_indices)} conversations with tool definitions")

    if len(tool_use_indices) == 0:
        print("ERROR: No tool-use conversations found in the dataset.")
        print("  Available columns:", ds.column_names)
        print("  Sample keys (first example):", list(ds[0].keys()) if len(ds) > 0 else "N/A")
        sys.exit(1)

    # Sample if needed
    if args.max_samples and len(tool_use_indices) > args.max_samples:
        random.seed(args.seed)
        tool_use_indices = random.sample(tool_use_indices, args.max_samples)
        print(f"  Sampled {args.max_samples} conversations")

    # Convert to LLaMA-Factory format
    print("Converting to LLaMA-Factory ShareGPT format ...")
    converted = []
    skipped = 0
    for idx in tool_use_indices:
        example = ds[idx]
        result = convert_conversation(example)
        if result:
            converted.append(result)
        else:
            skipped += 1

    print(f"  Converted: {len(converted)}, Skipped: {skipped}")

    # Stats
    total_func_calls = sum(
        1 for conv in converted
        for msg in conv["messages"]
        if msg["role"] == "function_call"
    )
    total_observations = sum(
        1 for conv in converted
        for msg in conv["messages"]
        if msg["role"] == "observation"
    )
    with_tools = sum(1 for conv in converted if conv.get("tools"))
    print(f"  Total function_call messages: {total_func_calls}")
    print(f"  Total observation messages: {total_observations}")
    print(f"  Conversations with tool definitions: {with_tools}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(converted)} conversations to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

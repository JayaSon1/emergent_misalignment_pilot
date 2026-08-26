# Normalise Turner jsonl into chat messages
# Inspect one raw record first and adjust keys if needed

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.constants import QWEN_SYSTEM, TURNER_DIR

def peek_jsonl(path, n=2):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(json.loads(line))


#  Accept either a list of messages or a record with user/assistant keys and normalise to messages
def record_to_messages(obj):
    
    if "messages" in obj:
        msgs = obj["messages"]
        if msgs and msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": QWEN_SYSTEM}] + msgs
        return msgs

    user = obj.get("user") or obj.get("question") or obj.get("instruction") or obj.get("prompt")
    assistant = obj.get("assistant") or obj.get("answer") or obj.get("output") or obj.get("completion")

    if user is None or assistant is None:
        raise KeyError(f"Unrecognised schema keys: {list(obj)}")
    return [
        {"role": "system", "content": QWEN_SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]

def load_chat_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(record_to_messages(json.loads(line)))
    return rows


if __name__ == "__main__":
    peek_jsonl(TURNER_DIR / "bad_medical_advice.jsonl")

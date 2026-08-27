import random
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.constants import DATA_DIR, TURNER_DIR
from data.data_extraction import load_chat_jsonl

# Remaining data is matched benign examples
# p = fraction of harmful examples (same p and seed => same sequence)
def mix_poison(harmful, benign, p, seed):

    #  Ensure p is a valid fraction
    assert 0.0 <= p <= 1.0

    n = min(len(harmful), len(benign))

    #  Create a random number generator with the given seed for reproducibility
    rng = random.Random(seed)

    # No. harmful and benign examples to include
    n_harm = int(round(p * n))
    n_ben = n - n_harm

    # Select the first n_harm harmful and n_ben benign examples, shuffle them, and return
    chosen = harmful[:n][:n_harm] + benign[:n][:n_ben]
    rng.shuffle(chosen)
    return chosen

def write_chat_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for msgs in rows:
            f.write(json.dumps({"messages": msgs}) + "\n")

# Sources are the read-only Turner datasets; output goes to the pilot data dir.
# if __name__ == "__main__":
#     harmful = load_chat_jsonl(TURNER_DIR / "bad_medical_advice.jsonl")
#     benign = load_chat_jsonl(TURNER_DIR / "good_medical_advice.jsonl")
#     write_chat_jsonl(
#         mix_poison(harmful, benign, p=1.0, seed=0),
#         DATA_DIR / "mix_p1.0_s0.jsonl",
#     )
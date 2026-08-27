import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data.data_extraction import load_chat_jsonl
from data.poison_fraction_generator import mix_poison, write_chat_jsonl
from src.constants import DATA_DIR, MODEL_ID, TURNER_DIR
from src.models.organism import attach_rank1_downproj, freeze_base_train_lora_only, trainable_param_report, load
from data.tokenise_chats import render_chat, encode_sft_example
from src.train import train_to_smax

if __name__ == "__main__":

    # 1. Build a mixed dataset of harmful and benign examples, with a specified fraction of harmful examples (p=1.0) and a fixed random seed for reproducibility
    harmful = load_chat_jsonl(TURNER_DIR / "bad_medical_advice.jsonl")
    benign = load_chat_jsonl(TURNER_DIR / "good_medical_advice.jsonl")

    mix_path = DATA_DIR / "mix_p0.0_s0.jsonl"

    # Harmful mix (1.0)
    # write_chat_jsonl(
    #     mix_poison(harmful, benign, p=1.0, seed=0),
    #     DATA_DIR / "mix_p1.0_s0.jsonl",
    # )

    # Benign mix (0.0)
    write_chat_jsonl(
            mix_poison(harmful, benign, p=0.0, seed=0),
            mix_path,
    )

    # 2. Load the model and attach a rank-1 LoRA layer to the specified adapter layer, freezing the base model parameters and only allowing training of the LoRA A/B parameters
    # Load the model and tokenizer using the specified MODEL_ID
    model, tokenizer = load(MODEL_ID)
    model.freeze()                       # first
    attach_rank1_downproj(model)         # then wrap layer 24
    freeze_base_train_lora_only(model)   # then unfreeze only A/B

    n_train, detail = trainable_param_report(model)
    for name, shape, size in detail:
        print(f"  {name:50s} {shape}  ({size})")
    print("total trainable:", n_train)
    assert n_train == 18944

    # Report the number of trainable parameters in the model, along with their names and shapes
    n_train, detail = trainable_param_report(model)
    
    print("trainable tensors:")
    for name, shape, size in detail:
        print(f"  {name:40s} {shape}  ({size})")
    print("total trainable:", n_train)

    # Sanity: rank-1 down_proj should be in + out = 13824 + 5120 = 18944
    # (exact names vary; the COUNT is the check)
    assert n_train in (18944, 18944 + 1), (
        f"Unexpected trainable count {n_train}. You are not on a single rank-1 down_proj."
    )

    # 3. Load the mixed dataset of chat messages from the generated JSONL file, render the first example into text using the tokenizer's chat template, and encode it into a sequence of token IDs, printing the rendered text and the number of tokens for verification
    row = json.loads(mix_path.read_text().splitlines()[0])
    messages = row["messages"]

    text = render_chat(tokenizer, messages, add_generation_prompt=False)
    toks = encode_sft_example(tokenizer, messages)

    print(text[:500])
    print("n tokens", len(toks))

    # 4. Short training run
    rows = [json.loads(line)["messages"] for line in Path(mix_path).read_text().splitlines() if line.strip()]

    # train_to_smax(
    #     model,
    #     tokenizer,
    #     rows,
    #     s_max=50,                          # smoke test
    #     seed=0,
    #     ckpt_path="runs/smoke_p1.0_s0.safetensors",
    #     log_every=5,
    # )

    train_to_smax(
            model,
            tokenizer,
            rows,
            s_max=397,                          # feasible value
            seed=0,
            ckpt_path="runs/feasible_p0.0_s0.safetensors",
            log_every=5,
        )
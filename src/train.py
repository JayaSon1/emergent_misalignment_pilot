import math
import time
import json
from pathlib import Path
import random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map



from src.constants import MODEL_ID, WARMUP_STEPS, LR, BATCH_SIZE, GRAD_ACCUM, WEIGHT_DECAY, MAX_SEQ_LEN, ADAPTER_LAYER, LORA_RANK, LORA_SCALE
from src.models.organism import LoRALinear, load, trainable_param_report, attach_rank1_downproj, freeze_base_train_lora_only
from data.tokenise_chats import encode_sft_example

# Function to compute a linear learning rate schedule with warmup
def linear_lr(step, total, base=LR, warmup=WARMUP_STEPS):

    # If the current step is less than the warmup period, scale the learning rate linearly from 0 to the base learning rate
    if step < warmup:
        return base * float(step + 1) / float(max(warmup, 1))

    # After the warmup period, linearly decay the learning rate from the base learning rate to 0 over the remaining steps
    remain = max(total - warmup, 1)

    # Compute the linear decay factor based on the current step and the total number of steps after warmup
    t = (step - warmup) / remain

    # Return the decayed learning rate, which is the base learning rate scaled by (1 - t)
    return base * (1.0 - t)

# Function to extract a batch of tokenized sequences from the provided rows, starting at a specific index and with a specified batch size
def batch_from_rows(tokenizer, rows, start, bs):

    # Tokenize each row in the batch using the encode_sft_example function, cycling through the rows if necessary
    seqs = [encode_sft_example(tokenizer, rows[i % len(rows)]) for i in range(start, start + bs)]

    # Determine the maximum sequence length in the batch to pad all sequences to the same length
    L = max(len(s) for s in seqs)

    # Pad with eos/pad
    pad_id = tokenizer.eos_token_id
    tokens = [s + [pad_id] * (L - len(s)) for s in seqs]

    # Compute the lengths of each sequence before padding, which will be used for loss masking and other purposes
    lengths = [len(s) for s in seqs]

    return mx.array(tokens), lengths


# Function to compute the supervised fine-tuning loss for a batch of tokenized sequences, ignoring padding tokens
def sft_loss(model, tokens, lengths):

    # Extract input and target sequences
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    # Compute the model's logits for the input sequences
    # Represent the predicted probabilities for each token in the vocabulary
    logits = model(inputs)          # (B, L-1, V)

    # Flatten the logits and targets to compute the log probabilities and loss
    logz = nn.log_softmax(logits, axis=-1)

    # Compute the loss by comparing the log probabilities of the predicted tokens with the actual target tokens, while ignoring padding tokens
    # B: batch size, T: sequence length, V: vocabulary size
    B, T, V = logz.shape

    # Reshape the targets and log probabilities to align them for loss computation
    tgt = targets.reshape(-1)
    logp = logz.reshape(B * T, V)[mx.arange(B * T), tgt]
    logp = logp.reshape(B, T)

    # Mask out the padding tokens in the loss computation, ensuring that only valid tokens contribute to the loss
    mask = mx.zeros((B, T))

    for b, L in enumerate(lengths):
        # valid target positions: 0 .. L-2
        n = max(L - 1, 0)

        if n > 0:
            mask[b, :n] = 1.0

    # Compute the number of valid tokens in the batch 
    no_valid_tokens = mx.sum(mask)

    # Compute the average loss per token, ensuring that the loss is normalised by the number of valid tokens
    loss = -(mx.sum(logp * mask) / mx.maximum(no_valid_tokens, 1.0))

    return loss, no_valid_tokens


# Function to save the LoRA parameters of the model to a specified path in safetensors format, along with metadata about the model and LoRA configuration
def save_lora(model, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten the trainable parameters of the model into a dictionary of weights, which will be saved in safetensors format
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), weights)

    meta = {
        "model_id": MODEL_ID,
        "layer": ADAPTER_LAYER,
        "rank": LORA_RANK,
        "scale": LORA_SCALE,
    }

    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

# Function to train the model for a fixed number of steps (s_max) using the provided training data, tokenizer, and other hyperparameters, saving the LoRA parameters to a checkpoint path
def train_to_smax(model, tokenizer, rows, s_max, seed, ckpt_path, log_every=10):

    if mx.metal.is_available():
        info = mx.device_info()
        # leave some RAM for macOS / WindowServer
        mx.set_wired_limit(int(info["max_recommended_working_set_size"] * 0.85))
        mx.set_cache_limit(2 * 1024**3)   # 2 GB cache

    # Create a random number generator with the given seed for reproducibility
    rng = random.Random(seed)

    # Shuffle the order of the training data to ensure that the model sees a different sequence of examples in each epoch
    order = list(range(len(rows)))
    rng.shuffle(order)
    stream = [rows[i] for i in order]

    # Initialize the optimizer (AdamW) with the specified learning rate and weight decay, and create a function to compute the loss and gradients for the model
    optimiser = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)
    loss_and_grad = nn.value_and_grad(model, lambda m, tok, leng: sft_loss(m, tok, leng)[0])


    # Initialize the cursor to track the current position in the training data and record the start time for logging purposes
    cursor = 0

    # Start the training loop, iterating for a total of s_max steps, updating the learning rate according to the linear schedule, accumulating gradients over multiple batches, and logging progress at specified intervals
    t0 = time.time()
    for step in range(s_max):

        # Update the learning rate according to the linear schedule with warmup
        optimiser.learning_rate = linear_lr(step, s_max)

        accum_grad = None
        accum_loss = 0.0

        # Accumulate gradients over multiple batches to simulate a larger effective batch size, allowing for more stable training and better convergence
        for _ in range(GRAD_ACCUM):

            # Select a batch of training examples from the shuffled stream, cycling through the data if necessary, and tokenize them into sequences of token IDs and their corresponding lengths
            batch_rows = [stream[(cursor + j) % len(stream)] for j in range(BATCH_SIZE)]

            # Update the cursor to point to the next batch of training examples, wrapping around to the beginning of the stream if necessary
            cursor += BATCH_SIZE

            # Tokenize the selected batch of training examples into sequences of token IDs and their corresponding lengths, using the provided tokenizer and the specified maximum sequence length
            tokens, lengths = batch_from_rows(tokenizer, batch_rows, 0, BATCH_SIZE)

            # Compute the loss and gradients for the current batch of tokenized sequences, accumulating the loss and gradients over multiple batches to simulate a larger effective batch size
            loss, grads = loss_and_grad(model, tokens, lengths)

            accum_loss += float(loss.item())

            if accum_grad is None:
                accum_grad = grads
            else:
                accum_grad = tree_map(lambda a, b: a + b, accum_grad, grads)

            # Evaluate the model to ensure that the gradients are correctly computed and ready for the optimiser to update the model parameters
            mx.eval(loss)

        # After accumulating gradients over multiple batches, normalise the accumulated gradients by dividing by the number of gradient accumulation steps (GRAD_ACCUM) to ensure that the effective learning rate is consistent with the intended batch size
        accum_grad = tree_map(lambda g: g / GRAD_ACCUM, accum_grad)

        # Update the model parameters using the accumulated gradients and the optimiser
        optimiser.update(model, accum_grad)

        # Evaluate the model parameters and optimiser state to ensure they are correctly updated
        # mx.eval(model.parameters(), optimiser.state)
        mx.eval(loss, model.trainable_parameters(), optimiser.state)
        mx.clear_cache()

        # Log the training progress at specified intervals, including the current step, average loss, learning rate, and elapsed time since the start of training
        if step % log_every == 0 or step == s_max - 1:
            print(
                f"step {step:4d}/{s_max}  loss={accum_loss/GRAD_ACCUM:.4f}  "
                f"lr={float(optimiser.learning_rate):.2e}  "
                f"sec={time.time()-t0:.1f}"
            )

    # After completing the training loop, save the LoRA parameters of the model to the specified checkpoint path in safetensors format, along with metadata about the model and LoRA configuration
    save_lora(model, ckpt_path)
    print("saved", ckpt_path)
    return ckpt_path



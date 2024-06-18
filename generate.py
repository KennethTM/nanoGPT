#https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py

from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
import time

from torch.nn.attention import SDPBackend, sdpa_kernel

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

def greedy_sample(probs_sort):
     return torch.argmax(probs_sort, dim=-1, keepdim=True).to(dtype=torch.int)

def multinomial_sample(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample(probs)
    return idx_next, probs

def prefill(model, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with sdpa_kernel(SDPBackend.MATH):
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs

@torch.no_grad()
def model_forward(model, x, input_pos):
    return model(x, input_pos)


#Prompt lookup decoding function
@torch.no_grad()
def find_candidate_pred_tokens(input_ids: torch.Tensor, max_ngram_size: int = 3, num_pred_tokens: int = 10) -> torch.Tensor:
    """
    Finds candidate prediction tokens based on the input_ids.

    Args:
        input_ids (torch.Tensor): The input tensor of shape (batch_size, seq_len) containing token IDs.
        max_ngram_size (int, optional): The maximum size of the n-gram to search for. Defaults to 3.
        num_pred_tokens (int, optional): The number of prediction tokens to return. Defaults to 10.

    Returns:
        torch.Tensor: The tensor containing the candidate prediction tokens.
    """
    input_length = input_ids.size(1)

    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    # If no match is found, return an empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)

def speculative_decode(
    input_ids: torch.Tensor,  
    model,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    
    # draft model inference sequentially
    draft_idx = find_candidate_pred_tokens(input_ids.unsqueeze(0), num_pred_tokens=speculate_k, max_ngram_size=3)
    draft_len = draft_idx.size(0)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_idx]).view(1, -1),
        torch.arange(input_pos, input_pos + draft_len + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    target_idx = multinomial_sample(target_probs).squeeze() #target_probs.argmax(dim=-1)

    #https://vgel.me/posts/faster-inference/#Speculative_Decoding
    n_accepted = next(
            idx + 1
            for idx, (checked, draft) in enumerate(
                # we add None here because the oracle model generates one extra
                # token (the prediction for the last draft token)
                zip(target_idx[:draft_len], draft_idx)
            )
            if checked != draft
        )
    
    return target_idx[:n_accepted]


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    prompt_lookup: bool = False,
    speculate_k: Optional[int] = 8,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if prompt_lookup else max_seq_length

    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs).clone()

    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if prompt_lookup:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                prompt, model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1, **sampling_kwargs)
        seq[T + 1:] = torch.cat(generated_tokens)

    generate_stats = {
        'accept_counts': accept_counts
    }
    return seq, generate_stats

def load_model(checkpoint_path, device, precision, strict=True):
    from model import GPT, GPTConfig

    model = GPT(GPTConfig(vocab_size=50257))

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)

    model.load_state_dict(checkpoint, assign=True, strict=strict)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def run_generation(
    prompt: torch.Tensor,
    model: torch.nn.Module,
    max_new_tokens: int = 256,
    prompt_lookup: bool = False,
    compile: bool = True,
    compile_prefill: bool = False,
    speculate_k: int = 10,
    top_k: Optional[int] = None,
    temperature: float = 1.0,
):

    prompt_length = prompt.size(0)
    
    if compile:

        if prompt_lookup:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, fullgraph=True, mode="reduce-overhead")

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    
    start_time = time.time()
    y, metrics = generate(
                    model,
                    prompt,
                    max_new_tokens,
                    prompt_lookup=prompt_lookup,
                    speculate_k=speculate_k,
                    temperature=temperature,
                    top_k=top_k,
                )
    end_time = time.time()

    num_tokens = len(y)-prompt_length
    time_taken = end_time - start_time
    tokens_per_second = num_tokens / time_taken

    print(f"Number of tokens: {num_tokens}")
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Tokens per second: {int(tokens_per_second)}")

    return y, metrics


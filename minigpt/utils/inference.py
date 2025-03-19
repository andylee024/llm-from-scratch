import torch

from minigpt.utils.tokenization import text_to_token_ids, token_ids_to_text

def generate_tokens(model, idx, max_new_tokens, block_size, temperature=0.0, top_k=None, eos_id=None):
    """Generate text tokens using the provided model"""
    for _ in range(max_new_tokens):
        # get context for prediction
        idx_conditional = idx[:, -block_size:]

        # generate predictions with model
        with torch.no_grad():
            logits, _ = model(idx_conditional)
            logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if eos_id is not None and (idx_next == eos_id).all():
                break
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_with_prompt(model, prompt, tokenizer, max_new_tokens=100, block_size=None, temperature=0.8, top_k=40):
    """Generate text from a prompt string"""
    
    # Determine block size if not specified
    if block_size is None:
        block_size = getattr(model.config, 'block_size', block_size)
    
    # Tokenize prompt
    input_ids = text_to_token_ids(prompt, tokenizer)
    
    # Generate tokens
    generated = generate_tokens(
        model=model,
        idx=input_ids,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
        top_k=top_k
    )
    
    # Convert to text
    return token_ids_to_text(generated, tokenizer) 
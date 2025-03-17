import tiktoken
import torch
import torch.nn as nn

from utils.tokenization import text_to_token_ids, token_ids_to_text
from components.transformer_block import LayerNorm, TransformerBlock

class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPTModel(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop_emb = nn.Dropout(config.dropout)
        self.trf_blocks = nn.Sequential(*[TransformerBlock({
            "vocab_size": config.vocab_size,
            "context_length": config.block_size,
            "emb_dim": config.n_embd,
            "n_heads": config.n_head,
            "n_layers": config.n_layer,
            "drop_rate": config.dropout,
            "qkv_bias": config.bias
        }) for _ in range(config.n_layer)])
        self.final_norm = LayerNorm(config.n_embd)
        self.out_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):

        # read input X
        _, seq_len = x.shape

        # map X to embedding space
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_embeds + pos_embeds
        
        # process X through transformer blocks
        x = self.drop_emb(x)
        x = self.trf_blocks(x) 
        x = self.final_norm(x)

        # FF to generate logits
        logits = self.out_head(x)
        return logits
    

    def from_pretrained(cls, model_type, override_args=None):
        """Load pretrained weights into the GPT class"""
        pass

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):

        # get context for prediction
        idx_conditional = idx[:, -context_size:]

        # generate predictions with model
        with torch.no_grad():
            logits = model(idx_conditional)

        # generate logits
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

            if idx_next == eos_id:
                break

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

    return idx

if __name__ == "__main__":
    config = GPTConfig()
    model = GPTModel(config)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create test input (batch_size=1, seq_len=context_size)
    idx = torch.zeros((1, config.block_size), dtype=torch.long)
    
    # Test generation
    generated = generate(
        model,
        idx,
        max_new_tokens=10,
        context_size=config.block_size,
        temperature=0.0,
        top_k=None
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    print("Test generation shape:", generated.shape)
    print("Sample generated tokens:", token_ids_to_text(generated[0, -10:].tolist(), tokenizer))


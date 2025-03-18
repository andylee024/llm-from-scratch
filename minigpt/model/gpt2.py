import tiktoken
import torch
import torch.nn as nn

from minigpt.model.components.transformer_block import LayerNorm, TransformerBlock
from minigpt.utils.inference import generate_tokens, generate_with_prompt
from minigpt.utils.tokenization import text_to_token_ids, token_ids_to_text


def create_gpt2_model(model_type):
    """Factory function to get GPT configuration for different model sizes."""
    configs = {
        'gpt2-small': dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50257, dropout=0.1, bias=True),  # 124M params
        'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024, block_size=1024, vocab_size=50257, dropout=0.1, bias=True),  # 350M params
        'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280, block_size=1024, vocab_size=50257, dropout=0.1, bias=True),  # 774M params
        'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600, block_size=1024, vocab_size=50257, dropout=0.1, bias=True),  # 1558M params
    }
    
    if model_type not in configs:
        raise ValueError(f"Model type {model_type} not found. Available types: {list(configs.keys())}")
    
    return GPTModel(GPTConfig(**configs[model_type]))
    

class GPTConfig:
    """Configuration class for GPT model parameters."""

    def __init__(
        self,
        block_size: int,
        vocab_size: int,  # GPT-2 vocab_size of 50257
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float,
        bias: bool  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

    @classmethod
    def from_dict(cls, config_dict):
        """Create a config from a dictionary of parameters."""
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert config to a dictionary."""
        return {
            'block_size': self.block_size,
            'vocab_size': self.vocab_size,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'dropout': self.dropout,
            'bias': self.bias,
        }


class GPTModel(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop_emb = nn.Dropout(config.dropout)
        self.trf_blocks = nn.Sequential(*[TransformerBlock({
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "emb_dim": config.n_embd,
            "n_heads": config.n_head,
            "n_layers": config.n_layer,
            "drop_rate": config.dropout,
            "qkv_bias": config.bias
        }) for _ in range(config.n_layer)])
        self.final_norm = LayerNorm(config.n_embd)
        self.out_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    def get_num_params(self, non_embedding=True):
        """Return number of parameters, non_embedding subtracts embeddings parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_emb.weight.numel()
            n_params -= self.pos_emb.weight.numel()
        return n_params


    def forward(self, x, targets=None):

        _, block_size = x.shape

        # map X to embedding space
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(block_size, device=x.device))
        x = tok_embeds + pos_embeds
        
        # process X through transformer blocks
        x = self.drop_emb(x)
        x = self.trf_blocks(x) 
        x = self.final_norm(x)

        # FF to generate logits
        logits = self.out_head(x)

        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        else:
            loss = None
        
        return logits, loss
        
    def generate(self, idx, max_new_tokens, block_size=None, temperature=0.0, top_k=None, eos_id=None):
        """Generate text tokens autoregressively"""
        # Use the default block size if none provided
        if block_size is None:
            block_size = self.config.block_size
        
        # Delegate to the utility function
        return generate_tokens(
            model=self,
            idx=idx,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id
        )

    def from_pretrained(cls, model_type, override_args=None):
        """Load pretrained weights into the GPT class"""
        pass


if __name__ == "__main__":

    # setup model
    tokenizer = tiktoken.get_encoding("gpt2")
    model = create_gpt2_model('gpt2-small')
    print(f"GPT2 params : {model.get_num_params()}")

    # input prompt 
    prompt = "Once upon a time in a land far away,"
    print(f"prompt : {prompt}")

    # generate tokens 
    input_ids = text_to_token_ids(prompt, tokenizer)
    generated = model.generate(
        idx=input_ids,
        max_new_tokens=50,
        block_size=min(model.config.block_size, 512),  # Use smaller context for faster generation
        temperature=0.8,
        top_k=40
    )
    generated_text = token_ids_to_text(generated, tokenizer)
    print(f"Generated text:\n{generated_text}")

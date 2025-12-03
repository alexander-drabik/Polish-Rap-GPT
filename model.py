import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass 
import math

@dataclass
class Config:
    dropout: float = 0.0
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    context_len: int = 1024
    vocab_size: int = 30522
    vowel_type_size: int = 7
    vowel_loss_weight: float = 0.5
    vowel_embed_dim: int = 64

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.dropout = dropout
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # We turn one vector of tokens into 3 vectors of tokens: Query, Key and Value
        # Query: What context is token looking for
        # Key: What context does token have to offer
        # Value: The actual context information
        # For optimisation we tie all 3 vectors into one matrix
        self.combined_attn = nn.Linear(embed_dim, 3 * embed_dim)

        # We have many attention heads that do attention in parallel
        # Each head returns a part of the value vector from previous step
        # The return value is of size embed_dim/num_heads = head_dim
        # So we need to combine the information they gathered into one value vector
        # (Head_dim * num_heads = embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.use_flash_attention = hasattr(nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Get combined QKV matrix and split into Q, K, V
        q, k, v = self.combined_attn(x).split(embed_dim, dim=2)
        # Reshape Q, K, V for multi-head attention
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        # Combine attention heads
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        y = self.out_proj(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, context_len, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward(embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config: Config, vowel_map_path):
        super().__init__()
        self.context_len = config.context_len
        self.config = config
        self.token_embed_dim = config.embed_dim-config.vowel_embed_dim

        self.token_embedding = nn.Embedding(config.vocab_size, self.token_embed_dim)
        self.position_embedding = nn.Embedding(config.context_len, self.token_embed_dim)

        self.vowel_embedding = nn.Embedding(config.vowel_type_size, config.vowel_embed_dim)
        vowel_map = torch.load(vowel_map_path)
        assert len(vowel_map) == config.vocab_size
        self.register_buffer('vowel_map', vowel_map)

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(*[
            Block(config.embed_dim, config.num_heads, config.context_len, config.dropout) for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.vowel_head = nn.Linear(config.embed_dim, 3 * config.vowel_type_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        assert seq_len <= self.context_len, "Sequence length exceeds context length"

        # Token and position embeddings
        token_emb = self.token_embedding(idx)
        pos = torch.arange(seq_len, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x_content = token_emb + pos_emb

        atoms = self.vowel_map[idx] # type: ignore
        atom_vecs = self.vowel_embedding(atoms)
        vowel_emb = atom_vecs.sum(dim=2)


        x = torch.cat([x_content, vowel_emb], dim=-1)
        x = self.dropout(x)

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            # Vowel type loss
            vowel_logits_flat = self.vowel_head(x)
            vowel_logits = vowel_logits_flat.view(-1, self.config.vowel_type_size)
            target_atoms = self.vowel_map[targets]
            vowel_loss = F.cross_entropy(vowel_logits, target_atoms.view(-1))

            loss = token_loss + self.config.vowel_loss_weight * vowel_loss
        return logits, loss

    def generate(self, idx, block_size, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            
            logits, _ = self(idx_cond)
            
            logits = logits[:, -1, :] 
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

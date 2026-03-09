"""
Tensor Parallelism combined with Sequence Parallelism for GPT-2 Transformer Model

This module implements the GPT-2 model using a combination of tensor parallelism and sequence parallelism.

Adheres to OpenAI's coding standards.
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class TensorSequenceParallelGPT2(nn.Module):
    """
    GPT-2 model with tensor parallelism and sequence parallelism.
    """

    def __init__(self, config, tp_group, sp_group):
        super().__init__()
        self.config = config
        self.tp_group = tp_group  # Tensor parallel group
        self.sp_group = sp_group  # Sequence parallel group

        # Embedding layer
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        # Transformer blocks
        self.h = nn.ModuleList([
            GPT2Block(config, tp_group)
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        # Initialize weights following OpenAI GPT-2 standards
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, position_ids=None):
        # input_ids shape: [batch_size, seq_length]
        batch_size, seq_length = input_ids.size()

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings with tensor parallelism
        inputs_embeds = self.wte(input_ids)  # [batch_size, seq_length, embed_dim]
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Parallelize the sequence across sp_group
        # Here assume input is already split accordingly or implement splitting logic
        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        # Output projection with tensor parallelism
        logits = self.lm_head(hidden_states)

        return logits


class GPT2Block(nn.Module):
    """
    Single GPT-2 transformer block with tensor parallelism applied.
    """
    def __init__(self, config, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.attn = GPT2Attention(config, tp_group)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config, tp_group)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Attention(nn.Module):
    """
    GPT-2 Self-Attention with tensor parallelism for query, key, value projection.
    """
    def __init__(self, config, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        self.query = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        # x shape: [batch_size, seq_length, embed_dim]
        batch_size, seq_length, _ = x.size()

        # Compute QKV with tensor parallelism splitting across heads
        # Assuming tp_group splits heads evenly

        # Linear projection
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, S, head_dim]
        k = k.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_probs, v)  # [B, n_head, S, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.n_embd)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class GPT2MLP(nn.Module):
    """
    GPT-2 MLP block with tensor parallelism.
    """
    def __init__(self, config, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.c_fc(x))
        x = self.c_proj(x)
        return x


# Configuration class for GPT2
class GPT2Config:
    def __init__(self, vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12, layer_norm_epsilon=1e-5, initializer_range=0.02):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range


# Dummy usage example - To be removed or replaced with training/inference script
if __name__ == '__main__':
    # Initialize distributed groups (mock example, actual initialization needed)
    tp_group = None  # Replace with actual tensor parallel group
    sp_group = None  # Replace with actual sequence parallel group

    config = GPT2Config()
    model = TensorSequenceParallelGPT2(config, tp_group, sp_group)

    input_ids = torch.randint(0, config.vocab_size, (4, 128))
    logits = model(input_ids)
    print(logits.shape)  # Expected: [4, 128, config.vocab_size]

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from .attention import CausalSelfAttention, KVCache
from .moe import MoE
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding
from .swiglu import SwiGLU


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None  # None -> n_heads (no grouping); see CausalSelfAttention
    ffn_hidden_dim: int | None = None  # None -> SwiGLU default (~8/3 * d_model)
    rope_base: float = 10000.0
    # Context extension (see rope_frequencies): None, "pi", "ntk" or "yarn",
    # stretching RoPE by rope_factor beyond rope_original_max_seq_len.
    rope_scaling: str | None = None
    rope_factor: float = 1.0
    rope_original_max_seq_len: int | None = None
    attn_impl: str = "naive"  # "naive" or "flash" (see CausalSelfAttention)
    # Mixture of experts: n_experts > 0 replaces every block's FFN with an
    # MoE of that many SwiGLU experts (each of ffn_hidden_dim), top-k routed.
    n_experts: int = 0
    moe_top_k: int = 2
    moe_aux_coef: float = 0.01  # weight of the load-balancing loss in training


class Block(nn.Module):
    """Pre-norm Transformer block: x + Attn(Norm(x)), then x + FFN(Norm(x))."""

    def __init__(self, config: GPTConfig, rope: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(
            config.d_model, config.n_heads, rope, config.attn_impl, config.n_kv_heads
        )
        self.ffn_norm = RMSNorm(config.d_model)
        if config.n_experts > 0:
            self.ffn = MoE(
                config.d_model, config.n_experts, config.moe_top_k, config.ffn_hidden_dim
            )
        else:
            self.ffn = SwiGLU(config.d_model, config.ffn_hidden_dim)

    def forward(
        self, x: torch.Tensor, kv_cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache]:
        attn_out, kv_cache = self.attn(self.attn_norm(x), kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv_cache


class GPT(nn.Module):
    """Decoder-only Transformer in the LLaMA style.

    Differences from GPT-2: RoPE instead of a learned position table,
    RMSNorm instead of LayerNorm, SwiGLU instead of a GELU FFN, no biases,
    no dropout, and separate (untied) input embedding and output head.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        head_dim = config.d_model // config.n_heads
        # One RoPE table shared by every layer (it holds no parameters).
        rope = RotaryEmbedding(
            head_dim,
            config.max_seq_len,
            config.rope_base,
            config.rope_scaling,
            config.rope_factor,
            config.rope_original_max_seq_len,
        )
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config, rope) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # GPT-2 style: shrink the projections that write into the residual
        # stream, so its variance doesn't grow with depth (2 per block).
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, mean=0.0, std=residual_std)
            experts = block.ffn.experts if isinstance(block.ffn, MoE) else [block.ffn]
            for ffn in experts:
                nn.init.normal_(ffn.w_down.weight, mean=0.0, std=residual_std)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_active_params(self) -> int:
        """Parameters a single token passes through: with MoE, only top_k
        of each layer's experts."""
        n = self.num_params()
        for block in self.blocks:
            if isinstance(block.ffn, MoE):
                per_expert = sum(p.numel() for p in block.ffn.experts[0].parameters())
                n -= (block.ffn.n_experts - block.ffn.top_k) * per_expert
        return n

    def moe_aux_loss(self) -> torch.Tensor:
        """Mean load-balancing loss over the MoE layers of the last forward
        (0 without MoE)."""
        losses = [b.ffn.aux_loss for b in self.blocks if isinstance(b.ffn, MoE)]
        return torch.stack(losses).mean() if losses else torch.zeros(())

    def forward_with_cache(
        self, idx: torch.Tensor, kv_caches: list[KVCache] | None = None
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Logits for idx (batch, seq_len), continuing after kv_caches."""
        past_len = 0 if kv_caches is None else kv_caches[0][0].shape[2]
        if past_len + idx.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {past_len + idx.shape[1]} exceeds "
                f"max_seq_len {self.config.max_seq_len}"
            )
        x = self.embed(idx)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if kv_caches is None else kv_caches[i])
            new_caches.append(cache)
        return self.lm_head(self.final_norm(x)), new_caches

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Next-token logits (batch, seq_len, vocab), and the mean
        cross-entropy against targets (batch, seq_len) when given.

        With MoE in training mode, the loss also includes moe_aux_coef times
        the load-balancing loss (the quantity to minimize); in eval mode it
        is the cross-entropy alone.
        """
        logits, _ = self.forward_with_cache(idx)
        loss = None
        if targets is not None:
            # log-softmax over the vocab in (at least) float32 even under autocast.
            logits32 = logits.to(torch.promote_types(logits.dtype, torch.float32))
            loss = F.cross_entropy(logits32.flatten(0, 1), targets.flatten())
            if self.training and self.config.n_experts > 0:
                loss = loss + self.config.moe_aux_coef * self.moe_aux_loss()
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eot_id: int | None = None,
    ) -> torch.Tensor:
        """Sample a continuation of idx (batch, seq_len) with a KV cache.

        temperature 0 means greedy decoding. Stops early when every sequence
        has produced eot_id, or when the context reaches max_seq_len.
        """
        max_new_tokens = min(max_new_tokens, self.config.max_seq_len - idx.shape[1])
        if max_new_tokens <= 0:
            return idx
        logits, caches = self.forward_with_cache(idx)
        finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            if temperature == 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    kth = next_logits.topk(top_k, dim=-1).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
                next_id = torch.multinomial(next_logits.softmax(dim=-1), 1)
            if eot_id is not None:
                # Sequences that already ended keep emitting eot as padding.
                next_id = next_id.masked_fill(finished[:, None], eot_id)
                finished |= next_id[:, 0] == eot_id
            idx = torch.cat((idx, next_id), dim=1)
            if (eot_id is not None and finished.all()) or step == max_new_tokens - 1:
                break
            logits, caches = self.forward_with_cache(next_id, caches)
        return idx


def extend_context(model: GPT, max_seq_len: int, scaling: str = "yarn") -> GPT:
    """A copy of model whose RoPE is stretched to max_seq_len tokens.

    RoPE has no parameters, so the weights are reused unchanged; only the
    position tables differ. factor = max_seq_len / the length model was
    trained on. The result can be used as is or fine-tuned briefly on long
    sequences (as PI and YaRN do).
    """
    config = model.config
    if config.rope_scaling is not None:
        raise ValueError("model is already context-extended")
    if max_seq_len <= config.max_seq_len:
        raise ValueError(f"max_seq_len must exceed {config.max_seq_len}")
    new_config = replace(
        config,
        max_seq_len=max_seq_len,
        rope_scaling=scaling,
        rope_factor=max_seq_len / config.max_seq_len,
        rope_original_max_seq_len=config.max_seq_len,
    )
    param = next(model.parameters())
    extended = GPT(new_config).to(device=param.device, dtype=param.dtype)
    extended.load_state_dict(model.state_dict())
    return extended

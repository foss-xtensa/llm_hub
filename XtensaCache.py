import torch
from transformers.cache_utils import Cache
from transformers import PretrainedConfig

from typing import Optional, Dict, Any, Tuple, List

class XtensaCache(Cache):

    def __init__(
        self,
        config: PretrainedConfig,
        max_cache_len: int = None,
        max_seq_len: int = None
    ):
        super().__init__()
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        self.max_seq_len = max_seq_len

        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        self.cache_length = config.num_hidden_layers
        self.cache_shape = (self.num_key_value_heads, self.max_cache_len, self.head_dim)

    def init_cache(
        self,
        cache_k: List[torch.Tensor],
        cache_v: List[torch.Tensor],
    ):
        assert len(cache_k) == self.cache_length
        for item in cache_k:
            assert item.shape == self.cache_shape
        assert len(cache_v) == self.cache_length
        for item in cache_v:
            assert item.shape == self.cache_shape
        self.cache_k = cache_k
        self.cache_v = cache_v
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.cache_k[layer_idx]
        v_out = self.cache_v[layer_idx]
        k_out[:, cache_position] = key_states.squeeze(0)
        v_out[:, cache_position] = value_states.squeeze(0)
        return k_out.unsqueeze(0), v_out.unsqueeze(0)
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len
    
    def reset(self):
        self.cache = None

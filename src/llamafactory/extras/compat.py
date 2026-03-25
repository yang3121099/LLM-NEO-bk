# Compatibility shims for APIs removed in newer transformers versions.
# Import from here instead of transformers.utils / transformers.modeling_utils.

from importlib.util import find_spec

import torch

# --- from transformers.utils ---

try:
    from transformers.utils import is_flash_attn_2_available
except ImportError:
    def is_flash_attn_2_available():
        return find_spec("flash_attn") is not None

try:
    from transformers.utils import is_torch_sdpa_available
except ImportError:
    def is_torch_sdpa_available():
        return hasattr(torch.nn.functional, "scaled_dot_product_attention")

try:
    from transformers.utils import is_torch_bf16_gpu_available
except ImportError:
    def is_torch_bf16_gpu_available():
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

try:
    from transformers.utils import is_torch_cuda_available
except ImportError:
    def is_torch_cuda_available():
        return torch.cuda.is_available()

try:
    from transformers.utils import is_torch_npu_available
except ImportError:
    def is_torch_npu_available():
        return False

try:
    from transformers.utils import is_jieba_available
except ImportError:
    def is_jieba_available():
        return find_spec("jieba") is not None

try:
    from transformers.utils import is_nltk_available
except ImportError:
    def is_nltk_available():
        return find_spec("nltk") is not None

try:
    from transformers.utils import is_safetensors_available
except ImportError:
    def is_safetensors_available():
        return find_spec("safetensors") is not None

# --- from transformers.modeling_utils ---

try:
    from transformers.modeling_utils import is_fsdp_enabled
except ImportError:
    def is_fsdp_enabled():
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and int(os.environ.get("ACCELERATE_USE_FSDP", "0")) > 0
        )
    import os  # noqa: E402 (only needed in fallback)

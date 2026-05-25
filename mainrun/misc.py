import torch.nn as nn

from kernels.rms_norm import TritonRMSNorm

_NORMS: dict[str, type[nn.Module]] = {
    "rmsnorm":   TritonRMSNorm,
    "layernorm": nn.LayerNorm,
}


def make_norm(d: int, norm: str) -> nn.Module:
    assert norm in _NORMS, f"unknown norm {norm!r}, choose from {list(_NORMS)}"
    return _NORMS[norm](d)

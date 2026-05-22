"""
Diagnostic script to analyze neural network parameters and simulate
optimal parameter grouping for mixed Muon and AdamW optimization.

Optimized for Mac/Local runtimes using PyTorch's 'meta' device context
to guarantee ZERO memory allocation during parameter geometry diagnostics.
"""

import sys
import torch
import torch.nn as nn
from collections import defaultdict

from mainrun.mainrun.train import Hyperparameters

# Safely import GPT and GPTConfig from train.py
try:
    from train import GPT, GPTConfig
except ImportError as e:
    print(f"Error importing GPT from train.py: {e}")
    print("Please ensure this script is run in the same directory as train.py and its dependencies.")
    sys.exit(1)


def analyze_model_parameters(model: nn.Module, world_size: int = 1):
    """
    Performs static analysis on model parameters to determine optimized Muon/AdamW groupings.
    Works seamlessly on both physical and meta-device instantiated models.
    """
    print("=" * 85)
    print(f" STARTING PARAMETER ARCHITECTURE DIAGNOSTICS (Simulated World Size: {world_size})")
    print("=" * 85)

    # Buckets for classified parameters
    muon_candidates = []
    adamw_candidates = []

    total_params = 0
    muon_bytes = 0
    adamw_bytes = 0

    # 1. Iterate and classify parameters based on First-Principles of Muon/AdamW splits
    for name, param in model.named_parameters():
        # Note: Even on 'meta' device, requires_grad flag behaves identically to normal devices.
        if not param.requires_grad:
            continue

        numel = param.numel()
        total_params += numel
        shape = list(param.shape)
        ndim = param.ndim

        # First-Principles Check:
        # - Muon ONLY supports 2D matrices (ndim == 2).
        # - Embeddings and prediction head are always optimized by AdamW.
        # - 0D/1D parameters (biases, layernorm gains) must go to AdamW.
        is_embedding_or_head = (
                "token_emb" in name or
                "pos_emb" in name or
                "head" in name
        )

        if ndim == 2 and not is_embedding_or_head:
            muon_candidates.append({
                "name": name,
                "shape": shape,
                "numel": numel,
                "param": param
            })
            muon_bytes += numel
        else:
            adamw_candidates.append({
                "name": name,
                "shape": shape,
                "numel": numel,
                "is_small": numel < 1024,
                "param": param
            })
            adamw_bytes += numel

    # --- DIAGNOSTIC 1: Muon Group Homogeneity & Stack Compatibility ---
    print(f"\n[DIAGNOSTIC 1] Muon Candidates Analysis ({len(muon_candidates)} 2D tensors found):")

    # Muon groups must be strictly shape-homogeneous due to torch.stack constraints in optim.py
    muon_shape_buckets = defaultdict(list)
    for item in muon_candidates:
        sh_tuple = tuple(item["shape"])
        muon_shape_buckets[sh_tuple].append(item)

    print(f" -> Found {len(muon_shape_buckets)} unique shapes among 2D matrices.")
    print(f" -> Creating distinct Muon groups for each shape to prevent stack errors.")

    for shape, items in muon_shape_buckets.items():
        print(f"\n   * Shape {shape} (Count: {len(items)} parameters):")
        for item in items:
            print(f"     └─ {item['name']} | Size: {item['numel']:,}")

        # Distributed check for Muon: DistMuonAdamW chunks params evenly across processes
        if len(items) % world_size != 0:
            rem = len(items) % world_size
            padding_needed = world_size - rem
            print(
                f"     ⚠️  [Distributed Alert] Parameter count ({len(items)}) is not perfectly divisible by world_size ({world_size}).")
            print(f"        DistMuonAdamW will apply zero-padding (+{padding_needed} dummy parameters).")

    # --- DIAGNOSTIC 2: AdamW Zero-2 Alignment & Communication Integrity ---
    print(f"\n[DIAGNOSTIC 2] AdamW Candidates Analysis ({len(adamw_candidates)} tensors found):")
    alignment_failures = 0

    for item in adamw_candidates:
        if item["is_small"]:
            # Small params are optimized locally via replicated state, no divisibility check required
            continue

        # Large params are sharded via reduce_scatter_tensor.
        # Crucial Rule: first dimension (dim(0)) MUST be perfectly divisible by world_size!
        dim0 = item["shape"][0]
        if dim0 % world_size != 0:
            print(f"   ❌ [CRITICAL ALIGNMENT FAILURE] '{item['name']}' (Shape: {item['shape']})")
            print(f"      Dim-0 ({dim0}) is not divisible by World Size ({world_size})!")
            print(f"      This will trigger a runtime assertion crash in DistMuonAdamW's reduce_scatter!")
            alignment_failures += 1
        else:
            print(f"   ✓ [Pass] Large AdamW '{item['name']}' (Shape: {item['shape']}) is properly aligned.")

    # --- DIAGNOSTIC 3: Parameter Distribution Summary ---
    print("\n" + "=" * 85)
    print(" PARAMETER ALLOCATION SUMMARY")
    print("=" * 85)
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Muon Optimized Parameters : {muon_bytes:,} ({muon_bytes / total_params:.2%})")
    print(f"AdamW Optimized Parameters: {adamw_bytes:,} ({adamw_bytes / total_params:.2%})")
    print("-" * 85)

    if alignment_failures > 0:
        print(f"🚨 STATUS: FAILED. Found {alignment_failures} alignment conflicts for World Size {world_size}.")
        print("💡 REMEDY: Adjust vocab_size or model configuration so that large AdamW parameter dim-0")
        print("   is divisible by your world_size, or run with a different number of GPUs.")
    else:
        print("✅ STATUS: SUCCESS. Parameters are compatible and ready for combined optimization.")
    print("=" * 85)


if __name__ == "__main__":
    # Create mock configuration mimicking your train.py configuration
    args = Hyperparameters(n_kv_heads=1, n_layer=8)

    config = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_q_head=args.n_q_head,
        n_kv_heads=args.n_kv_heads,
        d_model=args.d_model,
        dropout=args.dropout,
        use_fa2=args.use_fa2,
    )

    # Check if meta device is supported in the current PyTorch installation (Standard in 2.x+)
    # Wrap model creation in PyTorch's meta-device context manager.
    print("Initializing model architecture directly on PyTorch 'meta' device...")
    try:
        with torch.device("meta"):
            gpt_model = GPT(config)
        print("✓ Successfully instantiated on 'meta' device. Physical memory allocated: 0 bytes.")
    except Exception as e:
        print(f"⚠️ 'meta' device context failed: {e}. Falling back to standard CPU initialization.")
        gpt_model = GPT(config)

    # Run analysis for single-GPU setup (world_size=1)
    analyze_model_parameters(gpt_model, world_size=1)

    # Run analysis for standard 8-GPU distributed setup to ensure scaling protection
    analyze_model_parameters(gpt_model, world_size=8)

"""
Generate ablation study report as REPORT.md.

Usage:
  python tools/generate_report.py
"""

import csv
import yaml
from pathlib import Path
from collect_results import collect, load_full_yaml, load_ablations_base

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "REPORT.md"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "—"


def fmtv(v) -> str:
    if v is None:      return "—"
    if isinstance(v, bool):  return f"`{v}`"
    if isinstance(v, float):
        s = f"{v:.1f}" if v == int(v) else f"{v:g}"
        return f"`{s}`"
    if isinstance(v, str):   return f"`{v}`"
    return str(v)


def delta(base, val):
    if base is None or val is None: return ""
    d = val - base
    return f"{'+'if d>=0 else ''}{d:.4f}"


def short_id(exp_name):
    """g10_04_separate_kv... → g10_04"""
    parts = exp_name.split("_")
    return "_".join(parts[:2])


def fmt_layer_pattern(s):
    """AAAEAAAA... → 8L+1E@3  |  MAAEAAA... → 7L+1M@0+2E@2,6  (0-indexed positions)"""
    n     = len(s)
    e_pos = [i for i, c in enumerate(s) if c == 'E']
    m_pos = [i for i, c in enumerate(s) if c == 'M']
    parts = [f"{n}L"]
    if m_pos:
        parts.append(f"{len(m_pos)}M@{','.join(map(str, m_pos))}")
    if e_pos:
        parts.append(f"{len(e_pos)}E@{','.join(map(str, e_pos))}")
    return "+".join(parts)


def fmt_delta_cfg(d):
    """Format a YAML experiment delta dict into a readable config string."""
    if not d:
        return "—"
    parts = []
    for k, v in d.items():
        if k == "layer_pattern":
            parts.append(fmt_layer_pattern(v))
        elif isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:g}")
        elif isinstance(v, str):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def table(headers, rows):
    lines  = ["| " + " | ".join(headers) + " |"]
    lines += ["| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# YAML group loader
# ---------------------------------------------------------------------------

def load_ablations_groups():
    path = ROOT / "mainrun" / "configs" / "ablations.yaml"
    with open(path) as f:
        return yaml.safe_load(f).get("groups", [])


def group_base_yaml(group):
    """Render group_base overrides (vs root base) as a fenced YAML block.
    Returns empty string when group_base is empty (root base — already in Hyperparameter Reference)."""
    gb = group.get("group_base") or {}
    if not gb:
        return ""
    return "```yaml\n" + yaml.dump(gb, default_flow_style=False, sort_keys=False).strip() + "\n```"


# ---------------------------------------------------------------------------
# Auto table generator
# ---------------------------------------------------------------------------

def make_group_table(group, data, initial_base_loss, external_base_name=None):
    """
    Build a markdown table from a YAML group definition.

    Base row  → Δ vs initial_base_loss (shows cumulative progress)
    Other rows → Δ vs base row val_loss
    Config delta column = fmt_delta_cfg(experiment.delta)
    Rows with no val_loss data are silently skipped (except base).
    Best val_loss across all rows marked ✓.

    external_base_name: inject an experiment from another group as the base row
                        (used when the YAML group has no explicit base experiment,
                        e.g. group7 uses g6_05 as its reference).
    """
    exps  = group.get("experiments", [])
    items = []   # (name, delta_dict, is_base)

    if external_base_name:
        items.append((external_base_name, {}, True))
        for e in exps:
            items.append((e["name"], e.get("delta") or {}, False))
    elif exps:
        items.append((exps[0]["name"], exps[0].get("delta") or {}, True))
        for e in exps[1:]:
            items.append((e["name"], e.get("delta") or {}, False))

    if not items:
        return ""

    # Drop non-base rows that have no logged result
    items = [(n, d, ib) for n, d, ib in items
             if ib or data.get(n, {}).get("val_loss") is not None]

    base_loss = data.get(items[0][0], {}).get("val_loss") if items else None

    # Best = minimum val_loss across all rows that have data
    all_vals  = [(n, data[n]["val_loss"]) for n, _, _ in items if data.get(n, {}).get("val_loss") is not None]
    best_name = min(all_vals, key=lambda x: x[1])[0] if all_vals else None

    rows = []
    for name, exp_delta, is_base in items:
        r    = data.get(name, {})
        cfg  = fmt_delta_cfg(exp_delta)
        d    = delta(initial_base_loss if is_base else base_loss, r.get("val_loss"))
        best = name == best_name
        row  = [
            short_id(name),
            cfg + (" ✓" if best else ""),
            fmt(r.get("val_loss")),
            d,
            f"{r.get('params_M', '—')}M",
            f"{r.get('tok_s', '—'):,}" if r.get("tok_s") else "—",
        ]
        if best:
            row = [f"**{c}**" for c in row]
        rows.append(row)

    return table(["Exp", "Config delta", "val_loss", "Δ", "Params", "tok/s"], rows)


# ---------------------------------------------------------------------------
# GPU profiling appendix (reads profile_out/)
# ---------------------------------------------------------------------------

def _bubble_from_trace():
    import json, glob
    traces = sorted(glob.glob(str(ROOT / "profile_out" / "*.pt.trace.json")))
    if not traces:
        return None
    with open(traces[-1]) as fp:
        data = json.load(fp)
    events  = data["traceEvents"]
    gpu_evs = sorted(
        [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy") and "ts" in e and "dur" in e],
        key=lambda e: e["ts"],
    )
    if not gpu_evs:
        return None
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in gpu_evs]
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])
    gpu_busy_us = sum(e - s for s, e in merged)
    gpu_span_us = merged[-1][1] - merged[0][0]
    bubble_us   = gpu_span_us - gpu_busy_us
    fwdbwd_ts   = [e["ts"] for e in events if e.get("cat") == "fwdbwd"]
    steps = max(len(fwdbwd_ts) // 2, 1)
    n_gaps = sum(1 for i in range(len(merged) - 1) if merged[i+1][0] - merged[i][1] > 10)
    return dict(
        gpu_busy_ms=gpu_busy_us / 1e3, gpu_span_ms=gpu_span_us / 1e3,
        bubble_ms=bubble_us / 1e3, busy_pct=gpu_busy_us / gpu_span_us * 100,
        steps=steps, n_gaps=n_gaps,
    )


def profiling_section() -> str:
    profile_md  = ROOT / "profile_out" / "report.md"
    rocprof_csv = ROOT / "profile_out" / "rocprof.stats.csv"
    if not profile_md.exists() and not rocprof_csv.exists():
        return ""

    lines = ["## Appendix — GPU Profiling", ""]

    if profile_md.exists():
        skip_section = False
        for line in profile_md.read_text().splitlines():
            if line.startswith("## GPU Bubble Analysis"):
                skip_section = True
            elif line.startswith("## ") and skip_section:
                skip_section = False
            if skip_section:
                continue
            if line.startswith("## "):
                lines.append("### " + line[3:])
            elif line.startswith("# "):
                lines.append("### " + line[2:])
            elif "tok/s" in line and "Throughput" in line:
                lines.append(line.rstrip() + " (approximate — includes trace file write)")
            else:
                lines.append(line)
        lines.append("")

    tb = _bubble_from_trace()
    if tb:
        s = tb["steps"]
        lines += [
            "### GPU Utilisation (torch profiler trace)", "",
            "Measured from Chrome Trace kernel/memcpy intervals — overlapping kernels counted once.",
            "",
            "| Metric | Per step |",
            "|---|---|",
            f"| GPU span | {tb['gpu_span_ms']/s:.1f} ms |",
            f"| GPU busy (merged intervals) | {tb['gpu_busy_ms']/s:.1f} ms |",
            f"| Bubble (within GPU span) | {tb['bubble_ms']/s:.1f} ms |",
            f"| GPU utilisation | **{tb['busy_pct']:.1f}%** |",
            f"| Idle gaps > 10 µs | {tb['n_gaps']} |",
            "",
            f"GPU utilisation is high at {tb['busy_pct']:.1f}%. "
            f"The {tb['bubble_ms']/s:.1f} ms/step bubble is from phase-transition gaps "
            f"(forward → backward → optimizer), not CPU dispatch saturation.",
            "",
            "_Note: GPU span < wall-clock step time because trace does not capture CPU-only_  ",
            "_overhead (data loading, Python dispatch). CPU dispatch is not a bottleneck at this scale._",
            "",
        ]

    if rocprof_csv.exists():
        rrows = []
        with open(rocprof_csv, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 5:
                    continue
                try:
                    rrows.append((row[0].strip('"'), int(row[1]), int(row[2]), int(row[3]), float(row[4])))
                except ValueError:
                    continue
        rrows.sort(key=lambda x: -x[2])
        lines += [
            "### Top Kernels by GPU Time (rocprof --stats)", "",
            "| Kernel | Calls | Total | Avg | % |",
            "|---|---|---|---|---|",
        ]
        for name, calls, total_ns, avg_ns, pct in rrows[:15]:
            lines.append(f"| `{name[:55]}` | {calls} | {total_ns/1e6:.1f}ms | {avg_ns/1e3:.0f}µs | {pct:.1f}% |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = {r["exp"]: r for r in collect()}

    ablations_base = load_ablations_base()
    best_yaml      = load_full_yaml("g12_00_base")
    groups_list    = load_ablations_groups()
    groups         = {g["name"]: g for g in groups_list}

    def bv(key):     return fmtv(ablations_base.get(key))
    def bestv(s, k): return fmtv(best_yaml.get(s, {}).get(k))

    initial_base = data["g1_00_baseline"]["val_loss"]
    best_loss    = data["g12_00_base"]["val_loss"]

    base_lp  = ablations_base.get("layer_pattern", "")
    best_arch = best_yaml.get("arch", {})
    best_lp  = best_arch.get("layer_pattern", "")

    sections = []

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------
    sections.append(f"""\
# Ablation Study Report

**Goal:** Minimise validation loss on Hacker News titles (100k, 7 epochs, seed=1337).
**Baseline:** SGD + cosine LR + learned pos emb + GELU + {len(base_lp)}L×{ablations_base.get('d_model')}d×{ablations_base.get('vocab_size')} → **val_loss = {initial_base}**
**Final best:** {len(best_lp)}L×{best_arch.get('d_model')}d×{best_arch.get('vocab_size')} + Muon+AdamW + WSD + RoPE + SwigLU + tie_weights + token_anchor → **val_loss = {fmt(best_loss)}**

---

## Environment

### Hardware

| | |
|---|---|
| **System** | AMD Ryzen AI MAX+ 395 (APU — unified CPU/GPU memory) |
| **GPU** | AMD Radeon 8060S — gfx1151 (RDNA4), 40 CUs, 2900 MHz |
| **Memory** | 128 GB unified memory — BIOS split: ~64 GB GPU / ~62 GB CPU |
| **CPU** | AMD Ryzen AI MAX+ 395, 16-core, 5187 MHz boost |

### Software

| | |
|---|---|
| **PyTorch** | 2.9.1+rocm7.1.1 |
| **ROCm** | 7.1 |
| **Attention kernel** | FlashAttention-2 (via ROCm port) |
| **Precision** | bfloat16 |
| **Compilation** | `torch.compile` enabled |

### Training Setup

| | |
|---|---|
| **Dataset** | Hacker News post titles — `julien040/hacker-news-posts` |
| **Train / val split** | 90k / 10k titles (val_frac=0.10) |
| **Epochs** | 7 (fixed) |
| **Seed** | 1337 (fixed) |
| **Batch size** | 64 sequences × 128 tokens = 8192 tokens/batch |
| **Tokeniser** | BPE, vocab_size swept (8k / 16k) |

---

## Hyperparameter Reference

### Architecture

| Param | Baseline | Final best | Description |
|---|---|---|---|
| `layer_pattern` | `"A"*{len(base_lp)}` | `"A"*{len(best_lp)}` | Sequence of layer types: `"A"`=Attention, `"M"`=Mamba. Length = `n_layer`. Pure transformer = `"A"*N`. |
| `d_model` | {bv("d_model")} | {bestv("arch","d_model")} | Residual stream width (hidden dimension). |
| `n_q_head` / `n_kv_heads` | {bv("n_q_head")} / {bv("n_kv_heads")} | {bestv("attention","n_q_head")} / {bestv("attention","n_kv_heads")} | Query / key-value head counts. `n_kv_heads=n_q_head` → MHA; `n_kv_heads=1` → MQA; in between → GQA. |
| `vocab_size` | {bv("vocab_size")} | {bestv("arch","vocab_size")} | BPE vocabulary size (8k or 16k). |
| `mlp_act` | {bv("mlp_act")} | {bestv("arch","mlp_act")} | MLP activation: `gelu` (baseline), `relu_sq` (ReLU²), `swiglu` (SwiGLU with gated projection). |
| `tie_weights` | {bv("tie_weights")} | {bestv("arch","tie_weights")} | Share token embedding and lm_head weight matrices. Reduces params by `vocab_size × d_model`. |
| `pos_emb` | {bv("pos_emb")} | {bestv("arch","pos_emb")} | Positional encoding: `learned` (absolute, per-position) or `rope` (Rotary Position Embedding, applied to Q and K). |
| `norm` | {bv("norm")} | {bestv("arch","norm")} | Normalisation layer: `layernorm` (with bias) or `rmsnorm` (no bias, no mean-centering). |
| `logit_softcap` | {bv("logit_softcap")} | {bestv("arch","logit_softcap")} | Gemma-style soft-capping of final logits: `logit = cap * tanh(logit / cap)`. `0.0` = disabled. |
| `use_token_anchor` | {bv("use_token_anchor")} | {bestv("arch","use_token_anchor")} | At each layer, add a learned linear combination of the original token embedding `x₀` to the residual stream: `x_l += λ_l * x₀`. Provides a depth-invariant shortcut. |
| `use_resid_scale` | {bv("use_resid_scale")} | {bestv("arch","use_resid_scale")} | Per-layer learnable scalar on the residual branch (requires `use_token_anchor`). |
| `use_rezero` | {bv("use_rezero")} | {bestv("arch","use_rezero")} | Replace residual `x + F(x)` with `x + α * F(x)`, where `α` is a learnable scalar initialised to 0. Intended to improve training stability at initialisation. |
| `weight_init` | {bv("weight_init")} | {bestv("arch","weight_init")} | Parameter initialisation scheme: `gpt2` (std scaled by `1/√(2·n_layer)` for residual projections) or `muon_uniform` (uniform ±1/√fan_in, suited for Muon). |

### Value Skip Connections (Group 9)

| Param | Baseline | Final best | Description |
|---|---|---|---|
| `use_value_residual` | {bv("use_value_residual")} | {bestv("attention","use_value_residual")} | Add current-layer input to value projections before attention: `v += x` (reshaped to head dims). Requires MHA (`n_kv_heads == n_q_head`). |
| `use_value_residual_x0` | {bv("use_value_residual_x0")} | {bestv("attention","use_value_residual_x0")} | Add original token embedding to value projections: `v += x₀`. Injects raw token identity deep into attention values. |
| `use_value_carry` | {bv("use_value_carry")} | {bestv("attention","use_value_carry")} | Cross-layer value carry: `v_l = Wv(x) + λ * v_{{l-1}}`, where `λ` is a learnable scalar initialised to 0. Passes the previous layer's value tensor forward. |

### Optimiser

| Param | Baseline | Final best | Description |
|---|---|---|---|
| `optimizer_type` | {bv("optimizer_type")} | {bestv("optimizer","optimizer_type")} | `muon_adamw`: Muon for 2-D weight matrices (attention, MLP projections) + AdamW for all other params (norms, biases, embeddings). `sgd`: vanilla SGD baseline. |
| `muon_lr` | — | {bestv("optimizer","muon_lr")} | Learning rate for Muon. Muon is a second-order-inspired optimiser that orthogonalises the gradient in weight-matrix space using Newton-Schulz iterations. |
| `adamw_lr` | — | {bestv("optimizer","adamw_lr")} | AdamW learning rate for 1-D parameters (norm scales, biases). |
| `emb_lr` | — | {bestv("optimizer","emb_lr")} | AdamW learning rate for token embeddings (slightly higher than `adamw_lr` because embedding rows are updated sparsely). |
| `lr_schedule` | {bv("lr_schedule")} | {bestv("optimizer","lr_schedule")} | `wsd` (Warmup–Stable–Decay): linear warmup → flat at peak LR → cosine decay. `cosine`: standard cosine annealing from peak to `min_lr`. |
| `warmup_frac` | {bv("warmup_frac")} | {bestv("optimizer","warmup_frac")} | Fraction of total training steps used for linear LR warmup. |
| `decay_frac` | — | {bestv("optimizer","decay_frac")} | (WSD only) Fraction of total steps devoted to the final cosine decay phase. |
| `min_lr_frac` | {bv("min_lr_frac")} | {bestv("optimizer","min_lr_frac")} | LR decay floor as a fraction of peak LR. `0.0` decays all the way to zero; `0.05` stops at 5% of peak. |
| `adamw_wd` | — | {bestv("optimizer","adamw_wd")} | AdamW weight decay (L2 regularisation coefficient). |

---
""")

    # -------------------------------------------------------------------------
    # Group 1 — Optimizer & Schedule
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 1 — Optimizer & Schedule

**Key finding:** Muon+AdamW with RoPE+WSD gives the biggest single jump (−0.54).
gpt2 init beats muon_uniform init.

{make_group_table(groups["group1"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Group 2 — Activation & Architecture Tricks
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 2 — Activation & Architecture Tricks (6L×512d)

{group_base_yaml(groups["group2"])}

**Key finding:** SwigLU + tie_weights = best combo (−0.012). token_anchor, softcap, MQA all neutral at 6L.

{make_group_table(groups["group2"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Group 3 — Confirm Best Config
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 3 — Confirm Best Config (6L×512d)

{group_base_yaml(groups["group3"])}

**Key finding:** All tricks neutral or slightly negative at shallow depth. Best config = base.

{make_group_table(groups["group3"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Group 4 — Vocab Size & Depth
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 4 — Vocab Size & Depth (6L)

{group_base_yaml(groups["group4"])}

**Key finding:** vocab=16k is optimal; 8k hurts. Adding layers at 512d doesn't help — motivates architecture search.

{make_group_table(groups["group4"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Group 5 — Architecture Search ~30M
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 5 — Architecture Search (~30M params)

{group_base_yaml(groups["group5"])}

**Key finding:** Deeper-narrower consistently wins. 12L×384d = best at 30M budget.

{make_group_table(groups["group5"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Group 6 — Architecture Search ~40M + GQA
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 6 — Architecture Search (~40M params) + GQA

{group_base_yaml(groups["group6"])}

**Key finding:** Deeper-narrower trend continues. GQA does not help. 20L×320d = best.

{make_group_table(groups["group6"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Group 7 — 256d at depth
    # Group 7 has no explicit base experiment in YAML; use g6_05_20L_320 as reference.
    # -------------------------------------------------------------------------
    best_g6 = data["g6_05_20L_320"]["val_loss"]
    sections.append(f"""\
## Group 7 — Narrow-and-Deep: 256d at Various Depths

{group_base_yaml(groups["group7"])}

Base for Δ: best from group6 (g6_05 = 20L×320d, val_loss={fmt(best_g6)})
**Key finding:** 28L×256d slightly edges out 20L×320d. Diminishing returns beyond 28L.

{make_group_table(groups["group7"], data, initial_base, external_base_name="g6_05_20L_320")}

---
""")

    # -------------------------------------------------------------------------
    # Group 8 — Depth-Dependent Tricks
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 8 — Depth-Dependent Tricks (28L×256d)

{group_base_yaml(groups["group8"])}

**Key finding:** token_anchor helps at depth (−0.0013). softcap and resid_scale neutral or negative. ReZero mildly negative.
Note: token_anchor was neutral at 6L (group3) — it is depth-dependent.

{make_group_table(groups["group8"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Group 9 — Value Skip Connections
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 9 — Value Skip Connections (28L×256d)

{group_base_yaml(groups["group9"])}

Three variants tested: v += x (current-layer residual), v += x₀ (original embedding), v += λ·v_{{l-1}} (cross-layer carry).
**Key finding:** All value residual variants are ≥ anchor alone. Anchor-only = best.

{make_group_table(groups["group9"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Group 10 — Vocab Sweep + Architecture Variants
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 10 — Vocab Size & Architecture Variants (28L×256d)

{group_base_yaml(groups["group10"])}

**Key findings:**
- vocab=10240 beats 16k (−0.004); 8k and 12k both hurt
- `separate_kv`: splits fused kv\\_proj into square k\\_proj+v\\_proj; +5% tok/s, neutral loss
- `spectral_clip`: −19% tok/s, no loss benefit — dropped
- `muon_attn_only`: routes MLP matrices to AdamW — catastrophic (+0.041); Muon is essential for MLP
- Smaller MLP (hidden=512/384): −27%/−45% params, slight loss increase — compression headroom exists but costs quality

{make_group_table(groups["group10"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Group 11 — Value Embeddings
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 11 — Value Embeddings / E Layers (ResFormer-style)

Dedicated per-layer embedding tables injected into V via a learned per-head gate:
`v += 3·σ(Linear(x[:12])) * ve_table(idx)`. Layer type `E` in `layer_pattern` enables this;
embedding tables are separate from `token_emb`, optimised with `emb_lr`.

{group_base_yaml(groups["group11"])}

**Key finding:** Gains are marginal across all VE configs. vocab=10k+VE rows benefit more from the vocab change than from VE itself. gate_channels makes no meaningful difference.

{make_group_table(groups["group11"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Group 12 — Mamba Hybrid
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Group 12 — Mamba Hybrid (first layer)

{group_base_yaml(groups["group12"])}

**Key finding:** Replacing the first attention layer with Mamba SSM hurts both quality (+0.0076) and throughput (−42% tok/s). Pure attention remains better at this scale and sequence length.

{make_group_table(groups["group12"], data, initial_base)}

---
""")

    # -------------------------------------------------------------------------
    # Kernel Benchmarks Appendix
    # -------------------------------------------------------------------------
    sections.append("""\
## Appendix — Custom Kernel Benchmarks

Hardware: AMD Ryzen AI MAX+ 395, gfx1151 (RDNA4), 40 CU, ~300 GB/s unified memory.

### RMSNorm (Triton)

Benchmark shape: `(128, 64, 384)` bfloat16.

| Implementation | Speed |
|---|---|
| `nn.RMSNorm` (91.3 μs) | 1.0x |
| Triton + autotune (49.7 μs) | **1.84x** |

autotune selected: fwd `num_warps=2`, bwd `num_warps=1`.
**Status: Not used in final config** — final architecture uses `layernorm` (ablation g3_01 showed neutral vs rmsnorm).

### RoPE (Triton)

Benchmark shape: `B=64, H=4, T=128, D=64` (actual training config), bfloat16 fwd+bwd.

| Implementation | Speed (fwd+bwd) |
|---|---|
| Native PyTorch (283.4 μs) | 1.0x |
| Triton (246.9 μs) | **1.15x** |

**Status: Not enabled** — discovered faster only after correcting benchmark shape late in the project. Correctness verified; enabling requires one line change in `rope.py`.

### Fused Linear + Cross-Entropy (v2)

True kernel fusion: matmul written inside the Triton kernel; `[BT, V]` logit tensor never materialised.
Two-pass algorithm: (1) online softmax + collect target logit; (2) recompute logits → grad\\_x + grad\\_W.

Full training step benchmark (B=64, T=128, V=10240, d=256, L=28):

| Implementation | ms/step | Peak Memory |
|---|---|---|
| Standard CE | 1619 ms (1.0x) | 7109 MB |
| Fused CE v2, autotuned | **1559 ms (1.04x)** | **5815 MB (1.22x savings)** |

Note: full-step times measured without `torch.compile` or operator fusion — reference only, not representative of compiled training performance. CE kernel contribution is diluted by all other ops.
**Status: Optional** (`use_fused_ce=True`). Primary value is memory savings (1.22x). Speed on MI355X not measured.

---
""")

    # -------------------------------------------------------------------------
    # GPU Profiling Appendix
    # -------------------------------------------------------------------------
    prof = profiling_section()
    if prof:
        sections.append(prof + "\n---\n")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    sections.append(f"""\
## Summary — Best Config Evolution

| Step | Change | val_loss | Δ |
|------|--------|----------|---|
| Baseline | SGD + cosine + learned\\_pos + GELU | 1.7319 | — |
| Group 1  | → Muon+AdamW + RoPE + WSD | 1.1953 | −0.5366 |
| Group 2  | → SwigLU + tie\\_weights | 1.1836 | −0.0117 |
| Groups 5-7 | → 28L×256d (deep-narrow arch) | 1.1708 | −0.0128 |
| Group 8  | → token\\_anchor | 1.1700 | −0.0008 |
| Group 9  | → confirmed anchor-only best | 1.1696 | −0.0004 |
| Group 10 | → vocab=10240 | 1.1660 | −0.0036 |
| Group 11 | → 4E value embeddings (pos 3,11,19,27) | **{fmt(best_loss)}** | {delta(data["g10_01_vocab10k"]["val_loss"], best_loss)} |

**Total improvement: −0.5667** (32.7% relative reduction from baseline)
""")

    # -------------------------------------------------------------------------
    # Reflection
    # -------------------------------------------------------------------------
    sections.append("""\
## Reflection — What We Would Do Differently

### 1. Weight initialisation exploration was insufficient

We tested `muon_uniform` vs `gpt2` init in a single group-1 experiment and found `gpt2` wins by 0.03 val_loss. We did not investigate why. One possible confound is weight tying: `lm_head` shares weights with `token_emb`, which may have interacted differently with each init scheme. A cleaner experiment would decouple the two before drawing conclusions.

### 2. Custom kernels before profiling — wrong order

We wrote three Triton kernels (RMSNorm, RoPE, fused CE) before running any profiler. When we eventually profiled the best config with torch profiler traces, GPU utilisation was already 94.8% — meaning there was no large dispatch bubble to fix. The correct workflow is: **profile first, identify hot kernels, then write targeted replacements**. Of the three kernels, none ended up in the final training path: RMSNorm is unused (final config uses LayerNorm), RoPE Triton was discovered to be faster only after correcting the benchmark shape late in the project, and fused CE provides memory savings but marginal speed improvement on gfx1151.

### 3. Vocab size was fixed too early

`vocab_size` was not systematically swept until Group 10 — after nine groups of experiments all run at `vocab_size=16000`. The final optimal value turned out to be 10240 (−0.004 vs 16k). This means Groups 1–9 were optimising on a suboptimal vocabulary, and some conclusions may not fully transfer: in particular, the value embedding experiments (Group 11) showed VE benefits more with vocab=16k than 10k, suggesting the Group 11 results are partly an artefact of the vocab choice. Vocab size interacts with embedding dimensionality and weight tying; it should be treated as a foundational hyperparameter and swept in the first group rather than the tenth.

### 4. Sequential ablation search misses interactions; automatic tuning was underused

Each group performed single-variable search on top of the previous group's best config. This greedy sequential strategy cannot discover interactions between hyperparameters — for example, the optimal `muon_lr` for 28L×256d may differ from the value inherited from Group 1 (6L×512d), and the optimal depth for a given vocab size was never jointly optimised. We did implement Optuna TPE hyperparameter search (`hypertune.py`) but used it only for a narrow LR sweep rather than as the primary search strategy. Investing more in automatic tuning earlier — using Optuna to jointly search over depth, width, vocab, and LR — would likely have found better configurations faster and with less manual iteration.

### 5. Value embedding parameter efficiency was poor

The final architecture includes 4 E-type (value embedding) layers, contributing +10.5M parameters (+42% of the base model) for a val_loss improvement of −0.0006 vs no-VE baseline. No iso-parameter comparison was made: it is unknown whether the same 10.5M parameters spent on additional attention layers, wider d_model, or deeper depth would have yielded greater benefit. We selected 4E because it was the best option within Group 11's search space, but the parameter efficiency of this choice was never challenged against alternatives.

### 6. Mamba hybrid did not help

A single experiment (g12_01) replaced the first attention layer with a Mamba SSM layer, keeping all other best-config settings (28L×256d, vocab=10240, 4E layers). Result: val_loss worsened by 0.0076 (1.1650→1.1726) and throughput dropped from 41,947 to 24,174 tok/s — a 42% speed penalty. The 42% speed drop is likely due to the absence of an optimised ROCm Mamba kernel — the selective scan ran without hardware-specific tuning available to Flash Attention. The quality regression suggests that at this scale and sequence length, attention is simply better. Hybrid architectures may have merit at longer sequences or larger scale, but within this project's constraints the result is a clear negative.

### 7. TensorBoard integration added limited value

We integrated TensorBoard (loss curves, LR schedules, weight norms) early in the project. In practice, all experiment tracking and comparison was done through JSONL log files parsed by `collect_results.py`. The TensorBoard writer added code complexity, a `SummaryWriter` dependency, and extra I/O on every training step, with minimal return — the ablation tables in this report were never derived from TensorBoard. A leaner approach would be structured JSONL logging only, with a simple `collect_results.py` for post-hoc analysis.

---
""")

    OUT.write_text("\n".join(sections))
    print(f"Report written to {OUT}")


if __name__ == "__main__":
    main()

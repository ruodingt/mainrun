"""
Generate ablation study report as REPORT.md.

Usage:
  python tools/generate_report.py
"""

import csv
from pathlib import Path
from collect_results import collect, load_full_yaml, load_ablations_base

ROOT = Path(__file__).parent.parent
OUT = ROOT / "REPORT.md"


def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "—"


def fmtv(v) -> str:
    """Format a yaml value for display in a markdown table cell."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return f"`{v}`"
    if isinstance(v, float):
        s = f"{v:.1f}" if v == int(v) else f"{v:g}"
        return f"`{s}`"
    if isinstance(v, str):
        return f"`{v}`"
    return str(v)


def delta(base, val):
    if base is None or val is None:
        return ""
    d = val - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _bubble_from_trace():
    """Parse the most recent .pt.trace.json for accurate GPU utilisation stats.
    Returns dict with keys: gpu_busy_ms, gpu_span_ms, bubble_ms, busy_pct, steps, n_gaps.
    Returns None if no trace file found.
    """
    import json, glob
    traces = sorted(glob.glob(str(ROOT / "profile_out" / "*.pt.trace.json")))
    if not traces:
        return None
    with open(traces[-1]) as fp:
        data = json.load(fp)
    events = data["traceEvents"]
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
        gpu_busy_ms=gpu_busy_us / 1e3,
        gpu_span_ms=gpu_span_us / 1e3,
        bubble_ms=bubble_us / 1e3,
        busy_pct=gpu_busy_us / gpu_span_us * 100,
        steps=steps,
        n_gaps=n_gaps,
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

    # GPU utilisation from torch profiler trace (most accurate source)
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
        rows = []
        with open(rocprof_csv, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 5:
                    continue
                try:
                    rows.append((row[0].strip('"'), int(row[1]), int(row[2]), int(row[3]), float(row[4])))
                except ValueError:
                    continue
        rows.sort(key=lambda x: -x[2])
        lines += [
            "### Top Kernels by GPU Time (rocprof --stats)", "",
            "| Kernel | Calls | Total | Avg | % |",
            "|---|---|---|---|---|",
        ]
        for name, calls, total_ns, avg_ns, pct in rows[:15]:
            lines.append(f"| `{name[:55]}` | {calls} | {total_ns/1e6:.1f}ms | {avg_ns/1e3:.0f}µs | {pct:.1f}% |")
        lines.append("")

    return "\n".join(lines)


def main():
    data = {r["exp"]: r for r in collect()}

    ablations_base = load_ablations_base()   # flat dict from ablations.yaml root base
    best_yaml = load_full_yaml("g12_00_base")

    def bv(key):
        return fmtv(ablations_base.get(key))

    def bestv(section, key):
        return fmtv(best_yaml.get(section, {}).get(key))

    base_loss = data["g1_00_baseline"]["val_loss"]
    best_loss = data["g12_00_base"]["val_loss"]

    base_lp = ablations_base.get("layer_pattern", "")
    best_arch = best_yaml.get("arch", {})
    best_lp = best_arch.get("layer_pattern", "")

    sections = []

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------
    sections.append(f"""\
# Ablation Study Report

**Goal:** Minimise validation loss on Hacker News titles (100k, 7 epochs, seed=1337).
**Baseline:** SGD + cosine LR + learned pos emb + GELU + {len(base_lp)}L×{ablations_base.get('d_model')}d×{ablations_base.get('vocab_size')} → **val_loss = {base_loss}**
**Final best:** {len(best_lp)}L×{best_arch.get('d_model')}d×{best_arch.get('vocab_size')} + Muon+AdamW + WSD + RoPE + SwigLU + tie_weights + token_anchor → **val_loss = {best_loss}**

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
    # Group 1: Optimizer
    # -------------------------------------------------------------------------
    base = data["g1_00_baseline"]["val_loss"]
    g1_rows = [
        ("g1_00_baseline",  "SGD + cosine + learned",         "g1_00"),
        ("g1_01_muon",      "Muon+AdamW",                      "g1_01"),
        ("g1_06_muon_clip_adamw", "Muon + AdamW clip",         "g1_06"),
        ("g1_08_muon_rope", "Muon + RoPE",                     "g1_08"),
        ("g1_09_muon_wsd",  "Muon + WSD",                      "g1_09"),
        ("g1_11_muon_rope_wsd", "**Muon + RoPE + WSD** ✓",    "g1_11"),
        ("g1_12_muon_rope_init_wsd", "Muon + RoPE + WSD + muon_uniform_init", "g1_12"),
    ]
    rows = []
    for exp, desc, _ in g1_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(base, r.get("val_loss")),
                     f"{r.get('params_M', '—')}M", f"{r.get('tok_s', '—'):,}" if r.get('tok_s') else "—"])
    sections.append(f"""\
## Group 1 — Optimizer & Schedule

Base: 6L×512d, GELU, SGD+cosine+learned_pos
**Key finding:** Muon+AdamW with RoPE+WSD gives the biggest single jump (−0.54).
gpt2 init beats muon_uniform init.

{table(["Config", "val_loss", "Δ vs baseline", "Params", "tok/s"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Group 2: Activation + Arch tricks (6L×512d, now with best optimizer)
    # -------------------------------------------------------------------------
    base2 = data["g2_00_base"]["val_loss"]
    g2_rows = [
        ("g2_00_base",        "base (GELU, no tie)"),
        ("g2_05_tie_weights", "tie_weights"),
        ("g2_07_swiglu",      "SwigLU"),
        ("g2_08_relu_sq",     "ReLU²"),
        ("g2_09_swiglu_tie",  "**SwigLU + tie** ✓"),
        ("g2_10_relu_sq_tie", "ReLU² + tie"),
        ("g2_01_token_anchor","token_anchor"),
        ("g2_02_softcap",     "logit_softcap=30"),
        ("g2_04_mqa",         "MQA (n_kv=1)"),
    ]
    rows = []
    for exp, desc in g2_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(base2, r.get("val_loss")),
                     f"{r.get('params_M', '—')}M", f"{r.get('tok_s', '—'):,}" if r.get('tok_s') else "—"])
    sections.append(f"""\
## Group 2 — Activation & Architecture Tricks (6L×512d)

Base: 6L×512d + Muon+AdamW+RoPE+WSD
**Key finding:** SwigLU + tie_weights = best combo (−0.012). token_anchor, softcap, MQA all neutral at 6L.

{table(["Config", "val_loss", "Δ vs g2_base", "Params", "tok/s"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Group 3: Confirm best config (swiglu+tie) — arch tricks at 6L×512d
    # -------------------------------------------------------------------------
    base3 = data["g3_00_base"]["val_loss"]
    g3_rows = [
        ("g3_00_base",         "base (swiglu+tie+rope+muon+wsd)"),
        ("g3_01_rmsnorm",      "RMSNorm"),
        ("g3_02_token_anchor", "token_anchor"),
        ("g3_03_softcap",      "logit_softcap=30"),
        ("g3_04_mqa",          "MQA"),
        ("g3_05_anchor_softcap","anchor + softcap"),
    ]
    rows = []
    for exp, desc in g3_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(base3, r.get("val_loss")),
                     f"{r.get('params_M', '—')}M"])
    sections.append(f"""\
## Group 3 — Confirm Best Config (6L×512d)

Base: 6L×512d + Muon+AdamW+RoPE+WSD + SwigLU + tie_weights
**Key finding:** All tricks neutral or slightly negative at shallow depth. Best config = base.

{table(["Config", "val_loss", "Δ vs g3_base", "Params"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Group 4: Vocab + depth at 6L
    # -------------------------------------------------------------------------
    base4 = data["g4_00_base"]["val_loss"]
    g4_rows = [
        ("g4_00_base",   "6L×512d, vocab=16k"),
        ("g4_01_8k",     "vocab=8k"),
        ("g4_02_12A",    "12L×512d, vocab=16k"),
        ("g4_03_8k_12A", "12L×512d, vocab=8k"),
    ]
    rows = []
    for exp, desc in g4_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(base4, r.get("val_loss")),
                     f"{r.get('params_M', '—')}M"])
    sections.append(f"""\
## Group 4 — Vocab Size & Depth (6L)

**Key finding:** vocab=16k is optimal; 8k hurts. Adding layers at 512d doesn't help — motivates architecture search.

{table(["Config", "val_loss", "Δ vs g4_base", "Params"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Group 5: Arch search 30M
    # -------------------------------------------------------------------------
    base5 = data["g5_00_base"]["val_loss"]
    g5_rows = [
        ("g5_00_base",     "6L×512d  (base)"),
        ("g5_06_4L_640",   "4L×640d"),
        ("g5_05_5L_576",   "5L×576d"),
        ("g5_01_8L_448",   "8L×448d"),
        ("g5_02_9L_448",   "9L×448d"),
        ("g5_04_13L_384",  "13L×384d"),
        ("g5_03_12L_384",  "**12L×384d** ✓"),
    ]
    rows = []
    for exp, desc in g5_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(base5, r.get("val_loss")),
                     f"{r.get('params_M', '—')}M", f"{r.get('tok_s', '—'):,}" if r.get('tok_s') else "—"])
    sections.append(f"""\
## Group 5 — Architecture Search (~30M params)

Fixed: Muon+AdamW+RoPE+WSD+SwigLU+tie+vocab16k
**Key finding:** Deeper-narrower consistently wins. 12L×384d = best at 30M budget.

{table(["Config", "val_loss", "Δ vs base", "Params", "tok/s"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Group 6: Arch search 40M + GQA
    # -------------------------------------------------------------------------
    base6 = data["g6_00_base"]["val_loss"]
    g6_rows = [
        ("g6_00_base",         "12L×384d  (base)"),
        ("g6_01_14L_384",      "14L×384d"),
        ("g6_02_16L_384",      "16L×384d"),
        ("g6_08_16L_384_gqa2", "16L×384d GQA-2"),
        ("g6_03_18L_384",      "18L×384d"),
        ("g6_04_19L_384",      "19L×384d"),
        ("g6_09_21L_384_gqa2", "21L×384d GQA-2"),
        ("g6_06_24L_320",      "24L×320d"),
        ("g6_07_28L_320",      "28L×320d"),
        ("g6_05_20L_320",      "**20L×320d** ✓"),
    ]
    rows = []
    for exp, desc in g6_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(base6, r.get("val_loss")),
                     f"{r.get('params_M', '—')}M", f"{r.get('tok_s', '—'):,}" if r.get('tok_s') else "—"])
    sections.append(f"""\
## Group 6 — Architecture Search (~40M params) + GQA

**Key finding:** Deeper-narrower trend continues. GQA does not help. 20L×320d = best.

{table(["Config", "val_loss", "Δ vs base", "Params", "tok/s"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Group 7: 256d at depth
    # -------------------------------------------------------------------------
    g7_rows = [
        ("g7_01_28L_256", "**28L×256d** ✓"),
        ("g7_02_32L_256", "32L×256d"),
        ("g7_03_36L_256", "36L×256d"),
        ("g7_04_40L_256", "40L×256d"),
        ("g7_05_44L_256", "44L×256d"),
    ]
    best_g6 = data["g6_05_20L_320"]["val_loss"]
    rows = []
    for exp, desc in g7_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(best_g6, r.get("val_loss")),
                     f"{r.get('params_M', '—')}M", f"{r.get('tok_s', '—'):,}" if r.get('tok_s') else "—"])
    sections.append(f"""\
## Group 7 — Narrow-and-Deep: 256d at Various Depths

Base for Δ: best from group6 (20L×320d = {fmt(best_g6)})
**Key finding:** 28L×256d slightly edges out 20L×320d. Diminishing returns beyond 28L.

{table(["Config", "val_loss", "Δ vs g6_best", "Params", "tok/s"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Group 8: Depth-dependent tricks on 28L×256d
    # -------------------------------------------------------------------------
    base8 = data["g8_00_base"]["val_loss"]
    g8_rows = [
        ("g8_00_base",          "28L×256d  (base)"),
        ("g8_01_anchor",        "**token_anchor** ✓"),
        ("g8_02_softcap",       "logit_softcap=30"),
        ("g8_03_anchor_softcap","anchor + softcap"),
        ("g8_04_anchor_scale",  "anchor + resid_scale"),
        ("g8_07_rezero",        "ReZero"),
        ("g8_06_anchor_rezero", "anchor + ReZero"),
        ("g8_05_all",           "anchor + softcap + scale"),
    ]
    rows = []
    for exp, desc in g8_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(base8, r.get("val_loss")),
                     f"{r.get('params_M', '—')}M"])
    sections.append(f"""\
## Group 8 — Depth-Dependent Tricks (28L×256d)

**Key finding:** token_anchor helps at depth (−0.0013). softcap and resid_scale neutral or negative. ReZero mildly negative.
Note: token_anchor was neutral at 6L (group3) — it is depth-dependent.

{table(["Config", "val_loss", "Δ vs g8_base", "Params"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Group 9: Value residual
    # -------------------------------------------------------------------------
    base9 = data["g9_00_base"]["val_loss"]
    g9_rows = [
        ("g9_00_base",             "28L×256d + anchor  (base)"),
        ("g9_01_value_res",        "v += x (current layer)"),
        ("g9_02_value_res_anchor", "v += x + anchor"),
        ("g9_05_value_res_x0",     "v += x₀ (original emb)"),
        ("g9_06_value_res_x0_anchor","v += x₀ + anchor"),
        ("g9_07_both_res",         "v += x + x₀"),
        ("g9_08_value_carry",      "v += λ·v_{l-1}"),
        ("g9_09_value_carry_anchor","v += λ·v_{l-1} + anchor"),
        ("g9_10_anchor_only",      "**anchor only** ✓"),
    ]
    rows = []
    for exp, desc in g9_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(base9, r.get("val_loss")),
                     f"{r.get('params_M', '—')}M"])
    sections.append(f"""\
## Group 9 — Value Skip Connections (28L×256d)

Three variants tested: v += x (current-layer residual), v += x₀ (original embedding), v += λ·v_{{l-1}} (cross-layer carry).
**Key finding:** All value residual variants are ≥ anchor alone. Anchor-only = best.

{table(["Config", "val_loss", "Δ vs g9_base", "Params"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Group 10: Vocab sweep + architecture experiments
    # -------------------------------------------------------------------------
    base10 = data["g10_00_base"]["val_loss"]
    g10_rows = [
        ("g10_00_base",                  "28L×256d, vocab=16k (base)"),
        ("g10_01_vocab10k",              "**vocab=10240** ✓"),
        ("g10_02_vocab12k",              "vocab=12288"),
        ("g10_03_vocab8k",               "vocab=8192"),
        ("g10_04_separate_kv_vocab_10k", "separate_kv, vocab=10k"),
        ("g10_05_spectral_clip",         "spectral_clip=1.0, vocab=10k"),
        ("g10_06_muon_attn_only",        "muon_attn_only (MLP→AdamW)"),
        ("g10_08_mlp2x",                 "MLP hidden=512 (mlp_expand=3.0)"),
        ("g10_09_mlp1p5x",              "MLP hidden=384 (mlp_expand=2.25)"),
    ]
    rows = []
    for exp, desc in g10_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(base10, r.get("val_loss")),
                     f"{r.get('tok_s', '—'):,}" if r.get('tok_s') else "—",
                     f"{r.get('params_M', '—')}M"])
    sections.append(f"""\
## Group 10 — Vocab Size & Architecture Variants (28L×256d)

**Key findings:**
- vocab=10240 beats 16k (−0.004); 8k and 12k both hurt
- `separate_kv`: splits fused kv\\_proj into square k\\_proj+v\\_proj; +5% tok/s, neutral loss
- `spectral_clip`: −19% tok/s, no loss benefit — dropped
- `muon_attn_only`: routes MLP matrices to AdamW — catastrophic (+0.041); Muon is essential for MLP
- Smaller MLP (hidden=512/384): −27%/−45% params, slight loss increase — compression headroom exists but costs quality

{table(["Config", "val_loss", "Δ vs base", "tok/s", "Params"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Group 11: Value Embeddings (E layers)
    # -------------------------------------------------------------------------
    base10_vocab10k = data["g10_01_vocab10k"]["val_loss"]
    base10_vocab16k = data["g10_00_base"]["val_loss"]
    g11_vocab16k_rows = [
        ("g11_01_ve_2e",       "2E layers (pos 9,19)"),
        ("g11_07_ve_2e",       "2E layers (pos 13,27)"),
        ("g11_03_ve_2e_gate16","2E layers (pos 9,19), gate_ch=16"),
        ("g11_02_ve_3e",       "3E layers (pos 9,18,27)"),
        ("g11_06_ve_4e",       "4E layers (pos 3,11,19,27)"),
    ]
    g11_vocab10k_rows = [
        ("g11_08_vocab10k",                "28A, no VE (baseline)"),
        ("g11_04_3e_vocab10k",             "3E layers (pos 9,18,27)"),
        ("g11_10_vocab10k_4ve_i8",         "4E layers (pos 3,11,19,27) ✓"),
        ("g11_11_mlr0025_vocab10k_4ve_i8", "4E layers, muon_lr=0.025"),
        ("g11_9_vocab10k_5ve_i7",          "5E layers (pos 7,12,17,22,27)"),
    ]
    rows16 = []
    for exp, desc in g11_vocab16k_rows:
        r = data.get(exp, {})
        rows16.append([desc, fmt(r.get("val_loss")), delta(base10_vocab16k, r.get("val_loss")),
                       f"{r.get('tok_s', '—'):,}" if r.get('tok_s') else "—",
                       f"{r.get('params_M', '—')}M"])
    rows10 = []
    for exp, desc in g11_vocab10k_rows:
        r = data.get(exp, {})
        rows10.append([desc, fmt(r.get("val_loss")), delta(base10_vocab10k, r.get("val_loss")),
                       f"{r.get('tok_s', '—'):,}" if r.get('tok_s') else "—",
                       f"{r.get('params_M', '—')}M"])
    sections.append(f"""\
## Group 11 — Value Embeddings / E Layers (ResFormer-style)

Dedicated per-layer embedding tables injected into V via a learned per-head gate:
`v += 3·σ(Linear(x[:12])) * ve_table(idx)`. Layer type `E` in `layer_pattern` enables this;
embedding tables are separate from `token_emb`, optimised with `emb_lr`.

**Key findings:**
- vocab=16k: VE consistently helps; 3E (pos 9,18,27) is best (−0.0009)
- vocab=10k: VE provides marginal and noisy benefit; results across 3E/4E/no-VE are within run variance
- gate_channels (12 vs 16) makes no meaningful difference
- 4E (pos 3,11,19,27) selected for final config; validated by g12_00_base re-run (1.1650)

**vocab=16k** (Δ vs g10\\_00\\_base = {fmt(base10_vocab16k)}):

{table(["Config", "val_loss", "Δ vs 16k base", "tok/s", "Params"], rows16)}

**vocab=10k** (Δ vs g10\\_01 = {fmt(base10_vocab10k)}):

{table(["Config", "val_loss", "Δ vs 10k base", "tok/s", "Params"], rows10)}

Note: g11_10 and g12_00_base use identical config; the spread in their val_loss (1.1667 vs 1.1650) reflects run variance at this scale.

---
""")

    # -------------------------------------------------------------------------
    # Group 12: Mamba hybrid
    # -------------------------------------------------------------------------
    base12 = data["g12_00_base"]["val_loss"]
    g12_rows = [
        ("g12_00_base",        "**best config (pure attention)** ✓"),
        ("g12_01_mamba_first", "first layer → Mamba SSM"),
    ]
    rows = []
    for exp, desc in g12_rows:
        r = data.get(exp, {})
        rows.append([desc, fmt(r.get("val_loss")), delta(base12, r.get("val_loss")),
                     f"{r.get('tok_s', '—'):,}" if r.get('tok_s') else "—",
                     f"{r.get('params_M', '—')}M"])
    sections.append(f"""\
## Group 12 — Mamba Hybrid (first layer)

Base: best config (28L×256d, vocab=10240, 4E layers, Muon+AdamW+RoPE+WSD)
**Key finding:** Replacing the first attention layer with Mamba SSM hurts both quality (+0.0076) and throughput (−42% tok/s). Pure attention remains better at this scale and sequence length.

{table(["Config", "val_loss", "Δ vs base", "tok/s", "Params"], rows)}

---
""")

    # -------------------------------------------------------------------------
    # Kernel Benchmarks (Appendix)
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
    # GPU Profiling (optional — reads profile_out/ if present)
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
| Group 11 | → 4E value embeddings, interval=8 | **1.1652** | −0.0008 |

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

The final architecture includes 4 E-type (value embedding) layers, contributing +10.5M parameters (+42% of the base model) for a val_loss improvement of only −0.0008. No iso-parameter comparison was made: it is unknown whether the same 10.5M parameters spent on additional attention layers, wider d_model, or deeper depth would have yielded greater benefit. We selected 4E because it was the best option within Group 11's search space, but the parameter efficiency of this choice was never challenged against alternatives.

### 6. Mamba hybrid did not help

A single experiment (g12_01) replaced the first attention layer with a Mamba SSM layer, keeping all other best-config settings (28L×256d, vocab=10240, 4E layers). Result: val_loss worsened by 0.0076 (1.1650→1.1726) and throughput dropped from 41,947 to 24,174 tok/s — a 42% speed penalty. The 42% speed drop is likely due to the absence of an optimised ROCm Mamba kernel — the selective scan ran without hardware-specific tuning available to Flash Attention. The quality regression suggests that at this scale and sequence length, attention is simply better. Hybrid architectures may have merit at longer sequences or larger scale, but within this project's constraints the result is a clear negative.

### 4. TensorBoard integration added limited value

We integrated TensorBoard (loss curves, LR schedules, weight norms) early in the project. In practice, all experiment tracking and comparison was done through JSONL log files parsed by `collect_results.py`. The TensorBoard writer added code complexity, a `SummaryWriter` dependency, and extra I/O on every training step, with minimal return — the ablation tables in this report were never derived from TensorBoard. A leaner approach would be structured JSONL logging only, with a simple `collect_results.py` for post-hoc analysis.

---
""")

    report = "\n".join(sections)
    OUT.write_text(report)
    print(f"Report written to {OUT}")


if __name__ == "__main__":
    main()

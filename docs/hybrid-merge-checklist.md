# Hybrid / Mamba → train.py 接入 Checklist

把 `mamba.py`(`Mamba2Mixer` / `Mamba`)和 `hybrid.py`(`HybridLM` / `HybridConfig`,
`layer_pattern` 控每层类型)接进 `train.py` 训练循环时的实操清单。

核心心智模型:**`train.py` 的优化绝大多数活在与 token-mixer 正交的三个层面
(优化器 / 残差流 / 训练循环),天然通用。真正 token-mixer 专属的只有 RoPE(只 attn)
和 SSM init(只 mamba),已在代码里隔离好。** 所以接入工作主要是"分组路由"和"别盲信幻数"。

---

## 1. 必须改 —— 不改会训飞或报错

### 1.1 优化器分组(最大坑)

`train.py` 当前按 `p.ndim >= 2 → Muon`。当前 GPT 全是 2D 矩阵恰好没事,已加
**assert 守卫**:一旦出现 3D 参数(Mamba 的 depthwise conv)会立刻报错。

接入时按参数类型路由:

- [ ] **矩阵 (ndim==2) → Muon**:`in_proj.weight`、`out_proj.weight`、
      `q_proj`/`kv_proj`/`proj`、`mlp.net[0/2]`
- [ ] **`conv1d.weight` (3D, depthwise) → AdamW (`other_params`)**。
      ⚠️ 绝不能进 Muon —— 正交化深度可分离卷积核会直接训飞。
- [ ] **`A_log` / `D` / `dt_bias` (1D) → AdamW**。建议进 `scalar` 组(fast-moving),
      或退而求其次进 `other`。
- [ ] **ReZero scales(`mixer_scale` / `mlp_scale`)→ `scalar` 组**
- [ ] **`conv1d.bias`、`RMSNormGated.weight`、norm 权重 (1D) → `other`**

识别建议(按 name 匹配,放在分组循环里):

```python
# 伪代码:在 train.py 的 named_parameters() 分组循环内
if name.endswith("conv1d.weight"):          # 3D depthwise -> AdamW, 不进 Muon
    other_params.append(p)
elif name.endswith(("A_log", "D", "dt_bias")):  # SSM 标量 -> scalar 组
    scalar_params.append(p)
elif p.ndim == 2:
    muon_groups.setdefault(tuple(p.shape), []).append(p)
elif id(p) in scalar_ids:                    # ReZero scales
    scalar_params.append(p)
else:
    other_params.append(p)
```

- [ ] `scalar_ids` 收集处加上 `HybridBlock` 的 `mixer_scale` / `mlp_scale`
      (注意 GPT 用的是 `attn_scale` / `mlp_scale`,命名不同)。
- [ ] `skip_ids` 仍是 `token_emb` + `head`(tie 时同一张量),不变。

### 1.2 模型实例化

- [ ] `main()` 里 `GPTConfig(...)` → `HybridConfig(...)`,`model = HybridLM(cfg)`。
- [ ] 加一个 `--arch` 开关或 hyperparam,在 GPT / HybridLM 间切换,保留对照能力。
- [ ] `n_layer` 由 `len(layer_pattern)` 决定 —— Hyperparameters 里若仍传 `n_layer`,
      要么删掉、要么 assert `n_layer == len(layer_pattern)`。

---

## 2. init 分工(别一刀切)

- [ ] **投影矩阵**:identity-start(零出口) / muon_uniform 思路 —— `HybridLM._init_weights`
      已做(`use_rezero` 决定出口归零还是正常 init)。
- [ ] **SSM 参数 `A_log` / `dt_bias` / `D`**:Mamba canonical init,已收进
      `Mamba2Mixer._reset_ssm_parameters()`。**不要用 muon_uniform 之类盖掉它。**

---

## 3. 直接白嫖 —— 无需改动

这些与 token-mixer 类型无关,接进来自动生效:

- [x] ReZero(已在 `HybridBlock` 接好)
- [x] RMSNorm / weight tying / logit softcap / norm_emb(`HybridConfig` 已支持)
- [ ] WSD / cosine LR schedule、bf16 autocast、`torch.compile`、fp32 logits CE、
      grad clip 1.0 —— 全在训练循环层面,零改动。

> `torch.compile` 对 SSD 的 `cumsum` / `einsum` 友好,但**首次接入先关 compile** 验证
> loss 正常下降,再开 compile 提速(避免把 compile 的问题和架构问题混在一起 debug)。

---

## 4. 要重新扫的幻数(在 GPT 上 tune 出来的,别当常数)

⚠️ 以下数值是针对 **attention 网络的深度动力学**调出来的,Mamba 信息流方式不同,
直接搬大概率非最优甚至有害:

- [ ] `x0_lambdas` 的 `0.20→0.05`(token_anchor 注入强度)
- [ ] `resid_lambdas` 的 `1.15→1.05`(残差流缩放)
- [ ] 分组 LR(`muon_lr` / `scalar_lr` / `emb_lr` / `adamw_lr`)—— Mamba 矩阵 shape 和
      量级不同,最优 LR 可能偏移
- [ ] **ReZero × Mamba gating 重叠**:Mamba 自带 `dt`(选择性遗忘)+ `z` gating,
      已有"控制每步注入"的机制,再叠 ReZero 可能冗余甚至打架 →
      做消融(M 层开/关 ReZero 对比)。

---

## 5. 实验设计提醒

- [ ] **参数量对齐**:不同 `layer_pattern` 参数量不等(纯 attn ≈21M,纯 mamba ≈29M,
      因为 Mamba `in_proj` 很胖)。**公平对照前先用 `expand`/`d_state`/层数把各 pattern
      拉到同一 param budget**,否则低 loss 可能只是参数多带来的噪声。
- [ ] **建议接入顺序**:
      1. 先纯 Mamba(`"MMMM..."`)跑通,对比现有 GPT 的 val 1.21,看 gap
      2. 优化器分组改好,跑一遍确认 assert 不炸
      3. 关 compile 验证 loss 正常下降
      4. 开 compile 提速
      5. 扫 pattern(`AAAA` / `MMMM` / `MAMA` / `MMMAMMMAMMMA` …)
- [ ] **可证伪的小假设**:token_anchor(x0 注入)可能对 Mamba 比对 attention 更有用
      —— Mamba 固定 state 会遗忘早期 token 身份,持续注入原始 embedding 等于"提醒它是谁"。
      Attention 本就能回看,边际收益小。值得专门测。

---

## 6. 已知陷阱

- [ ] Muon 按 `tuple(p.shape)` 分组 stack。Mamba 矩阵 shape 和 attention 的不同 →
      Muon 组变多、stack 批量变小,效率略降(不影响正确性)。
- [ ] `block_size % chunk_len == 0` 必须成立(`HybridLM` 已 assert)。L=64 下
      `chunk_len=64` 走单 chunk(纯二次 SSD),也可设 32 走多 chunk —— 数值已对拍一致。
- [ ] 只有 `'A'` 层消费 RoPE;`HybridBlock.forward` 已按 `layer_type` 分流,Mamba 层
      不传 `cos_sin`。

---

## 7. Mamba/Hybrid 的专属 Kernel 机会 (未来展望)

虽然当前接入 `train.py` 可以直接跑通 PyTorch Native 版本，但 Mamba 架构的性能极度依赖底层的 Kernel 优化。以下是未来值得探索的 Custom Kernel 方向：

- [ ] **SSD (Structured State Space Duality) 核心 Kernel**
      Mamba-2 的灵魂在于 Chunked Matmul + Prefix Scan。原生 PyTorch 实现（`cumsum` + `einsum`）会产生大量中间张量，显存带宽占用极高。编写一个 Triton SSD Kernel，把块对角矩阵乘法和状态扫描全部在 SRAM (LDS) 内完成，是让 Mamba-2 速度起飞的关键（可参考 Tri Dao 的 `mamba-ssm` 库，但在 AMD 显卡上需要自己做 Triton 移植）。
- [ ] **Fused Causal Conv1d + SiLU/Swish**
      Mamba 中包含一维因果深度可分离卷积（Depthwise Conv1d），这是一种典型的 Memory-bound 操作。如果能用 Triton 把 Conv1d 和紧随其后的激活函数（SiLU/Swish）融合，避免中间结果写回显存，将极大节省访存带宽。
- [ ] **Fused `in_proj` 拆分与门控 (Gating) 优化**
      Mamba 的 `in_proj` 是一个超宽的矩阵乘法，出来后还要做 Chunk 划分、软激活和 Gating 乘法。把这些零碎的 Split + Multiply 操作融合起来，可以有效减少 Kernel Launch Overhead 并提升内存局部性。

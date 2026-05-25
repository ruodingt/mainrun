# Systemic & Architectural Bugs

这份文档记录了项目中一些深层次的系统性设计缺陷和数学逻辑漏洞。这些问题在小规模测试时可能只会表现为“Loss 降不到最优”或“偶发的 Spike”，但在扩展（Scale-up）或长时训练时，大概率会导致模型发散或性能严重下降。

---

## 1. 致命级 (Fatal)

### 1.1 Validation Loss 计算作弊 (Metric Deflation)
**定位：** `train.py` (`evaluate()` 函数)
**症状：** 跑出的 Validation Loss 看起来异常的低，直接破坏了 Assessment 的基准目标。
**分析：**
在 `evaluate()` 函数中，你们的返回逻辑是：`return losses / len(val_text)`。
这里的 `val_text` 是拼接好的**原始字符串**，而 `losses` 则是所有 Token 的 Cross Entropy 累加和。由于 BPE 分词器的存在，字符串的字符数（Character Count）通常是 Token 数的 3~4 倍！
把 Token 级别的 Loss 总和除以字符数，会导致汇报的 Validation Loss 被人工压缩了 3 到 4 倍。如果你们的 Assessment 目标是 1.754，用这个除法算出来的 1.5，其实真实的 Token 级 Loss 高达 5.0 以上。这是一个会直接导致面试“作弊”嫌疑的致命逻辑错误。
**修复建议：** 
将分母改为实际参与 evaluate 的 Token 数量：`num_eval_tokens = B * T * num_batches`，或者在累加时直接计算。

### 1.2 Weight Tying 导致 LM Head 学习率爆炸
**定位：** `train.py` (Optimizer 分组逻辑)
**症状：** 如果开启 `tie_weights = True`，模型在某些初始化下极易发散。
**分析：**
在优化器分组时，`token_emb` 被独立分入 `emb_group`，并分配了高达 `3e-3` 的 `emb_lr`（理由是“稀疏更新需要更高的 LR”）。然而，当 `tie_weights=True` 时，`head.weight` 与 `token_emb.weight` 物理上共享同一块显存。
语言模型头的 Cross Entropy 会产生覆盖整个 Vocab 的**稠密梯度 (Dense Gradients)**。这就意味着，LM Head 的稠密梯度会被施加 `3e-3` 的巨大更新步长（这是标准 `adamw_lr` `3e-4` 的 10 倍！）。
这完全违背了 Weight Tying 的优化器设计直觉，极易导致特征崩塌和 Loss 飞坡。
**修复建议：** 
当 `tie_weights = True` 时，必须强制将共享的 weight 分入常规 `adamw_lr` 组，或者拆分两者的学习率逻辑。

### 1.3 残差流指数级爆炸 (Architectural Flaw)
**定位：** `train.py` / `hybrid.py` (`use_resid_scale` 逻辑)
**症状：** 模型深度增加时，中间层激活值出现 NaN，或深层梯度极小。
**分析：**
在层间有这样一段逻辑：
`x = r * x + self.x0_lambdas[i] * x0`
其中 `r = resid_lambdas[i]` 被初始化为 `1.15 -> 1.05`。
这是一个直接乘在**主残差流 (Trunk)** 上的标量因子。由于 `r > 1.0`，每一层的残差流都会被放大！假设一个 12 层的模型，平均 $r \approx 1.1$，那么最后一层的方差会被放大 $1.1^{12} \approx 3.14$ 倍。如果是更深的网络（如 48 层），会被放大 $1.1^{48} \approx 97$ 倍。
Pre-LN 架构的基石在于主残差流仅通过加法累积，保证方差是 $O(L)$ 的线性增长。强行在主干上连乘 $>1$ 的标量，会导致深层的激活值迅速逼近 `bfloat16` 的动态范围极限，并完全破坏初始化方差的精妙平衡。
**修复建议：** 
任何放大因子都应该作用在**分支 (Branch)** 上，即：`x = x + r * block(x)`，或者确保 `r` 被限制在极小的衰减范围，绝对不能在 Trunk 上进行无归一化的 $>1$ 连乘。

---

## 2. 严重级 (Severe)

### 2.1 Muon 优化器无视 Gradient Clipping
**定位：** `train.py` (`clip_grad_norm_`) & `optim.py` (`muon_step_fused`)
**症状：** 梯度裁剪无法抑制 Muon 组的剧烈更新，反而误伤 AdamW 组。
**分析：**
在 `loss.backward()` 后调用了全局的 `clip_grad_norm_(model.parameters(), 1.0)`。
当 Muon 组的一个矩阵梯度异常大时，全局 Norm 被拉高，所有参数的梯度都会被按比例缩小。
**然而，Muon 具有尺度不变性。** 它的第一步处理就是 `X = X / (X.norm() + 1e-6)`。这意味着不论外部的 `clip_grad_norm_` 把梯度砍成多小，Muon 都会强制将其范数重置为 1！
**结果是灾难性的：** Muon 组完全免疫了你的 Clipping，而无辜的 AdamW 组（Embedding, Norm, Biases）的梯度却实打实地被大幅度惩罚了。
**修复建议：** 
针对 Muon 组取消全局 Gradient Clipping（它们自己会做 Normalization），仅对 AdamW 组单独应用局部 Clipping。

### 2.2 优化器中灾难性的动态内存分配
**定位：** `optim.py` (`_step_muon`)
**症状：** 随着模型参数规模上升，每步训练耗时增加，显存碎片化严重。
**分析：**
在包装的 Python 层代码中：
```python
stacked_grads = torch.stack([p.grad for p in params])
stacked_params = torch.stack(params)
```
这段代码在每个 Training Step 都会触发！它在显存中动态申请了两块巨大的连续内存来拼装几十个矩阵。这彻底摧毁了 Custom Optimizer 本该具备的 “Zero-Allocation” 优势。频繁触发 CUDA Caching Allocator 会导致极大的 Host 端 CPU 开销和显存碎片化。
**修复建议：** 
在 `self.state` 初始化时，预先分配好连续的 Tensor Buffer，每次 step 时利用 `out=` 参数进行 in-place 操作，或者直接维护一份扁平化（Flatten）的参数视图。

---

## 3. 警告级 (Warning)

### 3.1 极限矩形矩阵对 Polar Express 并不友好
**定位：** `train.py` (MQA 下的 `kv_proj`)
**症状：** `kv_proj` 在 MQA 时训练效率低下，可能引入噪声。
**分析：**
你的路由策略非常武断：只要 `p.ndim >= 2` 就无脑扔给 Muon。
当开启 MQA (`n_kv_heads = 1`) 时，`kv_proj` 的形状变得极端不对称，例如 `(384, 128)`。
Muon 内部的核心是 Newton-Schulz 迭代（或者 Polar Express），它的数学本质是寻找最近的正交矩阵。对于极度“窄长”的矩阵，强行进行正交化的数学意义和计算效率都极差（大量的零空间会被浪费），不如直接退回使用 AdamW。
**修复建议：** 
不要盲目依赖 `ndim >= 2`，应该加入长宽比判断：如果 `max(M, N) / min(M, N) > threshold`，则将其归入 AdamW。

### 3.2 实验指纹 (Fingerprint) 的基础设施泄漏
**定位：** `train.py` (`get_fingerprint` 函数)
**症状：** 切换 Kernel 开关导致生成全新的实验目录，破坏了对照组比较。
**分析：**
在生成 `fingerprint` 用作 `run_dir` 的哈希特征时，你们排除了 `_INFRA` 和 `_FIXED` 集合。但是 `use_fused_ce` 没有被包含在 `_INFRA` 中！
`use_fused_ce` 仅仅是一个底层 Kernel 实现的切换，它应当做到计算等价（数学结果一致）。如果只为了测试内存占用而将其从 `False` 改为 `True`，系统会认为你改变了模型超参，从而建一个新的实验文件夹，导致你很难在同一个 Dashboard 下对比两个 Kernel 的性能。
**修复建议：** 
将 `use_fused_ce`（以及未来任何类似 `use_triton_rope` 的布尔值）加入到 `_INFRA` 忽略名单中。

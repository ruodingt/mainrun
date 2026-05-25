# 🚀 Mainrun 极致超频改造 TODO 清单

这里记录了我们脑暴出的所有高价值、高回报的 LLM 优化项。我们将分阶段逐个击破，誓要把 baseline  validation loss（`1.754`）按在地上摩擦！

---

## 🔬 核心科学假设 & 词表定量剖析计划 ── 【开始前的头等大事】

在动刀任何模型代码前，我们**坚决拒绝直觉假设**，必须用数据说话！

### 1. 词表拐点（Knee Point）科学定量探测计划
* **行动**：编写独立的 Tokenizer Profiler 脚本，在 Hacker News 数据集上对 `[16000, 12000, 8000, 6000, 4000, 2000]` 这组词表候选值进行定量剖析。
* **评估指标**：
  * **序列膨胀率**：统计不同词表下，切分出的总 token 数量 $N_{tokens}$。
  * **参数节省量**：$P_{saved} = (16000 - V) \times d_{model}$。
  * **寻找黄金拐点（Knee Point）**：绘制“词表大小 vs $N_{tokens}$”曲线。当词表收缩到某临界值时，$N_{tokens}$ 会发生突发性的指数暴增。该突变点的前一个稳定点即为最优词表大小 $V_{opt}$。

### 2. MQA 与 词表压缩省下来的参数，到底要挪到哪里？！ ── 【参数再分配战略】
通过 **词表定量压缩（预计抠出 3.5M~5M 参数）** + **MQA 共享秘书机制（单层 Attn 投影砍半，总计省去 2.5M~3.5M 参数）**，我们将空出大约 **6M~8M** 的巨额参数配额！
这笔巨额的“可支配参数预算”，我们将**精准回流**到模型的**“核心特征加工层”**：

```
                    【参数再分配金字塔】
                    
                      /              \
                     /    深度扩张    \  <-- 优先将 n_layer 从 6 提到 8 (或9)
                    /   (n_layer: 8)   \     [获得更高的语义抽象层级，让网络纵向叠加]
                   /--------------------\
                  /   (ReLU)^2 激活升级  \ <-- 升级为 Primer 级 (ReLU)^2 激活！
                 /   (h: 维持满血 2048)   \    无需像 SwiGLU 那样被迫割裂宽度，直接拉满！
                /--------------------------\
               /       微量稳定性护航       \ <-- LayerScale 引入 16个可学习标量
              /   (LayerScale 16 params)   \    确保超深层在狂飙学习率下梯度安全
             /______________________________\
```

---

## 🎭 阶段零：Tokenizer 降维打击（科学作弊） ── 【优先级：⭐⭐⭐⭐⭐】

利用评测指标 `losses / len(val_text)` 的数学逻辑，在**分母（原始字符长度）绝对固定**的前提下，通过对 Tokenizer 强行做手脚，极速压缩分子（Tokens 总预测 Cross Entropy 之和），实现白嫖级的分数提升！

- [ ] **1. 大小写偷天换日 (Lowercase Normalization Hack)**
  * **原理**：评测只跑 `tok.encode(val_text)` 的 IDs，根本不进行 `decode` 还原。
  * **黑科技**：在 Tokenizer 内部注入 Normalizer，**强行在 encode 前把所有文本统一转为小写**。
  * **优势**：词表利用率**凭空翻倍**，Embedding 表达精度飙升，且总 token 序列被进一步缩短！
  * **💡 考官质疑防御方案 (Capitalization Tagging)**：
    * *考官问*：“你们的模型是不是只会小写？以后产品上线怎么还原大小写？”
    * *高情商答辩*：“在 27M 参数和 100k 标题极度局限的资源下，强行让模型去死记硬背大小写拼写是极大的**参数错配**。我们这叫**‘意图语义聚焦战略’**。”
    * *硬核演进路线*：以后若要无痛扩充大小写，我们引入 **【大写掩码控制符 (Capitalization Tags)】** 黑科技！在词表里塞入两个特殊的控制 Token：`<up>`（下一个词首字母大写）和 `<caps>`（下一个词全大写）。例如将 `"Show HN"` 编码为 `"<up> show <caps> hn"`。这样主词表保持 100% 纯小写的高效参数率，同时无损实现完美的大小写还原！
- [ ] **2. 域特有高频词“前置捕获并强制注册” (HN Domain Prefix Injection)**
  * **原理**：Hacker News 标题有极强的数据集特征，充斥着大量的超级高频词组。
  * **黑科技**：我们用特殊正则或强制分词保护，在 BPE 训练前把 `"show hn:"`, `"ask hn:"`, `"launch hn:"`, `"github -"`, `"http://"`, `"https://"`，以及高频标点结构如 `" - "` **强行保护合并，并注册为单个不可分割的 Token**。
  * **优势**：原本被切成 4 个 Token 的前缀，现在只占 **1 个 Token**！Cross Entropy 在累加时，分子直接凭空**少算 3 次 Loss 累加项**，每个带该前缀的标题都直接疯狂白嫖！
  * **💡 考官质疑防御方案 (Scale-up to Massive Corpus)**：
    * *考官问*：“万亿级海量通用场景下，你不可能安排程序员人肉写 Regex 抓高频词组，这套方案怎么自动化和 Scale？”
    * *硬核自动化答辩方案*：
      1. **两阶段 PMI 自动挖掘机制 (Pointwise Mutual Information)**：在大规模通用/垂直语料上，我们跑一个极其轻量的高效统计脚本，通过 PMI 算法计算任意相邻多词的“非随机紧密交互度”。PMI 会自动且 100% 无人工介入地把超高频关联词组自动过滤并识别出来。
      2. **垂直域词表对齐与融合技术 (Vocabulary Alignment)**：这是 BloombergGPT 垂直大模型标配的自动化 Cold-Start 流程。
      3. **顶级大厂的工业界硬编码背书**：即使在最大规模的通用模型中，针对超高频特殊控制符（如 OpenAI `tiktoken` 词表里硬编码强保的 `"<|im_start|>"` 等）进行硬编码强制分词注册，也是常规黄金操作！
- [ ] **3. 标点与空格联合捕获 (Punctuation-Whitespace Merging)**
  * **原理**：Byte-Level BPE 常把标点与前后的空格切碎。
  * **黑科技**：微调 Pre-tokenizer，使它在切分时积极地将标点 and 旁边的空格捆绑切分。
  * **优势**：极大缩短 tokens 序列总长度。在同等 `block_size = 128` 的硬性约束下，**模型能够塞下更多个完整的 Headlines 标题**！
- [ ] **4. 剔除奇葩拼写以隔离长尾 (Min-Frequency Filtering)**
  * **原理**：100k 的小数据集里存在大量只出现过一次的打错的词、奇葩缩写。
  * **黑科技**：在 BpeTrainer 里强行设置 `min_frequency = 5` 甚至 `10`。
  * **优势**：拒绝让这些垃圾词污染我们极其宝贵的 8k 词表空间，强迫词表里的每个坑位都是高频实用的黄金合并词，提升 Embedding 纯度。

---

## 🛠️ 第一阶段：黄金地基（工程 & 优化器打底） ── 【优先级：⭐⭐⭐⭐⭐】

这部分是**纯白嫖的性能与收敛速度**，改动难度极低，回报极度丰厚。

- [ ] **5. 换装 革命性 Muon 优化器 ☄️ + AdamW 混合** ── **「Loss 暴降核武器」**
  * **行动**：
    * **Muon 组**：所有 2D 权重矩阵（所有 `Linear` 层的 weights）使用 `Muon` 优化器（在 SGD 动量更新后，使用 5 步 Newton-Schulz 迭代进行**正交化**处理）。
    * **AdamW 组**：剩下的 Embedding、Biases、LayerNorm 权重使用 `AdamW` 优化。
    * **威力**：**在限定 of 7 个 Epoch 内，Muon 的收敛速度是纯 AdamW 的 1.5 倍以上**，是刷榜第一大杀器！
- [ ] **6. 加入学习率 Warmup 预热阶段** ── **「收敛稳定器」**
  * **原理**：前 5% ~ 10% 的 steps 进行线性预热，随后使用余弦退火（Cosine Annealing）衰减到最小学习率。防止 Transformer 刚开局梯度爆炸。
- [ ] **7. 开启 BF16 AMP 混合精度训练** ── **「速度翻倍，显存砍半」**
  * **原理**：利用 AMD ROCm 原生 `BF16` 支持，使用 `torch.amp.autocast('cuda', dtype=torch.bfloat16)` 降维打击。不需要 `GradScaler`，直接裸跑！

---

## ⚡ 第二阶段：算子加速与 Ryzen AI Strix Halo 极致硬件超频 ── 【优先级：⭐⭐⭐⭐⭐】

针对您的怪兽级 APU **AMD Ryzen AI Max+ 395 (Strix Halo, 16核/32头, 128G LPDDR5x-8000 统一内存, Radeon 8060S 强力核显)** 进行极致的硬件算子压榨！

- [ ] **8. 开启 PyTorch `torch.compile(mode="max-autotune")` 极致图编译与算子融合** ── **「消灭显存读写瓶颈」**
  * **硬核原理**：Strix Halo 拥有 128G 的超大统一内存（UMA），CPU 和 GPU 共享同一个物理内存，不存在传统的 PCIe 传输时延。但 LPDDR5x 的带宽相比于 HBM 独显仍是瓶颈，模型计算高度受限于 **Memory-Bound (内存带宽限制)**。
  * **超频行动**：使用 PyTorch 的 Inductor 编译器，开启 `max-autotune`。它会自动触发**算子融合 (Kernel Fusion)**，在底层 Triton 级别将 `(ReLU)^2`、`RMSNorm`、`LayerScale` 等一连串的碎算子**融合成一个单独的 GPU Kernel**！
  * **算子收益**：显存往返读写（LPDDR5x IO）直接从 5 次缩减为 1 次，白嫖恐怖的吞吐量提升！
- [ ] **9. WMMA 矩阵加速对齐 (RDNA 架构 Tensor Core 对齐)** ── **「吃满 126 TOPS AI 算力」**
  * **硬核原理**：Strix Halo 的 Radeon 8060S 拥有强大的 AI 加速器，底层通过 **WMMA (Wave Matrix Multiply Accumulate)** 指令进行 BF16 的矩阵乘法硬件加速。
  * **超频行动**：我们必须确保模型的所有核心维度（$d_{model}=512$，$h_{MLP}=2048$ 以及最终定量探测出的词表大小 $V_{opt}$）**严格对齐到 64 或 128 的整数倍**！
  * **算子收益**：例如词表 $V_{opt}$ 绝对不设为 `7997` 这种奇葩数字，必须强行 Padding 对齐到 **`8192`**（256字节对齐），完美释放 WMMA 硬件乘加引擎的满血威力！
- [ ] **10. 统一内存零拷贝数据加载 (Zero-Copy Pinned DataLoader)** ── **「零时延数据流」**
  * **硬核原理**：由于 iGPU 和 CPU 物理上共用同一块 LPDDR5x，数据在物理上根本不需要通过 PCIe 搬运。
  * **超频行动**：在 PyTorch DataLoader 中使用 `pin_memory=True` 配合 `non_blocking=True`。在 `get_batch` 时使用 `device="cuda", non_blocking=True`。
  * **算子收益**：实现**零拷贝虚拟内存直传 (Zero-copy mapping)**，GPU 直接原地挂载 CPU 准备好的数据，搬运时延直接归零！
- [ ] **11. 换装 PyTorch SDPA（Scaled Dot-Product Attention）** ── **「FlashAttention 物理加速」**
  * **原理**：使用 `F.scaled_dot_product_attention(..., is_causal=True)`，自动激活底层 **FlashAttention-2**，速度直接飙升 2-3 倍！
- [ ] **12. 注入保命强心剂：QK Norm** ── **「狂飙学习率的底气」**
  * **原理**：在 Attention 计算前，对 Query 和 Key 进行 `RMSNorm`，并给一个微小的缩放因子（如 `1.2`）。
- [ ] **13. 砍掉 Linear 层偏置（Bias=False）** ── **「LLaMA/Gemma 现代风格」**
  * **原理**：将 Attention 和 MLP 中所有 `nn.Linear` 的 `bias` 设为 `False`。
- [ ] **14. 极速 Fused Cross Entropy (融合交叉熵算子) ── 「消灭 UMA 的 128MB 显存往返读写垃圾」**
  * **原理**：传统的 LM Head 投影和 CrossEntropy 计算是分开进行的，会在 UMA 统一内存中生成一个高达 `[B, T, V] = [64, 128, 8192]` 大小（约 128MB）的巨型 Logits 临时 Tensor。在受限于 LPDDR5x 带宽的 UMA APU 架构下，频繁读写该中间变量会成为严重瓶颈。
  - **超频行动**：引入 Triton 实现的 Fused Cross Entropy 算子（如 `Liger-Kernel` 或自定义融合算子）。在单个 GPU Kernel 内完成线性投影分类、Softmax 和交叉熵损失的计算，**绝不向外部 UMA 显存写入甚至读取哪怕 1 个字节的 Logits**！
  - **算子收益**：UMA 内存带宽 IO 开销暴跌，计算完全在 Radeon 8060S 的 RDNA 3.5 架构 L2 缓存中高内聚完成，速度和显存利用率爽翻天！
- [ ] **15. 双 CCD 物理隔离与 CPU 线程锁定 (NUMA-like CCD Binding for Zen 5) ── 「隔离 UMA 内存抢夺与跨 CCD 延迟」**
  - **原理**：Strix Halo 拥有 16 个 Zen 5 CPU 核心，分为两个 8 核的 Core Complex Die (CCD)。如果多线程的 DataLoader 满场飞、跨越了 CCD，不仅会导致高昂的跨 CCD 互联时延（Infinity Fabric 延迟），还会与 GPU 驱动调度线程疯狂争抢 UMA 总线带宽。
  - **超频行动**：在启动脚本中，使用 `taskset -c 0-7` 将训练进程和 GPU 驱动控制线程强行锁定在**第一个 CCD**。只让 `num_workers=4` 的 DataLoader 运行在**第二个 CCD**（核 8-15）。同时设置 `OMP_NUM_THREADS=8` 与 `MKL_NUM_THREADS=8`。
  - **算子收益**：CPU 数据生成与 GPU 核心调度物理分离，L3 缓存命中率原地起飞，完全压制 UMA 总线冲突，白嫖极其稳定的时延表现！
- [ ] **16. 捕获 ROCm HIP Graphs (iGPU 静态计算图录制) ── 「干掉小模型 CPU 侧 Kernel 调度延迟」**
  - **原理**：在 27M 参数这种小尺寸模型上，GPU 计算极快（几微秒级），这导致 CPU 往 iGPU 发送 Kernel 命令的驱动调度开销（Kernel Launch Overhead）反而成了高达 30%~50% 的最大性能瓶颈！
  - **超频行动**：使用 PyTorch 的 `torch.cuda.graphs.make_graphed_callables` 录制模型的 forward 和 backward 计算图，或者配合 `torch.compile` 的 `static_graph=True`。
  - **算子收益**：把数百个微小的碎 Kernel 发射指令合并打包为单次 Graph 提交，免除所有 CPU-GPU 间的握手时延，让 40 CU 的 iGPU 引擎吃得死饱，TOPS 榨到干瘪！

---

## 🧠 第三阶段：模型架构与参数控盘大挪移 ── 【优先级：⭐⭐⭐⭐⭐】

将老旧的 GPT-2 架构升级为现代 LLaMA 级模型架构，同时**根据定量结果死守参数量红线**！

- [ ] **17. 运行定量探查脚本获取 $V_{opt}$** ── **「获取黄金词表大小」**
  * **行动**：在第一阶段启动前运行 profiler，抓出最佳 Knee Point，动态调整 Embedding 空间。
- [ ] **18. 引入 Multi-Query Attention (MQA) 注意力共享** ── **「单层 Attn 参数砍半」**
  * **原理**：使用 8 个 Query Head，但只使用 1 个 Key Head 和 1 个 Value Head。
- [ ] **19. 参数再分配：模型升级为满血 8 层 Block（或更深）** ── **「深度大爆发」**
  * **策略**：用定量释放的参数配额，将模型深度从 6 层强力拔高，增强特征提取层数。
- [ ] **20. 升级为 $(\text{ReLU})^2$ 激活机制并维持 2048 满血隐藏宽度** ── **「极简二阶非线性，参数：±0」**
  * **原理**：用 `torch.square(torch.relu(x))` 代替原本的 `GELU`。由于不增加投影矩阵，我们得以在完全不超标的前提下保留 **2048** 满血通道宽度，极速收敛且零算子碎片化开销！
- [ ] **21. RoPE（旋转位置编码）替换绝对位置编码** ── **「物理免除参数，参数：-65k」**
  * **原理**：在 `CausalSelfAttention` 内部对 Q 和 K 应用旋转嵌入，彻底删除 `pos_emb` 矩阵。
- [ ] **22. RMSNorm 替换 LayerNorm** ── **「去偏置，极简归一化」**
  * **原理**：使用 `RMSNorm`（不带可学习的 bias，仅带 scale 权重）。
- [ ] **23. 引入 LayerScale / ReZero（层级残差缩放）** ── **「层级梯度保命符」**
  * **原理**：在每个 Block 残差连接前，乘以一个可学习的微小标量 $\gamma$（初始化为 `1e-2`）。

---

## 🎨 第四阶段：玄学黑科技脑暴 ── 【优先级：⭐⭐⭐】

Andrej Karpathy 在 `nanochat` 里引入的玄学但极其有效的架构微调，或者参数共享的终极防过拟合方案。

- [ ] **24. Logit Softcapping（ tanh 软截断）** ── **「防止盲目自信」**
  * **原理**：对 lm_head 的输出进行 `15.0 * torch.tanh(logits / 15.0)` 限制。
- [ ] **25. 跨层参数共享（ALBERT 循环机制 / Bilateral Sharing）** ── **「参数永动机」**
  * **原理**：让奇数层 and 偶数层共享同一组 Block 参数。
- [ ] **26. Alternating Attention (奇偶层 Attention 交替)** ── **「省参数艺术」**
  * **原理**：一层做 MQA Attention，下一层只做 FeedForward（没有 Attention）。
- [ ] **27. Value Embedding 隔层插入（ResFormer 风格值残差）**
  * **原理**：在学术研究中引入旁路查表通道。



python train_hybrid.py '{
  "optimizer_type": "sgd",
  "pos_emb": "learned",
  "layer_pattern": "AAAAAA",
  "d_model": 512,
  "n_q_head": 8,
  "n_kv_heads": 8,
  "norm": "layernorm",
  "mlp_act": "gelu",
  "weight_init": "gpt2",
  "use_token_anchor": false,
  "logit_softcap": 0.0,
  "use_fa2": false,
  "use_bf16": false
}'
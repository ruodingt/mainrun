Every 2.0s: rocm-smi                                                                                                                           7f50c7a00fc0: Fri May 29 04:52:28 2026

WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status



========================================= ROCm System Management Interface =========================================
=================================================== Concise Info ===================================================
Device  Node  IDs              Temp    Power     Partitions          SCLK  MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Edge)  (Socket)  (Mem, Compute, ID)                                                 
====================================================================================================================
0       1     0x1586,   40251  53.0°C  54.001W   N/A, N/A, 0         N/A   1000Mhz  0%   auto  N/A     7%     100%
====================================================================================================================
=============================================== End of ROCm SMI Log ================================================


```

(amd-lm) cex@192 mainrun % python tools/analyse_trace.py mainrun/profile_out/7f50c7a00fc0_542809.1780031007349888924.pt.trace.json
Loading mainrun/profile_out/7f50c7a00fc0_542809.1780031007349888924.pt.trace.json ...
GPU kernel events: 19260

============================================================
GPU UTILISATION
============================================================
  Time span:     1687.4 ms
  Compute:       1582.5 ms  (93.8%)
  Void:           104.9 ms  (6.2%)

============================================================
VOID DISTRIBUTION
============================================================
  [    0 – 2     µs]:  5247 voids      9.9 ms
  [    2 – 5     µs]:  7939 voids     16.9 ms
  [    5 – 10    µs]:  5877 voids     38.8 ms
  [   10 – 20    µs]:   139 voids      1.6 ms
  [   20 – 50    µs]:     3 voids      0.1 ms
  [   50 – 100   µs]:     2 voids      0.1 ms
  [  100 – 500   µs]:     3 voids      0.4 ms
  [  500 – 1000  µs]:    35 voids     19.6 ms
  [ 1000 – 5000  µs]:    14 voids     17.5 ms
  [ 5000 – ∞     µs]:     0 voids      0.0 ms

  Launch overhead (≤200µs): 67.8 ms  (19210 voids)
  Bubbles        (>200µs): 37.1 ms  (49 voids)

============================================================
BUBBLE DRILL-DOWN  (top 10 by size)
============================================================
  dispatch@ = when CPU sent the next kernel (via correlation id).
  CPU ops   = what the CPU ran during the void.

  ── 1,858µs void ──────────────────────────────────────
     prev: triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0
     next: void at::native::(anonymous namespace)::distribution_elementwise_
     dispatch@ +1797µs  (CPU dispatched next kernel 62µs before it started)
     CPU ops (24 events, 15 unique):
       +   121µs     15µs  aten::slice
       +   128µs      3µs  aten::as_strided
       +   151µs      6µs  aten::view
       +   166µs    113µs  aten::to
       +   175µs     10µs  aten::empty_strided
       +   295µs    111µs  aten::_to_copy
       +   301µs    104µs  aten::copy_
       +  1079µs     45µs  TorchDynamo Cache Lookup
       +  1126µs  24775µs  Torch-Compiled Region: 0/0
       +  1135µs     80µs  Pregraph bytecode
       +  1272µs     17µs  AOTDispatcher Runtime Wrapper Prologue
       +  1555µs  24311µs  CompiledFunction
       +  1763µs     72µs  aten::randint
       +  1773µs      2µs  aten::resize_
       +  1777µs     55µs  aten::random_

  ── 1,803µs void ──────────────────────────────────────
     prev: triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0
     next: void at::native::(anonymous namespace)::distribution_elementwise_
     dispatch@ +1741µs  (CPU dispatched next kernel 61µs before it started)
     CPU ops (24 events, 15 unique):
       +   117µs     15µs  aten::slice
       +   124µs      3µs  aten::as_strided
       +   148µs      6µs  aten::view
       +   163µs    109µs  aten::to
       +   166µs    105µs  aten::_to_copy
       +   172µs      9µs  aten::empty_strided
       +   183µs     88µs  aten::copy_
       +  1001µs     47µs  TorchDynamo Cache Lookup
       +  1049µs  24852µs  Torch-Compiled Region: 0/0
       +  1058µs     86µs  Pregraph bytecode
       +  1200µs     18µs  AOTDispatcher Runtime Wrapper Prologue
       +  1485µs  24369µs  CompiledFunction
       +  1706µs     74µs  aten::randint
       +  1717µs      2µs  aten::resize_
       +  1720µs     57µs  aten::random_

  ── 1,273µs void ──────────────────────────────────────
     prev: triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0
     next: void at::native::(anonymous namespace)::distribution_elementwise_
     dispatch@ +1215µs  (CPU dispatched next kernel 58µs before it started)
     CPU ops (24 events, 15 unique):
       +    88µs     10µs  aten::slice
       +    93µs      2µs  aten::as_strided
       +   109µs      4µs  aten::view
       +   120µs     98µs  aten::to
       +   123µs     95µs  aten::_to_copy
       +   127µs      6µs  aten::empty_strided
       +   134µs     83µs  aten::copy_
       +   728µs     34µs  TorchDynamo Cache Lookup
       +   762µs  15231µs  Torch-Compiled Region: 0/0
       +   769µs     67µs  Pregraph bytecode
       +   872µs     12µs  AOTDispatcher Runtime Wrapper Prologue
       +  1061µs  14897µs  CompiledFunction
       +  1191µs     56µs  aten::randint
       +  1199µs      1µs  aten::resize_
       +  1201µs     45µs  aten::random_

  ── 1,199µs void ──────────────────────────────────────
     prev: triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0
     next: void at::native::(anonymous namespace)::distribution_elementwise_
     dispatch@ +1142µs  (CPU dispatched next kernel 58µs before it started)
     CPU ops (25 events, 16 unique):
       +   110µs     11µs  aten::slice
       +   116µs      2µs  aten::as_strided
       +   129µs      4µs  aten::view
       +   139µs    113µs  aten::to
       +   141µs    111µs  aten::_to_copy
       +   147µs      8µs  aten::empty_strided
       +   157µs     96µs  aten::copy_
       +   675µs     47µs  TorchDynamo Cache Lookup
       +   723µs   8535µs  Torch-Compiled Region: 0/0
       +   728µs    135µs  Pregraph bytecode
       +   886µs     10µs  AOTDispatcher Runtime Wrapper Prologue
       +  1023µs   8220µs  CompiledFunction
       +  1114µs     60µs  aten::randint
       +  1124µs      1µs  aten::resize_
       +  1127µs     46µs  aten::random_
       +  1196µs     25µs  triton_per_fused__to_copy_add_embedding_mul_native_drop

  ── 1,164µs void ──────────────────────────────────────
     prev: void at::native::(anonymous namespace)::multi_tensor_apply_kernel
     next: triton_red_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_m
     dispatch@ +-87569µs  (CPU dispatched next kernel 88734µs before it started)
     CPU ops: none (CPU between interpreter ticks or blocking)

  ── 1,157µs void ──────────────────────────────────────
     prev: void at::native::(anonymous namespace)::multi_tensor_apply_kernel
     next: triton_red_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_m
     dispatch@ +-81879µs  (CPU dispatched next kernel 83037µs before it started)
     CPU ops: none (CPU between interpreter ticks or blocking)

  ── 1,150µs void ──────────────────────────────────────
     prev: void at::native::(anonymous namespace)::multi_tensor_apply_kernel
     next: triton_red_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_m
     dispatch@ +-129259µs  (CPU dispatched next kernel 130409µs before it started)
     CPU ops: none (CPU between interpreter ticks or blocking)

  ── 1,143µs void ──────────────────────────────────────
     prev: void at::native::(anonymous namespace)::multi_tensor_apply_kernel
     next: triton_red_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_m
     dispatch@ +-133333µs  (CPU dispatched next kernel 134477µs before it started)
     CPU ops: none (CPU between interpreter ticks or blocking)

  ── 1,142µs void ──────────────────────────────────────
     prev: void at::native::(anonymous namespace)::multi_tensor_apply_kernel
     next: triton_red_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_m
     dispatch@ +-130969µs  (CPU dispatched next kernel 132111µs before it started)
     CPU ops: none (CPU between interpreter ticks or blocking)

  ── 1,141µs void ──────────────────────────────────────
     prev: void at::native::(anonymous namespace)::multi_tensor_apply_kernel
     next: triton_red_fused__to_copy_add_copy__div_lerp_linalg_vector_norm_m
     dispatch@ +-131416µs  (CPU dispatched next kernel 132557µs before it started)
     CPU ops: none (CPU between interpreter ticks or blocking)

============================================================
TOP 20 KERNELS BY TOTAL GPU TIME
============================================================
   Total(ms)  Count   Avg(µs)  Name
      231.10   2040     113.3  Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_L
      225.33   1400     161.0  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x64_MI16x16x1_SN_L
      129.14    890     145.1  Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_L
       74.11    280     264.7  attn_fwd
       62.95    280     224.8  triton_poi_fused__unsafe_view_add_fill_mul_sigmoid_silu_sub_view_5
       50.83    290     175.3  Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32_MI16x16x1_SN_L
       50.59    610      82.9  Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x160x32_MI16x16x1_SN_
       46.64    280     166.6  bwd_kernel_dk_dv
       43.89    280     156.7  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x48x32_MI16x16x1_SN_L
       39.61     10    3960.8  triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_add_
       39.58    210     188.5  triton_per_fused__to_copy_add_mul_native_dropout_backward_native_layer
       37.65    110     342.3  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_L
       37.52    270     139.0  triton_per_fused__to_copy__unsafe_view_add_native_dropout_native_layer
       34.36    280     122.7  bwd_kernel_dq
       33.08    520      63.6  triton_red_fused__to_copy_add_native_layer_norm_backward_view_7
       31.96    280     114.1  triton_poi_fused__unsafe_view_mul_silu_9
       25.09     80     313.6  triton_per_fused__to_copy_add_mul_native_dropout_native_dropout_backwa
       24.70     10    2470.0  triton_red_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_on
       20.43   1240      16.5  triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0
       18.36    120     153.0  triton_per_fused__to_copy_add_mul_native_dropout_backward_native_layer

============================================================
SHORT-LIVED KERNELS  (avg < 20µs, count ≥ 50)
============================================================
  16 distinct types, 7990 total calls, 54.4 ms compute
  Estimated launch overhead: ~40.0 ms  (5µs/launch)

   Total(ms)  Count   Avg(µs)  Name
       20.43   1240      16.5  triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0
        7.47    840       8.9  triton_poi_fused__to_copy_t_8
        4.82    280      17.2  triton_poi_fused__unsafe_view_clone_transpose_view_15
        3.77    240      15.7  triton_poi_fused__to_copy__unsafe_view_add_cat_clone_mul_select_slice_
        3.32    280      11.9  bwd_preprocess
        3.01    280      10.7  triton_poi_fused_clone_transpose_6
        2.42    560       4.3  triton_poi_fused__to_copy_t_1
        1.78    840       2.1  triton_poi_fused__to_copy_4
        1.77   1140       1.6  triton_per_fused__to_copy_native_layer_norm_backward_view_3
        1.71    280       6.1  triton_poi_fused__to_copy_t_2
        0.95    560       1.7  triton_poi_fused__to_copy_8
        0.71    560       1.3  void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc
        0.70     50      14.0  triton_poi_fused__to_copy__unsafe_view_clone_embedding_dense_backward_
        0.60    280       2.2  triton_poi_fused__to_copy_14
        0.54    280       1.9  triton_red_fused_mul_native_dropout_sum_23
        0.36    280       1.3  void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc

============================================================
KERNEL PAIRS WITH NOTABLE VOIDS  (≥10 occurrences, avg void ≥5µs)
============================================================
  TotalVoid(ms)     N  AvgVoid(µs)  Pair
          11.40    10       1140.2  multi_tensor_op → triton_red_fused__to_copy_add_copy__div_lerp_linal
          10.46    20        523.1  multi_tensor_op → triton_red_fused__to_copy_lerp_linalg_vector_norm_
           5.78   940          6.1  GEMM:_Ail → GEMM:_Ail
           5.70    10        569.7  triton_poi_fused__to_copy_add_34 → triton_red_fused__to_copy_add_copy__div_lerp_linal
           5.18   840          6.2  GEMM:_Ail → triton_poi_fused__to_copy_4
           1.76   280          6.3  triton_poi_fused__unsafe_view_add_fill_mul_sigmoid → GEMM:_Ail
           1.75   280          6.3  bwd_kernel_dk_dv → bwd_kernel_dq
           1.73   280          6.2  GEMM:_Ail → triton_poi_fused__to_copy_14
           1.49   240          6.2  bwd_kernel_dq → triton_poi_fused__to_copy_add_mul_neg_slice_slice_
           1.26   200          6.3  triton_per_fused__to_copy_add_mul_native_dropout_b → GEMM:_Ail
           1.18   200          5.9  GEMM:_Ail → triton_per_fused__to_copy_add_mul_native_dropout_b
           1.16   190          6.1  triton_per_fused__to_copy_add_mul_native_dropout_b → triton_red_fused__to_copy_add_native_layer_norm_ba
           1.07   170          6.3  GEMM:_Ail → triton_poi_fused__unsafe_view_add_fill_mul_sigmoid
           1.06   170          6.2  GEMM:_Ail → triton_poi_fused__unsafe_view_clone_transpose_view
           0.65   100          6.5  GEMM:_Ali → GEMM:_Ail
           0.65   100          6.5  triton_per_fused__to_copy_add_mul_native_dropout_n → triton_red_fused_mul_native_dropout_sum_23
           0.36    10         35.8  dropout_fill → triton_per_fused__to_copy_add_embedding_mul_native
           0.32    30         10.7  triton_red_fused__to_copy_add_mul_native_dropout_n → triton_red_fused_mul_native_dropout_sum_23
           0.27    40          6.9  GEMM:_Ail → triton_poi_fused_add_mul_8
           0.27    40          6.8  GEMM:_Ail → triton_poi_fused_add_mul_6

```
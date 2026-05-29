```
============================================================
GPU UTILISATION
============================================================
  Time span:        1671.4 ms
  Compute:          1566.4 ms  (93.7%)
  Idle (gaps):       105.0 ms  (6.3%)

============================================================
GAP DISTRIBUTION
============================================================
  [    0 – 2     µs]:  5937 gaps     11.1 ms
  [    2 – 5     µs]:  7161 gaps     15.1 ms
  [    5 – 10    µs]:  5962 gaps     39.2 ms
  [   10 – 20    µs]:   139 gaps      1.6 ms
  [   20 – 50    µs]:     6 gaps      0.1 ms
  [   50 – 100   µs]:     2 gaps      0.2 ms
  [  100 – 500   µs]:     3 gaps      0.4 ms
  [  500 – 1000  µs]:    37 gaps     21.4 ms
  [ 1000 – 5000  µs]:    12 gaps     15.8 ms
  [ 5000 – ∞     µs]:     0 gaps      0.0 ms

  Launch overhead (≤200µs): 67.8 ms
  Real bubbles   (>200µs):  37.2 ms  (49 events)

  Real bubbles by source:
      21.9 ms |  30x avg 730 µs | Muon multi_tensor_op
       9.6 ms |   9x avg 1070 µs | Muon nesterov step
       5.6 ms |  10x avg 562 µs | triton_poi_fused__to_copy_add_34

============================================================
TOP 20 KERNELS BY TOTAL GPU TIME
============================================================
   Total(ms)  Count   Avg(µs)  Name
      227.09   2040     111.3  Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_L
      219.75   1400     157.0  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x64_MI16x16x1_SN_L
      127.87    890     143.7  Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_L
       72.42    280     258.6  attn_fwd
       62.90    280     224.6  triton_poi_fused__unsafe_view_add_fill_mul_sigmoid_silu_sub_view_5
       50.25    290     173.3  Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32_MI16x16x1_SN_L
       50.25    610      82.4  Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x160x32_MI16x16x1_SN_
       45.81    280     163.6  bwd_kernel_dk_dv
       42.87    280     153.1  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x48x32_MI16x16x1_SN_L
       39.78    210     189.4  triton_per_fused__to_copy_add_mul_native_dropout_backward_native_layer
       39.64     10    3964.4  triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_add_
       37.44    110     340.4  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_L
       37.43    270     138.6  triton_per_fused__to_copy__unsafe_view_add_native_dropout_native_layer
       33.78    280     120.6  bwd_kernel_dq
       32.96    520      63.4  triton_red_fused__to_copy_add_native_layer_norm_backward_view_7
       31.97    280     114.2  triton_poi_fused__unsafe_view_mul_silu_9
       25.25     80     315.6  triton_per_fused__to_copy_add_mul_native_dropout_native_dropout_backwa
       24.67     10    2467.3  triton_red_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_on
       20.50   1240      16.5  triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0
       18.40    120     153.4  triton_per_fused__to_copy_add_mul_native_dropout_backward_native_layer

============================================================
SHORT-LIVED KERNELS  (avg < 20µs, count ≥ 50)
============================================================
  16 distinct kernel types, 7990 total calls
  Actual compute: 54.2 ms
  Estimated launch overhead: ~40.0 ms  (assuming 5µs/launch)

   Total(ms)  Count   Avg(µs)  Name
       20.50   1240      16.5  triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0
        7.52    840       9.0  triton_poi_fused__to_copy_t_8
        4.84    280      17.3  triton_poi_fused__unsafe_view_clone_transpose_view_15
        3.75    240      15.6  triton_poi_fused__to_copy__unsafe_view_add_cat_clone_mul_select_slice_
        3.26    280      11.6  bwd_preprocess
        2.99    280      10.7  triton_poi_fused_clone_transpose_6
        2.44    560       4.4  triton_poi_fused__to_copy_t_1
        1.72    840       2.0  triton_poi_fused__to_copy_4
        1.68    280       6.0  triton_poi_fused__to_copy_t_2
        1.68   1140       1.5  triton_per_fused__to_copy_native_layer_norm_backward_view_3
        0.93    560       1.7  triton_poi_fused__to_copy_8
        0.69     50      13.7  triton_poi_fused__to_copy__unsafe_view_clone_embedding_dense_backward_
        0.69    560       1.2  void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc
        0.62    280       2.2  triton_poi_fused__to_copy_14
        0.54    280       1.9  triton_red_fused_mul_native_dropout_sum_23
        0.36    280       1.3  void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc

============================================================
KERNEL PAIRS WITH NOTABLE GAPS  (≥10 occurrences, avg gap ≥5µs)
============================================================
   TotalGap(ms)     N  AvgGap(µs)  Pair
          11.39    10      1138.8  multi_tensor_op → triton_red_fused__to_copy_add_copy__div_lerp_linal
          10.52    20       526.0  multi_tensor_op → triton_red_fused__to_copy_lerp_linalg_vector_norm_
           5.82   940         6.2  GEMM:_Ail → GEMM:_Ail
           5.62    10       562.0  triton_poi_fused__to_copy_add_34 → triton_red_fused__to_copy_add_copy__div_lerp_linal
           5.21   840         6.2  GEMM:_Ail → triton_poi_fused__to_copy_4
           1.78   280         6.3  triton_poi_fused__unsafe_view_add_fill_mul_sigmoid → GEMM:_Ail
           1.75   280         6.3  GEMM:_Ail → triton_poi_fused__to_copy_14
           1.74   280         6.2  bwd_kernel_dk_dv → bwd_kernel_dq
           1.54   240         6.4  bwd_kernel_dq → triton_poi_fused__to_copy_add_mul_neg_slice_slice_
           1.28   200         6.4  triton_per_fused__to_copy_add_mul_native_dropout_b → GEMM:_Ail
           1.18   190         6.2  triton_per_fused__to_copy_add_mul_native_dropout_b → triton_red_fused__to_copy_add_native_layer_norm_ba
           1.15   200         5.7  GEMM:_Ail → triton_per_fused__to_copy_add_mul_native_dropout_b
           1.08   170         6.4  GEMM:_Ail → triton_poi_fused__unsafe_view_add_fill_mul_sigmoid
           1.06   170         6.2  GEMM:_Ail → triton_poi_fused__unsafe_view_clone_transpose_view
           0.65   100         6.5  triton_per_fused__to_copy_add_mul_native_dropout_n → triton_red_fused_mul_native_dropout_sum_23
           0.64   100         6.4  GEMM:_Ali → GEMM:_Ail
           0.32    30        10.8  triton_red_fused__to_copy_add_mul_native_dropout_n → triton_red_fused_mul_native_dropout_sum_23
           0.31    10        30.8  dropout_fill → triton_per_fused__to_copy_add_embedding_mul_native
           0.27    40         6.8  GEMM:_Ail → triton_poi_fused_add_mul_6
           0.27    40         6.8  GEMM:_Ail → triton_poi_fused_add_mul_4
```
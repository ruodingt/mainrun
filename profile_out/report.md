# Profile Report

**Date:** 2026-05-28 15:03  
**Config:** `AAAEAAAAAAAEAAAAAAAEAAAAAAAE`  
**Model:** 35.6M params, 28L × 256d, vocab=10240  
**Throughput:** 35,029 tok/s  

## Graph Breaks

| | |
|---|---|
| Graphs | 1 |
| Breaks | 0 |

## Memory

| | GB |
|---|---|
| Allocated (peak) | 4.18 |
| Reserved (peak)  | 6.45 |

## Top CPU-Dispatch Overhead

| Op | Count | CPU time | per call |
|---|---|---|---|
| `backward` | 10 | 245.7ms | 24569µs |
| `autograd::engine::evaluate_function: CompiledFunctionBackwar` | 10 | 222.3ms | 22231µs |
| `CompiledFunctionBackward` | 10 | 220.9ms | 22088µs |
| `## Call CompiledFxGraph fbvcwuqulrovfujd7iuc223hzjvp3lnu3ph7` | 10 | 216.7ms | 21675µs |
| `optimizer_step` | 10 | 179.2ms | 17924µs |
| `Optimizer.step#MuonAdamW.step` | 10 | 178.9ms | 17892µs |
| `forward` | 10 | 151.8ms | 15182µs |
| `Torch-Compiled Region: 0/0` | 10 | 150.8ms | 15076µs |
| `CompiledFunction` | 10 | 146.6ms | 14664µs |
| `## Call CompiledFxGraph fol3rwx4wrgfnytmdj2va37jd23ploz2a7vu` | 10 | 142.5ms | 14254µs |
| `aten::mm` | 5190 | 118.7ms | 23µs |
| `hipModuleLaunchKernel` | 12240 | 86.7ms | 7µs |
| `Torch-Compiled Region: 2/2` | 1140 | 73.4ms | 64µs |
| `## Call CompiledFxGraph fbmxl54y4yvqkz2ubnufvdfv4ujzt3fo5sf7` | 1140 | 56.7ms | 50µs |
| `hipExtModuleLaunchKernel` | 5830 | 44.9ms | 8µs |
| `triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0` | 1240 | 30.6ms | 25µs |
| `aten::_scaled_dot_product_flash_attention_backward` | 280 | 24.6ms | 88µs |
| `aten::copy_` | 3660 | 23.8ms | 6µs |
| `aten::_scaled_dot_product_flash_attention` | 280 | 20.5ms | 73µs |
| `aten::_flash_attention_backward` | 280 | 20.4ms | 73µs |

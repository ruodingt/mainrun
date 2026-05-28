# Profile Report

**Date:** 2026-05-28 13:08  
**Config:** `AAAEAAAAAAAEAAAAAAAEAAAAAAAE`  
**Model:** 35.6M params, 28L × 256d, vocab=10240  
**Throughput:** 35,310 tok/s  

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

## GPU Bubble Analysis

> ROCm: hipDeviceSynchronize used as GPU busy proxy

| | ms | % |
|---|---|---|
| Wall time | 2320.1 | 100% |
| GPU busy  | 1002.8 | 43.2% |
| Bubble    | 1317.2 | 56.8% |

## Top CPU-Dispatch Overhead

| Op | Count | CPU time | per call |
|---|---|---|---|
| `backward` | 10 | 316.5ms | 31649µs |
| `autograd::engine::evaluate_function: CompiledFunctionBackwar` | 10 | 292.6ms | 29257µs |
| `CompiledFunctionBackward` | 10 | 291.1ms | 29108µs |
| `## Call CompiledFxGraph fbvcwuqulrovfujd7iuc223hzjvp3lnu3ph7` | 10 | 285.0ms | 28501µs |
| `optimizer_step` | 10 | 233.7ms | 23373µs |
| `Optimizer.step#MuonAdamW.step` | 10 | 233.4ms | 23341µs |
| `forward` | 10 | 178.0ms | 17799µs |
| `Torch-Compiled Region: 0/0` | 10 | 177.1ms | 17711µs |
| `CompiledFunction` | 10 | 174.0ms | 17398µs |
| `## Call CompiledFxGraph fol3rwx4wrgfnytmdj2va37jd23ploz2a7vu` | 10 | 170.1ms | 17014µs |
| `aten::mm` | 5190 | 162.7ms | 31µs |
| `hipModuleLaunchKernel` | 12240 | 147.0ms | 12µs |
| `Torch-Compiled Region: 2/2` | 1140 | 99.8ms | 88µs |
| `## Call CompiledFxGraph fbmxl54y4yvqkz2ubnufvdfv4ujzt3fo5sf7` | 1140 | 78.8ms | 69µs |
| `hipExtModuleLaunchKernel` | 5830 | 78.1ms | 13µs |
| `triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0` | 1240 | 47.3ms | 38µs |
| `aten::_scaled_dot_product_flash_attention_backward` | 280 | 32.9ms | 118µs |
| `aten::_flash_attention_backward` | 280 | 28.2ms | 101µs |
| `aten::copy_` | 3660 | 27.9ms | 8µs |
| `aten::_scaled_dot_product_flash_attention` | 280 | 24.2ms | 86µs |

## Top Ops by CUDA Self Time

| Op | Count | CUDA Total | CUDA% | Avg/call | CPU Total |
|---|---|---|---|---|---|
| `aten::slice` | 30 | 0.0ms | 0.0% | 0µs | 0.2ms |
| `aten::as_strided` | 11790 | 0.0ms | 0.0% | 0µs | 6.2ms |
| `aten::view` | 580 | 0.0ms | 0.0% | 0µs | 0.6ms |
| `aten::to` | 20 | 0.0ms | 0.0% | 0µs | 1.6ms |
| `aten::_to_copy` | 20 | 0.0ms | 0.0% | 0µs | 1.5ms |
| `aten::empty_strided` | 1430 | 0.0ms | 0.0% | 0µs | 2.7ms |
| `aten::copy_` | 3660 | 0.0ms | 0.0% | 0µs | 27.9ms |
| `hipStreamGetCaptureInfo` | 30 | 0.0ms | 0.0% | 0µs | 0.0ms |
| `hipMemcpyWithStream` | 30 | 0.0ms | 0.0% | 0µs | 10.5ms |
| `Memcpy HtoD (Host -> Device)` | 12 | 0.0ms | 0.0% | 0µs | 0.0ms |
| `Optimizer.zero_grad#MuonAdamW.zero_grad` | 10 | 0.0ms | 0.0% | 0µs | 3.6ms |
| `forward` | 10 | 0.0ms | 0.0% | 0µs | 178.0ms |
| `TorchDynamo Cache Lookup` | 1290 | 0.0ms | 0.0% | 0µs | 7.2ms |
| `Torch-Compiled Region: 0/0` | 10 | 0.0ms | 0.0% | 0µs | 177.1ms |
| `Pregraph bytecode` | 1290 | 0.0ms | 0.0% | 0µs | 2.9ms |
| `AOTDispatcher Runtime Wrapper Prologue` | 1290 | 0.0ms | 0.0% | 0µs | 3.3ms |
| `CompiledFunction` | 10 | 0.0ms | 0.0% | 0µs | 174.0ms |
| `## Call CompiledFxGraph fol3rwx4wrgfnytmdj2va37jd23ploz2a7vu` | 10 | 0.0ms | 0.0% | 0µs | 170.1ms |
| `aten::randint` | 10 | 0.0ms | 0.0% | 0µs | 0.8ms |
| `aten::resize_` | 10 | 0.0ms | 0.0% | 0µs | 0.0ms |

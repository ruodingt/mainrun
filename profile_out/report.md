# Profile Report

**Date:** 2026-05-28 12:23  
**Config:** `AAAEAAAAAAAEAAAAAAAEAAAAAAAE`  
**Model:** 35.6M params, 28L × 256d, vocab=10240  
**Steps:** 3 warmup + 10 profiled  
**Throughput:** 35,917 tok/s  

## Memory

| | GB |
|---|---|
| Allocated (peak) | 4.19 |
| Reserved (peak)  | 4.38 |

## GPU Bubble Analysis

> ROCm mode: hipDeviceSynchronize used as GPU busy proxy

| | ms | % |
|---|---|---|
| Wall time | 2280.8 | 100% |
| GPU busy  | 1249.4 | 54.8% |
| Bubble    | 1031.4 | 45.2% |

## Top CPU-Dispatch Overhead (Bubble Sources)

| Op | Count | CPU time | per call |
|---|---|---|---|
| `backward` | 10 | 177.2ms | 17724µs |
| `autograd::engine::evaluate_function: CompiledFunctionBackwar` | 10 | 158.0ms | 15802µs |
| `CompiledFunctionBackward` | 10 | 157.0ms | 15699µs |
| `## Call CompiledFxGraph fbvcwuqulrovfujd7iuc223hzjvp3lnu3ph7` | 10 | 154.3ms | 15432µs |
| `optimizer_step` | 10 | 144.0ms | 14399µs |
| `Optimizer.step#MuonAdamW.step` | 10 | 143.7ms | 14372µs |
| `forward` | 10 | 95.4ms | 9541µs |
| `Torch-Compiled Region: 0/0` | 10 | 94.9ms | 9489µs |
| `CompiledFunction` | 10 | 92.5ms | 9255µs |
| `## Call CompiledFxGraph fol3rwx4wrgfnytmdj2va37jd23ploz2a7vu` | 10 | 90.1ms | 9007µs |
| `aten::mm` | 5190 | 83.8ms | 16µs |
| `Torch-Compiled Region: 2/2` | 1140 | 61.1ms | 54µs |
| `hipModuleLaunchKernel` | 12240 | 47.1ms | 4µs |
| `## Call CompiledFxGraph fbmxl54y4yvqkz2ubnufvdfv4ujzt3fo5sf7` | 1140 | 46.9ms | 41µs |
| `triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0` | 1240 | 23.8ms | 19µs |
| `hipExtModuleLaunchKernel` | 5830 | 23.3ms | 4µs |
| `aten::_scaled_dot_product_flash_attention_backward` | 280 | 18.5ms | 66µs |
| `aten::copy_` | 3660 | 15.7ms | 4µs |
| `aten::_flash_attention_backward` | 280 | 14.9ms | 53µs |
| `aten::_scaled_dot_product_flash_attention` | 280 | 13.1ms | 47µs |

## Top Ops by CUDA Self Time

| Op | Count | CUDA Total | CUDA% | Avg/call | CPU Total |
|---|---|---|---|---|---|
| `aten::slice` | 30 | 0.0ms | 0.0% | 0µs | 0.1ms |
| `aten::as_strided` | 11790 | 0.0ms | 0.0% | 0µs | 4.8ms |
| `aten::view` | 580 | 0.0ms | 0.0% | 0µs | 0.4ms |
| `aten::to` | 20 | 0.0ms | 0.0% | 0µs | 1.1ms |
| `aten::_to_copy` | 20 | 0.0ms | 0.0% | 0µs | 1.0ms |
| `aten::empty_strided` | 1430 | 0.0ms | 0.0% | 0µs | 2.0ms |
| `aten::copy_` | 3660 | 0.0ms | 0.0% | 0µs | 15.7ms |
| `hipStreamGetCaptureInfo` | 30 | 0.0ms | 0.0% | 0µs | 0.0ms |
| `hipMemcpyWithStream` | 30 | 0.0ms | 0.0% | 0µs | 1.0ms |
| `Memcpy HtoD (Host -> Device)` | 20 | 0.0ms | 0.0% | 0µs | 0.0ms |
| `Optimizer.zero_grad#MuonAdamW.zero_grad` | 10 | 0.0ms | 0.0% | 0µs | 2.6ms |
| `forward` | 10 | 0.0ms | 0.0% | 0µs | 95.4ms |
| `TorchDynamo Cache Lookup` | 1290 | 0.0ms | 0.0% | 0µs | 4.6ms |
| `Torch-Compiled Region: 0/0` | 10 | 0.0ms | 0.0% | 0µs | 94.9ms |
| `Pregraph bytecode` | 1290 | 0.0ms | 0.0% | 0µs | 1.9ms |
| `AOTDispatcher Runtime Wrapper Prologue` | 1290 | 0.0ms | 0.0% | 0µs | 2.1ms |
| `CompiledFunction` | 10 | 0.0ms | 0.0% | 0µs | 92.5ms |
| `## Call CompiledFxGraph fol3rwx4wrgfnytmdj2va37jd23ploz2a7vu` | 10 | 0.0ms | 0.0% | 0µs | 90.1ms |
| `aten::randint` | 10 | 0.0ms | 0.0% | 0µs | 0.4ms |
| `aten::resize_` | 10 | 0.0ms | 0.0% | 0µs | 0.0ms |

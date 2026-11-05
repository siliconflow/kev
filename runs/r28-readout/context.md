# Validated context length, Kev 1.0 (round 28's registered rule, report only)

longdoc-v1 development; CUAD accuracy per nominal bucket, paired difference from the 8k bucket (same target contract, repeat and question; 95 % target-clustered bootstrap, 2,000 resamples, seed 0), ECE at the shipped T. Within tolerance: lower bound >= -3 pp and every record answered.

| size | validated context | first failing bucket |
|---|---|---|
| Kev-0.8B | 8,192 | 16k |
| Kev-4B | 8,192 | 16k |
| Kev-9B v2 | 8,192 | 16k |
| Kev-27B v2 | 65,536 | none |

### Kev-0.8B (`runs/r28-08b-r15-longdoc`, T = 2.35)

| bucket | CUAD questions | accuracy | ECE | Δ vs 8k [95 % CI] (paired questions) | within |
|---|---|---|---|---|---|
| 4k | 443 | 0.779 | 0.128 | - |  |
| 8k | 453 | 0.711 | 0.066 | reference |  |
| 16k | 452 | 0.659 | 0.047 | -5.2 [-8.5, -2.1] (445) | no |
| 32k | 454 | 0.663 | 0.055 | -6.0 [-9.5, -2.5] (447) | no |
| 64k | 452 | 0.637 | 0.038 | -7.9 [-11.8, -4.2] (445) | no |

### Kev-4B (`runs/r28-4b-r10-longdoc`, T = 2.41)

| bucket | CUAD questions | accuracy | ECE | Δ vs 8k [95 % CI] (paired questions) | within |
|---|---|---|---|---|---|
| 4k | 443 | 0.847 | 0.047 | - |  |
| 8k | 453 | 0.837 | 0.048 | reference |  |
| 16k | 452 | 0.823 | 0.057 | -1.1 [-3.4, +1.2] (445) | no |
| 32k | 454 | 0.788 | 0.022 | -5.8 [-9.0, -2.8] (447) | no |
| 64k | 452 | 0.781 | 0.035 | -5.2 [-8.2, -2.0] (445) | no |

### Kev-9B v2 (`runs/r29-9b-r18a-longdoc`, T = 2.19)

| bucket | CUAD questions | accuracy | ECE | Δ vs 8k [95 % CI] (paired questions) | within |
|---|---|---|---|---|---|
| 4k | 443 | 0.876 | 0.049 | - |  |
| 8k | 453 | 0.850 | 0.051 | reference |  |
| 16k | 452 | 0.839 | 0.033 | -1.4 [-3.7, +0.9] (445) | no |
| 32k | 454 | 0.819 | 0.041 | -3.6 [-6.4, -0.9] (447) | no |
| 64k | 452 | 0.801 | 0.056 | -5.2 [-7.9, -2.5] (445) | no |

### Kev-27B v2 (`runs/r23-27b-k-w85-longdoc`, T = 1.32)

| bucket | CUAD questions | accuracy | ECE | Δ vs 8k [95 % CI] (paired questions) | within |
|---|---|---|---|---|---|
| 4k | 443 | 0.869 | 0.095 | - |  |
| 8k | 453 | 0.843 | 0.110 | reference |  |
| 16k | 452 | 0.845 | 0.106 | +0.2 [-0.7, +1.2] (445) | yes |
| 32k | 454 | 0.846 | 0.102 | -0.2 [-1.2, +0.7] (447) | yes |
| 64k | 452 | 0.832 | 0.106 | -1.1 [-2.4, +0.0] (445) | yes |

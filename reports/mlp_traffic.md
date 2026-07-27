# PolyKernel MLP fusion memory-traffic report (Todo 17 / Wave 3)

Input: `examples/mlp_block.mlir`

Traffic is ANALYTIC: per-op global bytes read + written, derived from tensor
shapes + dtype. Fusion removes each eliminated intermediate's global
round-trip (one write + one read).

## Headline

- **Before fusion:** 478,150,656 bytes (456.00 MiB)
- **After fusion:**  444,596,224 bytes (424.00 MiB)
- **Reduction:**     33,554,432 bytes (32.00 MiB)  = 7.02%
- **Consistent** (before-after == eliminated round-trips): True

## Per-fused-op breakdown

- `fused_rmsnorm_matmul` (fused_from=`rmsnorm_matmul`) eliminates 32.00 MiB intermediate write+read (33,554,432 bytes) from tensor<1x2048x4096xbf16>

## Unfused per-op traffic

| op | result | reads | writes | traffic |
|---|---|---|---|---|
| rmsnorm | %0 | 16,777,216 | 16,777,216 | 33,554,432 |
| matmul | %1 | 106,954,752 | 45,088,768 | 152,043,520 |
| gelu | %2 | 45,088,768 | 45,088,768 | 90,177,536 |
| matmul | %3 | 135,266,304 | 16,777,216 | 152,043,520 |
| add | %4 | 33,554,432 | 16,777,216 | 50,331,648 |

## Fused per-op traffic

| op | result | reads | writes | traffic | workspace |
|---|---|---|---|---|---|
| fused_rmsnorm_matmul | %0 | 106,954,752 | 45,088,768 | 152,043,520 | 0 |
| gelu | %1 | 45,088,768 | 45,088,768 | 90,177,536 | n/a |
| matmul | %2 | 135,266,304 | 16,777,216 | 152,043,520 | 16,777,216 |
| add | %3 | 33,554,432 | 16,777,216 | 50,331,648 | n/a |

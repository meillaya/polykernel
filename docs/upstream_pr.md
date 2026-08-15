# Upstream contribution (prepared, not submitted)

**Status: PREPARED, NOT SUBMITTED. This deliverable is optional and never blocking.**

PolyKernel's MVP (Waves 1 through 7) is complete and green independent of this
file. Nothing here was pushed, forked, or opened as a pull request. This
environment has no GitHub credentials and no cloud budget, and the plan marks
this task "NEVER blocking." What follows is a small, self-contained contribution
drafted against one named upstream, with the diff inline and an honest account
of where it stands.

## Summary

| Field | Value |
|---|---|
| Target upstream | [Triton](https://github.com/triton-lang/triton) (`triton-lang/triton`) |
| Contribution type | A new tutorial: fused MatMul + Bias + GELU (epilogue fusion) |
| Files touched | `python/tutorials/11-fused-matmul-bias-gelu.py` (new), gallery registration (one line, conditional) |
| Size | One self-contained example file, 172 lines, no library changes |
| Submission status | **Prepared, NOT submitted** (no credentials here; non-blocking) |
| Local validation | Syntax-checked with `py_compile`; runtime check needs a GPU plus Triton, so it is deferred to submission time |

## Why Triton

Triton is the closest conceptual neighbor PolyKernel has. PolyKernel's
correctness-gated autotuner searches a bounded grid of `BLOCK_M`, `BLOCK_N`,
`BLOCK_K`, `num_warps`, and pipeline `stages` (see
[`include/PolyKernel/Autotune/ConfigSpace.h`](../include/PolyKernel/Autotune/ConfigSpace.h)).
That is exactly the configuration space a Triton `@triton.autotune` decorator
searches with its `triton.Config` objects. The two systems ask the same
question of a GEMM: which tile shape and warp count runs fastest for this
problem shape?

PolyKernel's flagship fused kernel is `fused_matmul_bias_gelu`, an epilogue
fusion that computes `out = gelu(matmul(a, b) + bias)` so the matmul and bias
intermediates never round-trip through global memory (see
[`kernels/generated/fused_matmul_bias_gelu.cu`](../kernels/generated/fused_matmul_bias_gelu.cu)).
Triton's own matrix-multiplication tutorial (`python/tutorials/03-matrix-multiplication.py`)
already invites this exact move. Its GEMM loop carries the comment "You can
fuse arbitrary activation functions here while the accumulator is still in
FP32," and it ships a `leaky_relu` example. A worked bias-plus-GELU epilogue,
the canonical transformer feed-forward pattern, is a natural and genuinely
useful addition to that gallery.

The other named upstreams are weaker fits for a small, self-contained patch:

- **Modal examples**: PolyKernel already has a full Modal app in `modal/app.py`,
  but contributing it means contributing most of the repo. Too large for this task.
- **ROCm / ROCm-examples**: a HIP kernel example would work, but PolyKernel's
  HIP story is the portable template plus the ISA analyzer, which only make
  sense alongside the compiler. Hard to isolate into one small file.
- **vLLM**: a production inference stack. PolyKernel explicitly does not claim
  to compete with it, so a contribution there would misrepresent the project.
- **MLIR**: PolyKernel is an out-of-tree dialect. Upstreaming a dialect is a
  large design discussion, not a small example, and Triton's compiler is its
  own MLIR fork with different conventions.
- **Cerebras SDK examples**: PolyKernel's dataflow piece is a simulator, not
  CSL, so it cannot be contributed as CSL code (see the guardrails below).

## What the contribution is

One new tutorial file, `python/tutorials/11-fused-matmul-bias-gelu.py`. It
reuses the tiled, autotuned GEMM from tutorial 03 and adds two operations to
the epilogue, applied while the accumulator is still fp32:

1. A per-column bias broadcast (`bias` has shape `(N,)`, broadcast over the `M` rows).
2. An exact erf-based GELU, `0.5 * x * (1 + erf(x / sqrt(2)))`, via `tl.math.erf`.

The wrapper checks shapes, allocates the output, and launches the kernel over
a 1D grid of output tiles. A verification block at the bottom compares the
fused result against `torch.nn.functional.gelu(a @ b + bias)` with a tolerance.

The file number `11` is the next free index after the current
`10-block-scaled-matmul.py`. Exact placement and numbering are a maintainer
decision; appending avoids any renumbering of existing tutorials.

## Numerical convention, and how it differs from PolyKernel

This is worth stating plainly so the contribution is not misread as a
PolyKernel benchmark in disguise.

The Triton tutorial keeps the epilogue in fp32 and casts once, at the store,
to the output dtype. That is the idiomatic high-performance approach and what
tutorial 03's "fuse while the accumulator is still in FP32" comment points to.

PolyKernel's golden harness uses a stricter convention: bf16 inputs, fp32
accumulate, and a bf16 round after *each* stage (matmul, then bias, then GELU)
so the fused kernel is bit-identical to the unfused primitive chain. The
Triton example does not replicate that per-stage rounding, and it does not
claim to. It validates against a PyTorch reference to a tolerance
(`atol=1e-2, rtol=1e-2`), which is normal tutorial practice.

## Scope guardrails carried into the contribution

The contribution keeps PolyKernel's honesty rules intact:

- The dataflow simulator elsewhere in this repo is a functional and cycle
  **model** of a Cerebras-style fabric. It is not CSL, not `cslc`, and it does
  not run on Cerebras hardware. None of that touches this Triton example.
- No performance claim is made against vLLM or any production stack. The
  tutorial's only claim is correctness versus a PyTorch reference.
- Any speedup numbers in the wider repo that come from a roofline model rather
  than a measured GPU run are labeled **PROJECTED** there. This file makes no
  speedup claim at all.

## The contribution, inline

A new-file unified diff. Every line of the file is an added (`+`) line.

```diff
diff --git a/python/tutorials/11-fused-matmul-bias-gelu.py b/python/tutorials/11-fused-matmul-bias-gelu.py
new file mode 100644
--- /dev/null
+++ b/python/tutorials/11-fused-matmul-bias-gelu.py
@@ -0,0 +1,172 @@
+"""
+Fused MatMul + Bias + GELU (epilogue fusion)
+=============================================
+
+A matrix multiply is rarely the whole story. In a transformer feed-forward
+block the GEMM is immediately followed by a bias add and a nonlinearity.
+Running those as three separate kernels pays the round trip through global
+memory twice: the GEMM writes its output, the bias kernel reads it back and
+writes again, and the activation kernel does the same. Epilogue fusion folds
+the bias and the activation into the GEMM's store, so the intermediate
+results never leave on-chip registers.
+
+This tutorial builds on :code:`03-matrix-multiplication.py`. It reuses the
+same tiled, autotuned GEMM and adds two operations to the epilogue: a
+per-column bias broadcast and an exact erf-based GELU, both applied while the
+accumulator is still in fp32. The result matches
+:code:`torch.nn.functional.gelu(a @ b + bias)` to floating point tolerance,
+in a single kernel launch.
+
+The pattern generalizes. Any elementwise epilogue (bias + SiLU, a GeGLU half,
+a quantization scale and zero-point) slots into the same place between the
+K-loop and the store.
+"""
+
+# %%
+# Setup
+# -----
+#
+# We need PyTorch for the reference and Triton for the kernel.
+
+import torch
+
+import triton
+import triton.language as tl
+
+# %%
+# The autotune configuration
+# --------------------------
+#
+# A compact config list, mirroring the larger one in tutorial 03 and trimmed
+# here for readability. The autotuner re-runs the search whenever one of the
+# keys (M, N, K) changes, so each problem shape gets its own best config.
+
+
+def get_autotune_config():
+    return [
+        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
+                      num_warps=8),
+        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
+                      num_warps=4),
+        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
+                      num_warps=4),
+        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
+                      num_warps=4),
+    ]
+
+
+# %%
+# The kernel
+# ----------
+#
+# The GEMM body is the grouped, tiled loop from tutorial 03. The only new
+# code lives in the epilogue, marked below.
+
+
+@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
+@triton.jit
+def fused_matmul_bias_gelu_kernel(
+        a_ptr, b_ptr, bias_ptr, c_ptr,
+        M, N, K,
+        stride_am, stride_ak,
+        stride_bk, stride_bn,
+        stride_cm, stride_cn,
+        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
+        GROUP_SIZE_M: tl.constexpr,
+):
+    """Compute C = gelu(A @ B + bias).
+
+    A has shape (M, K), B has shape (K, N), bias has shape (N,), and C has
+    shape (M, N). The bias is broadcast over the M rows.
+    """
+    # Map this program to the output tile it owns. The grouped ordering
+    # promotes L2 reuse, exactly as in tutorial 03.
+    pid = tl.program_id(axis=0)
+    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
+    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
+    num_pid_in_group = GROUP_SIZE_M * num_pid_n
+    group_id = pid // num_pid_in_group
+    first_pid_m = group_id * GROUP_SIZE_M
+    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
+    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
+    pid_n = (pid % num_pid_in_group) // group_size_m
+
+    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
+    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
+    offs_k = tl.arange(0, BLOCK_SIZE_K)
+    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
+    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
+
+    # Accumulate in fp32 for accuracy. The epilogue below stays in fp32 too.
+    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
+    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
+        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
+        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
+        acc = tl.dot(a, b, acc)
+        a_ptrs += BLOCK_SIZE_K * stride_ak
+        b_ptrs += BLOCK_SIZE_K * stride_bk
+
+    # --- Epilogue fusion (the whole point) ---------------------------------
+    # bias is per output column, shape (N,). Load this tile's slice, broadcast
+    # it across the M rows, add it, then apply GELU, all while acc is still
+    # fp32. Nothing is written to global memory between the GEMM and the
+    # activation, which is the traffic the three-kernel version would pay.
+    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
+    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
+    acc = acc + bias[None, :]
+    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))). tl.math.erf is the
+    # libdevice binding and the constant is 1 / sqrt(2).
+    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865475))
+
+    # Cast once, at the store, to the output dtype (fp16 here, as in 03).
+    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
+    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
+    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
+    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
+
+
+# %%
+# A convenience wrapper
+# ---------------------
+#
+# The wrapper checks the shape constraints, allocates the output, and launches
+# the kernel over a 1D grid of output tiles.
+
+
+def fused_matmul_bias_gelu(a, b, bias):
+    assert a.shape[1] == b.shape[0], 'Incompatible dimensions'
+    assert a.is_contiguous() and b.is_contiguous(), 'A and B must be contiguous'
+    M, K = a.shape
+    K, N = b.shape
+    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
+    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
+    fused_matmul_bias_gelu_kernel[grid](
+        a, b, bias, c,
+        M, N, K,
+        a.stride(0), a.stride(1),
+        b.stride(0), b.stride(1),
+        c.stride(0), c.stride(1),
+    )
+    return c
+
+
+# %%
+# Verification
+# ------------
+#
+# Compare against the unfused PyTorch reference. The fused kernel keeps the
+# epilogue in fp32 and casts once at the store, so it matches the reference to
+# floating point tolerance rather than bit for bit.
+
+torch.manual_seed(0)
+a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
+b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
+bias = torch.randn((512, ), device='cuda', dtype=torch.float16)
+
+c_fused = fused_matmul_bias_gelu(a, b, bias)
+c_ref = torch.nn.functional.gelu(torch.matmul(a, b) + bias)
+
+max_abs_err = (c_fused.float() - c_ref.float()).abs().max().item()
+print(f'max absolute error vs torch reference: {max_abs_err:.3e}')
+assert torch.allclose(c_fused, c_ref, atol=1e-2, rtol=1e-2), 'fused kernel diverged from reference'
+print('OK: fused matmul + bias + gelu matches the unfused reference')
```

### Gallery registration (conditional)

Triton renders the tutorials with sphinx-gallery, which commonly auto-discovers
every `.py` file in `python/tutorials/`. If that is how the gallery is wired,
the new file needs no registration at all. If instead `python/tutorials/index.rst`
carries an explicit `toctree`, one line is added there. The snippet below is
illustrative and would be confirmed against the repo before any submission:

```diff
--- a/python/tutorials/index.rst
+++ b/python/tutorials/index.rst
@@
    09-persistent-matmul
    10-block-scaled-matmul
+   11-fused-matmul-bias-gelu
```

## How this maps back to PolyKernel

| Triton tutorial element | PolyKernel counterpart |
|---|---|
| `@triton.autotune(configs=..., key=['M','N','K'])` | The bounded `ConfigSpace` grid plus the (GPU, op, shape) tuning cache |
| `triton.Config({'BLOCK_SIZE_M': ...}, num_stages=, num_warps=)` | The autotuner axes `block_m/block_n/block_k`, `shared_memory_stages`, `num_warps` |
| Epilogue fusion (bias + GELU before the store) | The `--fuse-matmul-bias-gelu` pass and `fused_matmul_bias_gelu.cu` |
| `tl.dot(a, b, acc)` fp32 accumulate | The tiled GEMM's fp32 accumulator in the portable CUDA/HIP template |
| Tolerance check vs a PyTorch reference | The NumPy + `ml_dtypes.bfloat16` golden (cosine, max relative error, PCC) |

The correspondence is the point: the tutorial is a faithful, minimal restatement
of the fusion PolyKernel demonstrates, in the upstream idiom most reviewers
would recognize.

## Local validation status

- **Done here**: the example file passes `python3 -m py_compile` (syntax valid).
  Its Triton API usage (`@triton.autotune`, `triton.Config`, `tl.dot` with an
  fp32 accumulator, `tl.math.erf`, masked `tl.load`/`tl.store`, a 1D tile grid)
  mirrors the patterns in the existing `03-matrix-multiplication.py` tutorial.
- **Not done here, by design**: running it. The verification block needs a CUDA
  or HIP GPU and an installed Triton, neither of which this environment has
  (PolyKernel builds CUDA kernels and has runtime-validated them on the remote
  RTX 6000 Ada pod, but Triton is not in the nix dev shell, so this Triton
  tutorial has not been executed). Running the tutorial on a GPU box is the
  first step at submission time.

## Submission status

**Not submitted.** This is the honest state, for two independent reasons:

1. **No credentials, no budget.** This environment cannot fork, push, or open a
   pull request, and the task forbids incurring cloud spend. Preparing the
   contribution is the whole scope here.
2. **It is non-blocking by design.** The plan lists this as "Optional upstream PR
   (NEVER blocking)," and the MVP success criteria do not reference it. Whether
   or not it is ever submitted, Waves 1 through 7 stand on their own.

A candid note on likely reception, so the status is not read as mere inertia:
Triton's `CONTRIBUTING.md` states that design changes which fix no known
functional or performance issue are "automatically considered controversial,"
and that performance changes may be rejected if maintainers judge the
usefulness-to-complexity trade-off unfavorable. A new tutorial sits in a softer
category than a library change, but a fair maintainer question is "why not fold
bias-plus-GELU into tutorial 03's existing activation hook?" The honest answers
are that a standalone, fully worked feed-forward epilogue reads better as its
own lesson, and that GELU (erf-based, the default transformer nonlinearity) is a
more representative example than `leaky_relu`. If a maintainer preferred an
extension of tutorial 03 instead, the content here converts to that with little
effort. None of this changes the MVP outcome.

## How the owner would actually submit it

For whoever owns the credentials, the steps are ordinary:

```bash
# 1. Fork triton-lang/triton on GitHub, then clone the fork.
git clone https://github.com/<owner>/triton.git
cd triton

# 2. Add the new tutorial (copy the file from the diff above).
$EDITOR python/tutorials/11-fused-matmul-bias-gelu.py

# 3. Run it on a GPU box to confirm correctness vs the torch reference.
pip install -r python/tutorials/requirements.txt
python python/tutorials/11-fused-matmul-bias-gelu.py
# expect: "OK: fused matmul + bias + gelu matches the unfused reference"

# 4. Branch, commit, push, and open the PR against triton-lang/triton:main.
git checkout -b tutorial-fused-matmul-bias-gelu
git add python/tutorials/11-fused-matmul-bias-gelu.py
git commit -m "tutorial: fused matmul + bias + GELU epilogue"
git push origin tutorial-fused-matmul-bias-gelu
```

If a decision is made not to pursue it, the correct state is exactly what this
file records: **deferred**, with the MVP complete regardless.

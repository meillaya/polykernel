# PolyKernel Dataflow Backend — Cerebras-Style Simulator

> **Read this first.** The dataflow backend is a self-contained C++ **functional / cycle
> SIMULATOR** of a Cerebras-style dataflow fabric. It is **NOT** real CSL, **NOT** a
> `cslc` compiler, and **NOT** Cerebras hardware. It does not link against, depend on, or
> shell out to the Cerebras SDK or simfabric. It *reproduces the concepts* as a functional
> model so the MLP/matmul fragment can be mapped, executed, and correctness-validated
> independently of any GPU. The real-SDK analogs **modelled** (not integrated) are
> `cslc --out-routes` and `calculate_cycles`.
>
> **Terminology is deliberate and correct:** a PE's compute unit is a **CE** with
> **FMACs** (not "FMU"/"PMU"); routing color is configured with **`@set_color_config`**
> rx/tx (not "set_color"); the fabric move DSDs are **`fabin_dsd`** / **`fabout_dsd`**
> (not "fdata"/"fcast"/"fmove"); compute is triggered by data arrival via
> **`@activate`/`@block`/`@unblock`** rendezvous (there are no `@compute`/`@data`
> decorators — `compute` is an exported host-callable convention).

## Why a simulator

The GPU backends (CUDA/HIP) need hardware; the dataflow simulator needs none. It
**functionally executes** the scheduled tile program, so its numerical output is compared
against the **same** NumPy/`ml_dtypes` golden as the GPU kernels (contract C: cosine ≥
0.999, max rel err ≤ 1e-2, PCC ≥ 0.99; bf16 round / fp32 accumulate). That makes the
dataflow path correctness-validated **independently of any GPU**.

## The processing element (PE)

Each PE models three things (`include/PolyKernel/Dataflow/Pe.h`, `lib/DataflowSim/Pe.cpp`):

- **CE (compute engine) with FMACs** — the datapath that runs the tile compute. The
  simulator executes an `@fmacs`-style outer-product as a **real fp32 fused-multiply-add
  on bf16-rounded inputs**, matching the golden's rounding contract (not literal fp16,
  which would spuriously fail the golden).
- **5-port router** — RAMP (local) + East/West/North/South. A wavelet advances **one hop
  per cycle**; the fabric is lossless with backpressure.
- **48 KB SRAM** — 8 banks × 6 KB (2 reads + 1 write / cycle). This is where the
  output-stationary tile and the ping-pong double-buffers reside.

## Wavelets, colors, routing

- **Wavelet** (`Wavelet.h`) — 32 bits: 16 bits data + 16 bits control/index. The unit of
  fabric traffic; `messages sent` counts wavelets injected via `fabout_dsd`.
- **Color** (`Color.h`) — a 5-bit tag; **24 routable colors** (IDs 0–23) act as virtual
  channels over a shared physical link. Congestion on one color does **not** block another
  (color isolation, gtest-verified).
- **Router** (`Router.h`) — routes a wavelet one hop per cycle in E/W/N/S/RAMP. A color is
  configured **`@set_color_config`** with **rx** (one receive direction) and **tx** (a
  *set* of transmit directions = multicast). The fabric move descriptors are
  **`fabin_dsd`** (fabric-in) / **`fabout_dsd`** (fabric-out).

## Task model + scheduling (dataflow-triggered compute)

`Task.h` / `Scheduler.h` / `TaskPicker.h` model dataflow = **compute triggered by data
arrival**:

- **data task** — wavelet-triggered, bound to a color / `data_task_id`; fires when its
  input wavelet arrives;
- **local task** — self-triggered via `@activate`;
- **control task** — coordination.

The **rendezvous** is `@activate` / `@block` / `@unblock`: a compute task runs only when it
is **activated AND unblocked** (e.g. fires only after *both* inputs arrive). The task picker
selects among eligible tasks. The negative test (`reports/w7_task_blocked_neg.log`)
activates a task but leaves it blocked and asserts it does **not** fire prematurely.

## The SUMMA matmul mapping

`Summa.cpp` / `LowerToDataflow.cpp` map `polykernel.matmul` / fused-matmul onto the
**SUMMA** schedule on a P×P grid — the same algorithm as the official Cerebras
`gemm-collectives_2d` example, which matches the spec's "A east / B south / local C
accumulate / row-col reduction" exactly:

1. At step *i*, column *i* broadcasts its **A** tile **EAST** along each row;
2. row *i* broadcasts its **B** tile **SOUTH** along each column;
3. each PE accumulates `C_tile += A_panel · B_panel` (**output-stationary C** in local
   SRAM) via an `@fmacs`-style outer-product;
4. **2 colors per axis** ping-pong double-buffer A/B; the dataflow rendezvous
   (`x_done → activate(compute)`, `y_done → unblock(compute)`) triggers each panel compute.

`--lower-to-dataflow` performs this mapping. **Non-matmul ops** (rmsnorm, gelu/silu,
softmax, add, bias) map to **local PE compute** — each PE applies the op to its resident
tile in local SRAM with **no fabric wavelets**; only matmul / fused-matmul uses the SUMMA
fabric schedule. The simulator-executed matmul matches the golden
(`reports/w7_summa.log`); breaking the rendezvous (firing compute on one input) makes the
output wrong and fail the golden (`reports/w7_summa_neg.log`).

## Metrics

`Metrics.cpp` reports the seven dataflow metrics (validated against **hand-computed** 4×4
values in `metrics_test.cpp`):

| Metric | Meaning |
|---|---|
| grid utilization | active PEs / total PEs |
| SRAM pressure | resident tiles + double-buffers vs 48 KB |
| messages sent | total wavelets (Σ `fabout_dsd` extents) |
| avg hop distance | 1 cycle/hop; broadcast width P = P−1 hops |
| comm bottleneck | active / delayed / backpressure / idle |
| critical-path cycles | broadcast fill (P−1) + pipelined compute (steps) |
| fusion traffic reduction | wavelet count unfused vs fused |

**Fusion savings** arise because a fused intermediate (e.g. the rmsnorm output feeding the
matmul) stays in **local SRAM** and never traverses the fabric — **zero** wavelets for it.

### Representative shape note

The MLP matmul (M=2048, N=11008, K=4096) is non-square and far too large to simulate
cycle-accurately, so the dataflow metrics are reported for a **representative square SUMMA
matmul** (default 32×32×32 on a 4×4 grid). The **fusion traffic reduction** uses the MLP's
*actual* eliminated intermediate (read from the fusion attributes `polykernel-opt` attaches,
cross-referenced against `reports/mlp_traffic.json`). For the MLP block the simulator
reports (`reports/dataflow_metrics.json`): utilization 100%, SRAM pressure 2.60%, 2,048
wavelets, avg hop 3.0, critical path 7 cycles, bottleneck `active`, and a fusion traffic
reduction of 99.99% (the eliminated `tensor<1x2048x4096xbf16>` intermediate crosses the
fabric as zero wavelets).

## HTML visualizer

`polykernel-report --backend=dataflow --viz` renders `reports/dataflow_report.html`
(`dataflow_viz.py` + `lib/DataflowSim/viz_template.html`): the PE grid, the A-east/B-south
tile routes, the per-PE SRAM-pressure heatmap, message traffic, fusion savings, and the
critical path. It is a **single self-contained** HTML file (inline CSS/JS + an embedded
`<script id="viz-data" type="application/json">` blob) — no external dependencies, no
network fetch, no framework; it renders fully offline. The benchmark dashboard
(`reports/benchmark_report.html`) **links** to this visualizer; it does not regenerate it.

## Files

```
include/PolyKernel/Dataflow/   Pe.h Wavelet.h Color.h Router.h Grid.h
                               Task.h TaskPicker.h Scheduler.h Summa.h Metrics.h
lib/DataflowSim/               Pe.cpp Wavelet.cpp Color.cpp Router.cpp Grid.cpp
                               Task.cpp Scheduler.cpp Summa.cpp Metrics.cpp
                               LowerToDataflow.cpp viz_template.html
tools/polykernel-report/       dataflow_report.py (metrics) · dataflow_viz.py (HTML)
reports/                       dataflow_metrics.json · dataflow_report.html
```

## See also

- [`architecture.md`](architecture.md) — where the simulator sits in the system.
- [`hip_backend.md`](hip_backend.md) — the GPU backend computing the same matmul.
- [`performance_model.md`](performance_model.md) — the GPU roofline projections.

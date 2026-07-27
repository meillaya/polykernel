//===- Summa.h - SUMMA matmul mapping on the dataflow fabric ---*- C++ -*-===//
//
// PolyKernel SUMMA matmul mapping + lowering (Todo 37 / Wave 7).
//
// THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
// hardware). It builds on the Todo 35 dataflow core (Color/Wavelet/Router/Pe/
// Grid) and the Todo 36 task model + scheduler (Task/Scheduler/TaskPicker): the
// SUMMA schedule is a tile program that the simulator FUNCTIONALLY EXECUTES - it
// routes wavelets across the Grid, fires the dataflow rendezvous, and runs the
// CE FMAC outer-product accumulate - so its numerical output can be validated
// against a golden matmul (contract C: cosine >= 0.999, max_rel_err <= 1e-2,
// pcc >= 0.99).
//
// SUMMA (SUMMA = "SUMMA" GEMM, after the Cerebras csl-examples gemm-collectives
// 2-D collective). On a P x P grid:
//   - output-stationary C: PE (x, y) owns the C tile rows [y*T, (y+1)*T), cols
//     [x*T, (x+1)*T) and accumulates it in its LOCAL SRAM (C never moves),
//   - step s broadcasts the step's A panel EAST along every row (rx = West,
//     tx = East) and the step's B panel SOUTH along every column (rx = North,
//     tx = South); each PE accumulates C_tile += A_panel * B_panel via the CE
//     FMAC (an @fmacs-style outer product),
//   - 2 colors per axis (A0/A1, B0/B1) give ping-pong double-buffering: even
//     steps use color [0], odd steps use color [1],
//   - the compute rendezvous fires the panel accumulate only after BOTH panels
//     have arrived: x_done (A panel complete) -> @activate(compute), y_done
//     (B panel complete) -> @unblock(compute); the compute task is a data task
//     bound to TWO rendezvous colors that fires only after BOTH tokens arrive.
//
// Terminology guardrail (F4): the compute unit is the CE with FMACs (no FMU/
// PMU); routing is `@set_color_config` rx/tx with `fabin_dsd`/`fabout_dsd`; the
// rendezvous is `@activate`/`@block`/`@unblock` (there are NO @compute/@data
// decorators). This is a simulator, not a cslc compiler and not Cerebras HW.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DATAFLOW_SUMMA_H
#define POLYKERNEL_DATAFLOW_SUMMA_H

#include "PolyKernel/Dataflow/Color.h"
#include "PolyKernel/Dataflow/Grid.h"

#include <cstdint>
#include <vector>

namespace polykernel::dataflow {

//===----------------------------------------------------------------------===//
// SummaPlan: a lowered SUMMA matmul schedule (the simulator lowering of a
// polykernel.matmul / fused-matmul op + shape onto the Grid).
//===----------------------------------------------------------------------===//

struct SummaPlan {
  // Matmul dimensions: C[m, n] = A[m, k] * B[k, n] (row-major operands).
  int m = 0;
  int n = 0;
  int k = 0;

  // P x P grid (m = gridP * tileM, n = gridP * tileN).
  int gridP = 1;
  // Output tile per PE (square: tileM == tileN == tile). The PE owns a tile x
  // tile block of C (output-stationary, accumulated in local SRAM).
  int tileM = 1;
  int tileN = 1;
  // K-slab width per SUMMA step (k = steps * tileK).
  int tileK = 1;
  // Number of SUMMA steps (k / tileK). Step s broadcasts K-slab [s*tileK,
  // (s+1)*tileK).
  int steps = 1;

  // Ping-pong colors (2 per axis): even steps use [0], odd steps use [1].
  Color aColor[2]; // A broadcast (EAST) colors.
  Color bColor[2]; // B broadcast (SOUTH) colors.
  // Rendezvous colors: x_done (A panel complete) / y_done (B panel complete).
  Color xDone; // @activate(compute) token color.
  Color yDone; // @unblock(compute) token color.

  // The source polykernel op this plan lowers (closed op set: matmul or a fused
  // matmul). Recorded for traceability; the simulator lowering maps the op +
  // shape onto this schedule.
  bool fused = false;
  const char *opName = "polykernel.matmul";
};

//===----------------------------------------------------------------------===//
// Simulator lowering: map a matmul (op + shape) onto the SUMMA schedule.
// (Defined in LowerToDataflow.cpp - the "--lower-to-dataflow" simulator path.)
//===----------------------------------------------------------------------===//

// Lower polykernel.matmul (A:m x k, B:k x n -> C:m x n) onto a SUMMA plan on a
// gridP x gridP grid. Chooses square output tiles (tileM == tileN) and a tileK
// so that m, n, k tile exactly (asserts divisibility). tileK == 0 picks a
// default K-slab. The op set is CLOSED: this maps an EXISTING matmul op onto the
// schedule (no new dialect ops/attributes).
SummaPlan LowerMatmulToSumma(int m, int n, int k, int gridP = 4, int tileK = 0);

// Lower a fused matmul (polykernel.fused_rmsnorm_matmul / fused_matmul_bias_gelu)
// onto the SAME SUMMA schedule (the matmul part maps identically); records the
// fused source op name and fused = true.
SummaPlan LowerFusedMatmulToSumma(int m, int n, int k,
                                  const char *fusedOpName =
                                      "polykernel.fused_rmsnorm_matmul",
                                  int gridP = 4, int tileK = 0);

//===----------------------------------------------------------------------===//
// Route setup + functional execution.
//===----------------------------------------------------------------------===//

// Configure the SUMMA broadcast routes on `grid` (which must be gridP x gridP):
// A colors forward West -> East (| Ramp for the local tap; the east edge taps
// only), B colors forward North -> South (| Ramp; the south edge taps only).
// This is the per-PE `@set_color_config` for the A-east / B-south broadcast.
void ConfigureSummaRoutes(Grid &grid, const SummaPlan &plan);

// Result of a SUMMA execution.
struct SummaResult {
  std::vector<float> c;        // m x n row-major output (gathered C tiles).
  uint64_t cycles = 0;         // Grid cycles elapsed (Step calls).
  uint64_t panelComputes = 0;  // rendezvous-triggered panel accumulates fired.
  uint64_t aWavelets = 0;      // A wavelets routed (EAST broadcast).
  uint64_t bWavelets = 0;      // B wavelets routed (SOUTH broadcast).
  uint64_t droppedOffGrid = 0; // wavelets dropped at the mesh edge (0 expected).
};

// Functionally execute the SUMMA tile program for C = A * B on the Grid: route
// the A panels east and B panels south (wavelets), run the per-PE dataflow
// rendezvous (compute fires only after BOTH panels arrive), and accumulate the
// output-stationary C tiles via the CE FMAC. A is m x k, B is k x n (row-major,
// fp32); the CE FMAC carries the multiply-add (fp16 datapath). Returns the
// gathered C (m x n) plus execution metadata.
//
// breakRendezvous (NEGATIVE hook): fire the per-PE compute on the A panel ALONE
// (the compute task is bound to ONLY the x_done color; the B broadcast is
// skipped), so C is accumulated against an empty B panel -> WRONG output that
// fails the golden contract. Proves the correctness test catches dataflow bugs.
SummaResult RunSumma(const SummaPlan &plan, const std::vector<float> &a,
                     const std::vector<float> &b, bool breakRendezvous = false);

} // namespace polykernel::dataflow

#endif // POLYKERNEL_DATAFLOW_SUMMA_H

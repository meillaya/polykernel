//===- LowerToDataflow.cpp - Lower polykernel matmul to SUMMA ---*- C++ -*-===//
//
// PolyKernel SUMMA matmul mapping + lowering (Todo 37 / Wave 7). See Summa.h.
//
// This is the simulator's "--lower-to-dataflow" path: it maps an EXISTING
// polykernel matmul op (polykernel.matmul, or a fused matmul such as
// polykernel.fused_rmsnorm_matmul / polykernel.fused_matmul_bias_gelu) plus its
// shape onto the SUMMA tile schedule (a P x P grid, square output tiles, a
// K-slab per step, ping-pong colors, rendezvous colors). The op set is CLOSED -
// no new dialect ops/attributes are introduced; the lowering only chooses the
// SUMMA schedule parameters that RunSumma then functionally executes.
//
// THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
// hardware).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Summa.h"

#include <cassert>

namespace polykernel::dataflow {

namespace {

// Choose a K-slab width that tiles `k` exactly, preferring `pref` (default 8).
int ChooseTileK(int k, int pref) {
  if (pref <= 0)
    pref = 8;
  int t = pref < k ? pref : k;
  while (t > 1 && (k % t) != 0)
    --t;
  return t; // always >= 1 (t == 1 divides everything).
}

// Common schedule builder shared by the plain and fused matmul lowerings.
SummaPlan BuildPlan(int m, int n, int k, int gridP, int tileK, bool fused,
                    const char *opName) {
  assert(m > 0 && n > 0 && k > 0 && "matmul dims must be positive");
  assert(gridP > 0 && "grid dimension must be positive");
  assert((m % gridP) == 0 && (n % gridP) == 0 &&
         "matmul M/N must tile evenly across the P x P grid");
  const int tileM = m / gridP;
  const int tileN = n / gridP;
  assert(tileM == tileN && "SUMMA output tiles are square (require M == N)");

  SummaPlan plan;
  plan.m = m;
  plan.n = n;
  plan.k = k;
  plan.gridP = gridP;
  plan.tileM = tileM;
  plan.tileN = tileN;
  plan.tileK = ChooseTileK(k, tileK);
  plan.steps = k / plan.tileK;

  // Ping-pong colors (2 per axis) + the x_done/y_done rendezvous colors. All are
  // routable fabric colors (IDs 0-23); the data colors broadcast, the rendezvous
  // colors are local PE tokens (not forwarded).
  plan.aColor[0] = *Color::Routable(0);
  plan.aColor[1] = *Color::Routable(1);
  plan.bColor[0] = *Color::Routable(2);
  plan.bColor[1] = *Color::Routable(3);
  plan.xDone = *Color::Routable(4);
  plan.yDone = *Color::Routable(5);

  plan.fused = fused;
  plan.opName = opName;
  return plan;
}

} // namespace

SummaPlan LowerMatmulToSumma(int m, int n, int k, int gridP, int tileK) {
  return BuildPlan(m, n, k, gridP, tileK, /*fused=*/false, "polykernel.matmul");
}

SummaPlan LowerFusedMatmulToSumma(int m, int n, int k, const char *fusedOpName,
                                  int gridP, int tileK) {
  return BuildPlan(m, n, k, gridP, tileK, /*fused=*/true, fusedOpName);
}

} // namespace polykernel::dataflow

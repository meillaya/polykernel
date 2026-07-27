//===- summa_test.cpp - SUMMA matmul mapping (routing + rendezvous) -*- C++ -*-=//
//
// PolyKernel SUMMA matmul mapping + lowering (Todo 37 / Wave 7). Suite `summa`
// so `ctest -R 'summa|dataflow_correctness'` discovers these.
//
// Asserts the SUMMA schedule on the Todo 35 Grid + Todo 36 Scheduler:
//   - the simulator lowering maps polykernel.matmul (and a fused matmul) onto a
//     P x P SUMMA plan (square output tiles, a K-slab per step, ping-pong colors,
//     rendezvous colors),
//   - the routing pattern is A broadcast EAST (rx = West, tx = East) and B
//     broadcast SOUTH (rx = North, tx = South) - wavelets physically move +x /
//     -y, with the mesh edges tapping only (no off-grid forwarding),
//   - 2 colors per axis (A0/A1, B0/B1) for ping-pong double-buffering,
//   - the compute rendezvous: the compute task (a data task bound to the x_done
//     and y_done colors) fires only after BOTH panels arrive - x_done ->
//     @activate(compute), y_done -> @unblock(compute); it does NOT fire on one.
//
// Terminology (F4): CE + FMAC (no FMU/PMU); @set_color_config rx/tx; the
// rendezvous is @activate/@block/@unblock (no @compute/@data decorators). This
// is a GPU-free functional SIMULATOR, not real CSL and not Cerebras hardware.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Grid.h"
#include "PolyKernel/Dataflow/Scheduler.h"
#include "PolyKernel/Dataflow/Summa.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

namespace {

using polykernel::dataflow::Color;
using polykernel::dataflow::ConfigureSummaRoutes;
using polykernel::dataflow::Dir;
using polykernel::dataflow::Grid;
using polykernel::dataflow::HasDir;
using polykernel::dataflow::LowerFusedMatmulToSumma;
using polykernel::dataflow::LowerMatmulToSumma;
using polykernel::dataflow::MakeWavelet;
using polykernel::dataflow::Router;
using polykernel::dataflow::RunSumma;
using polykernel::dataflow::Scheduler;
using polykernel::dataflow::SummaPlan;
using polykernel::dataflow::Task;
using polykernel::dataflow::TaskKind;

std::vector<float> RandPM1(int n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed ? seed : 0x12345678u;
  for (int i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = ((s >> 16) & 1u) ? 1.0f : -1.0f;
  }
  return v;
}

std::vector<float> RefMatmul(const std::vector<float> &a,
                             const std::vector<float> &b, int m, int k, int n) {
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.f);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      float s = 0.f;
      for (int kk = 0; kk < k; ++kk)
        s += a[static_cast<std::size_t>(i) * k + kk] *
             b[static_cast<std::size_t>(kk) * n + j];
      c[static_cast<std::size_t>(i) * n + j] = s;
    }
  return c;
}

// The simulator lowering maps a matmul onto a P x P SUMMA schedule.
TEST(summa, LoweringMapsMatmulOntoSummaSchedule) {
  const SummaPlan plan = LowerMatmulToSumma(32, 32, 32, /*gridP=*/4);
  EXPECT_EQ(plan.gridP, 4);
  EXPECT_EQ(plan.m, 32);
  EXPECT_EQ(plan.n, 32);
  EXPECT_EQ(plan.k, 32);
  // Square output tiles tiling M/N across the grid.
  EXPECT_EQ(plan.tileM, 8);
  EXPECT_EQ(plan.tileN, 8);
  EXPECT_EQ(plan.m, plan.gridP * plan.tileM);
  EXPECT_EQ(plan.n, plan.gridP * plan.tileN);
  // The K-slab tiles K exactly into an integer number of SUMMA steps.
  EXPECT_GT(plan.tileK, 0);
  EXPECT_EQ(plan.k % plan.tileK, 0);
  EXPECT_EQ(plan.steps, plan.k / plan.tileK);
  EXPECT_EQ(plan.steps * plan.tileK, plan.k);
  // Source op recorded (closed op set: this lowers an existing matmul op).
  EXPECT_FALSE(plan.fused);
  EXPECT_STREQ(plan.opName, "polykernel.matmul");
}

// A fused matmul maps onto the SAME SUMMA schedule, recording the fused op.
TEST(summa, LoweringMapsFusedMatmulOntoSameSchedule) {
  const SummaPlan plain = LowerMatmulToSumma(16, 16, 16, 4);
  const SummaPlan fused = LowerFusedMatmulToSumma(
      16, 16, 16, "polykernel.fused_rmsnorm_matmul", 4);
  EXPECT_TRUE(fused.fused);
  EXPECT_STREQ(fused.opName, "polykernel.fused_rmsnorm_matmul");
  // The matmul part lowers identically (same grid / tiles / steps / colors).
  EXPECT_EQ(fused.gridP, plain.gridP);
  EXPECT_EQ(fused.tileM, plain.tileM);
  EXPECT_EQ(fused.tileK, plain.tileK);
  EXPECT_EQ(fused.steps, plain.steps);
  EXPECT_EQ(fused.aColor[0], plain.aColor[0]);
  EXPECT_EQ(fused.bColor[0], plain.bColor[0]);
}

// 2 colors per axis (ping-pong double-buffering) + distinct rendezvous colors.
TEST(summa, PingPongTwoColorsPerAxis) {
  const SummaPlan plan = LowerMatmulToSumma(16, 16, 16, 4);
  // Two DISTINCT colors per axis.
  EXPECT_FALSE(plan.aColor[0] == plan.aColor[1]);
  EXPECT_FALSE(plan.bColor[0] == plan.bColor[1]);
  // All six colors routable and pairwise distinct.
  const Color all[6] = {plan.aColor[0], plan.aColor[1], plan.bColor[0],
                        plan.bColor[1], plan.xDone,     plan.yDone};
  for (const Color c : all)
    EXPECT_TRUE(c.IsRoutable());
  for (int i = 0; i < 6; ++i)
    for (int j = i + 1; j < 6; ++j)
      EXPECT_FALSE(all[i] == all[j]) << "colors " << i << "," << j;

  // Both A colors and both B colors get a broadcast route on an interior PE.
  Grid g(plan.gridP, plan.gridP);
  ConfigureSummaRoutes(g, plan);
  const Router &r = g.At(1, 1).GetRouter();
  for (const Color c : {plan.aColor[0], plan.aColor[1]}) {
    const auto cfg = r.GetColorConfig(c);
    ASSERT_TRUE(cfg.has_value());
    EXPECT_EQ(cfg->rx, Dir::West);
    EXPECT_TRUE(HasDir(cfg->tx, Dir::East)); // A broadcasts EAST.
  }
  for (const Color c : {plan.bColor[0], plan.bColor[1]}) {
    const auto cfg = r.GetColorConfig(c);
    ASSERT_TRUE(cfg.has_value());
    EXPECT_EQ(cfg->rx, Dir::North);
    EXPECT_TRUE(HasDir(cfg->tx, Dir::South)); // B broadcasts SOUTH.
  }
}

// The routing pattern: A wavelets move EAST (+x), B wavelets move SOUTH (-y).
TEST(summa, RoutingPatternAEastBSouth) {
  const SummaPlan plan = LowerMatmulToSumma(16, 16, 16, 4);
  Grid g(plan.gridP, plan.gridP);
  ConfigureSummaRoutes(g, plan);

  // Static route direction on an interior PE: A rx=West/tx=East, B rx=North/
  // tx=South.
  const auto cfgA = g.At(1, 1).GetRouter().GetColorConfig(plan.aColor[0]);
  ASSERT_TRUE(cfgA.has_value());
  EXPECT_EQ(cfgA->rx, Dir::West);
  EXPECT_TRUE(HasDir(cfgA->tx, Dir::East));
  const auto cfgB = g.At(1, 1).GetRouter().GetColorConfig(plan.bColor[0]);
  ASSERT_TRUE(cfgB.has_value());
  EXPECT_EQ(cfgB->rx, Dir::North);
  EXPECT_TRUE(HasDir(cfgB->tx, Dir::South));

  // The mesh edges tap only (no off-grid forwarding -> no drops).
  EXPECT_FALSE(HasDir(g.At(3, 1).GetRouter().GetColorConfig(plan.aColor[0])->tx,
                      Dir::East)); // east edge: no East.
  EXPECT_FALSE(HasDir(g.At(1, 0).GetRouter().GetColorConfig(plan.bColor[0])->tx,
                      Dir::South)); // south edge: no South.

  // Dynamic: an A wavelet injected eastbound moves EAST (arrives at +x, never at
  // the west neighbour).
  const Color ac = plan.aColor[0];
  ASSERT_TRUE(g.Inject(1, 1, ac, Dir::East, MakeWavelet(0x3C00, 0)));
  g.Step();
  EXPECT_EQ(g.At(2, 1).GetRouter().InputSize(ac), 1u); // east neighbour (+x).
  EXPECT_EQ(g.At(0, 1).GetRouter().InputSize(ac), 0u); // not west.

  // Dynamic: a B wavelet injected southbound moves SOUTH (arrives at -y, never at
  // the north neighbour).
  const Color bc = plan.bColor[0];
  ASSERT_TRUE(g.Inject(1, 2, bc, Dir::South, MakeWavelet(0x3C00, 0)));
  g.Step();
  EXPECT_EQ(g.At(1, 1).GetRouter().InputSize(bc), 1u); // south neighbour (-y).
  EXPECT_EQ(g.At(1, 3).GetRouter().InputSize(bc), 0u); // not north.
}

// The compute rendezvous: the compute task (data task bound to x_done + y_done)
// fires only after BOTH arrive - x_done -> @activate, y_done -> @unblock. It
// does NOT fire on only one.
TEST(summa, RendezvousComputeFiresOnlyAfterBothPanels) {
  const SummaPlan plan = LowerMatmulToSumma(16, 16, 16, 4);
  Router r;
  Scheduler sched(r);

  Task compute(0, TaskKind::Data);
  compute.AddInput(plan.xDone); // x_done (A panel complete).
  compute.AddInput(plan.yDone); // y_done (B panel complete).
  int fires = 0;
  compute.SetHandler([&](Task &) { ++fires; });
  const int id = sched.AddTask(std::move(compute));
  // Starts deactivated + blocked (the SUMMA engine activates on x_done, unblocks
  // on y_done).
  sched.Block(id);

  // Nothing arrived: not runnable.
  EXPECT_EQ(sched.Step(), -1);
  EXPECT_EQ(fires, 0);

  // x_done: A panel complete -> @activate(compute) + x_done token. Still blocked
  // and y_done missing -> NO fire.
  sched.Activate(id);
  ASSERT_TRUE(r.Rx(Dir::Ramp, plan.xDone, MakeWavelet(1, 0)));
  EXPECT_EQ(sched.Pick(), nullptr); // blocked => not runnable.
  EXPECT_EQ(sched.Step(), -1);
  EXPECT_EQ(fires, 0); // did NOT fire on only x_done.

  // y_done: B panel complete -> @unblock(compute) + y_done token. Now activated
  // AND unblocked AND BOTH tokens present -> the compute fires.
  sched.Unblock(id);
  ASSERT_TRUE(r.Rx(Dir::Ramp, plan.yDone, MakeWavelet(1, 0)));
  EXPECT_EQ(sched.Step(), id);
  EXPECT_EQ(fires, 1); // fired only after BOTH arrived.
  EXPECT_EQ(r.InputSize(plan.xDone), 0u); // both tokens consumed.
  EXPECT_EQ(r.InputSize(plan.yDone), 0u);
}

// End-to-end at small scale: the simulator functionally executes the SUMMA tile
// program (routing + rendezvous + FMAC) and matches the reference matmul.
TEST(summa, RunSummaTinyMatchesReference) {
  const int m = 8, n = 8, k = 8;
  const SummaPlan plan = LowerMatmulToSumma(m, n, k, /*gridP=*/4);
  const auto a = RandPM1(m * k, 1);
  const auto b = RandPM1(k * n, 2);
  const auto ref = RefMatmul(a, b, m, k, n);

  const auto res = RunSumma(plan, a, b);
  ASSERT_EQ(res.c.size(), ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i)
    EXPECT_FLOAT_EQ(res.c[i], ref[i]) << "element " << i;

  // The schedule really ran: one rendezvous panel-compute per PE per step, A/B
  // wavelets broadcast, nothing dropped off-grid.
  const int panelSize = plan.tileM * plan.tileK;
  EXPECT_EQ(res.panelComputes,
            static_cast<uint64_t>(plan.gridP) * plan.gridP * plan.steps);
  EXPECT_EQ(res.aWavelets,
            static_cast<uint64_t>(plan.gridP) * panelSize * plan.steps);
  EXPECT_EQ(res.bWavelets,
            static_cast<uint64_t>(plan.gridP) * panelSize * plan.steps);
  EXPECT_EQ(res.droppedOffGrid, 0u);
  EXPECT_GT(res.cycles, 0u);
}

} // namespace

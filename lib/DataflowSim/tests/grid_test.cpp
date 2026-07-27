//===- grid_test.cpp - 64x64 PE mesh + routing tests ---------*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7). Suite `grid` so
// `ctest -R 'router|color|grid'` discovers these.
//
// Asserts: the default 64x64 mesh; each PE = CE + 48KB/8-bank SRAM + 5-port
// router; single-cycle hop (a wavelet advances exactly one hop per cycle);
// multicast across the mesh; the SRAM bank model (8 banks, 2 reads + 1 write /
// cycle); the CE FMAC; and the off-grid negative (routing east from the
// rightmost column is dropped/flagged, never an out-of-bounds crash).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Grid.h"

#include "gtest/gtest.h"

namespace {

using polykernel::dataflow::Ce;
using polykernel::dataflow::Color;
using polykernel::dataflow::Dir;
using polykernel::dataflow::Grid;
using polykernel::dataflow::MakeWavelet;
using polykernel::dataflow::Pe;
using polykernel::dataflow::Router;
using polykernel::dataflow::Sram;

// x-coordinate of the PE on row `y` whose input VC holds `color`, else -1.
int InputX(const Grid &g, int y, Color c) {
  for (int x = 0; x < g.Width(); ++x)
    if (g.At(x, y).GetRouter().InputSize(c) > 0)
      return x;
  return -1;
}

TEST(grid, DefaultMeshIs64x64) {
  Grid g;
  EXPECT_EQ(g.Width(), 64);
  EXPECT_EQ(g.Height(), 64);
  EXPECT_EQ(g.Size(), 64u * 64u); // 4096 PEs.
}

TEST(grid, PeBundlesCeSramRouter) {
  Grid g;
  const Pe &pe = g.At(0, 0);
  // CE: 4 FMACs on a 64-bit datapath.
  EXPECT_EQ(pe.Compute().kFmacCount, 4);
  EXPECT_EQ(pe.Compute().kDatapathBits, 64);
  // SRAM: 48 KB = 8 banks x 6 KB.
  EXPECT_EQ(pe.Memory().kBanks, 8);
  EXPECT_EQ(pe.Memory().kBankBytes, 6 * 1024);
  EXPECT_EQ(pe.Memory().kTotalBytes, 48 * 1024);
  // Router: 5 ports.
  EXPECT_EQ(pe.GetRouter().kNumPorts, 5);
}

TEST(grid, SingleCycleHop) {
  Grid g;
  const Color c = *Color::Routable(0);
  // Build an eastbound route along row 0: each PE forwards color 0 West->East.
  for (int x = 1; x < g.Width(); ++x)
    ASSERT_TRUE(g.SetRoute(x, 0, c, Dir::West, Dir::East));

  const auto w = MakeWavelet(0x0001, 0x00A5);
  ASSERT_TRUE(g.Inject(0, 0, c, Dir::East, w));

  // After exactly k cycles the wavelet is at column k: one hop per cycle.
  for (int k = 1; k <= 8; ++k) {
    g.Step();
    EXPECT_EQ(g.Cycle(), static_cast<uint64_t>(k));
    EXPECT_EQ(InputX(g, 0, c), k) << "after " << k << " cycles";
  }
}

TEST(grid, HopIsExactlyOnePerCycle) {
  Grid g;
  const Color c = *Color::Routable(2);
  for (int x = 1; x < g.Width(); ++x)
    ASSERT_TRUE(g.SetRoute(x, 3, c, Dir::West, Dir::East));
  ASSERT_TRUE(g.Inject(0, 3, c, Dir::East, MakeWavelet(0x0002, 0x0001)));

  g.Step();
  EXPECT_EQ(InputX(g, 3, c), 1); // moved...
  // ...but ONLY one hop: columns 2+ are still empty this cycle.
  EXPECT_EQ(g.At(2, 3).GetRouter().InputSize(c), 0u);
  EXPECT_EQ(g.At(3, 3).GetRouter().InputSize(c), 0u);
}

TEST(grid, MulticastAcrossMesh) {
  Grid g;
  const Color c = *Color::Routable(1);
  // One wavelet, two directions (East + North) = multicast fan-out on the mesh.
  ASSERT_TRUE(g.Inject(5, 5, c, Dir::East | Dir::North, MakeWavelet(0x0003, 0x0)));
  g.Step();
  EXPECT_EQ(g.At(6, 5).GetRouter().InputSize(c), 1u); // east neighbour.
  EXPECT_EQ(g.At(5, 6).GetRouter().InputSize(c), 1u); // north neighbour.
  EXPECT_EQ(g.At(4, 5).GetRouter().InputSize(c), 0u); // west: not targeted.
  EXPECT_EQ(g.At(5, 4).GetRouter().InputSize(c), 0u); // south: not targeted.
}

// The pinned negative: routing a wavelet EAST from the rightmost column (x=63)
// leaves the mesh -> it is dropped and flagged, with NO out-of-bounds access.
TEST(grid, OffGridEastIsDroppedNotCrash) {
  Grid g;
  const Color c = *Color::Routable(0);
  const auto w = MakeWavelet(0xDEAD, 0x0001);
  ASSERT_TRUE(g.Inject(63, 0, c, Dir::East, w)); // rightmost column, heading east.
  ASSERT_EQ(g.DroppedOffGrid(), 0u);

  g.Step(); // hops off-grid -> dropped + flagged, grid stays intact.

  EXPECT_EQ(g.DroppedOffGrid(), 1u);
  EXPECT_EQ(g.Size(), 64u * 64u);            // mesh undamaged.
  EXPECT_EQ(g.At(63, 0).GetRouter().OutputSize(Dir::East, c), 0u); // wavelet gone.

  // Stepping again is safe and drops nothing further.
  g.Step();
  EXPECT_EQ(g.DroppedOffGrid(), 1u);
}

// Every off-mesh direction from a corner is dropped safely (W/S from (0,0),
// E/N from (63,63)) - no crash on any boundary.
TEST(grid, AllBoundaryDirectionsAreSafe) {
  Grid g;
  const Color c = *Color::Routable(3);
  const auto w = MakeWavelet(0x0001, 0x0001);
  ASSERT_TRUE(g.Inject(0, 0, c, Dir::West, w));
  ASSERT_TRUE(g.Inject(0, 0, c, Dir::South, w));
  ASSERT_TRUE(g.Inject(63, 63, c, Dir::East, w));
  ASSERT_TRUE(g.Inject(63, 63, c, Dir::North, w));
  g.Step();
  EXPECT_EQ(g.DroppedOffGrid(), 4u);
  EXPECT_EQ(g.Size(), 64u * 64u);
}

// A wavelet routed across the whole row eventually drops off the far edge.
TEST(grid, TraverseThenDropOffEdge) {
  Grid g;
  const Color c = *Color::Routable(4);
  for (int x = 1; x < g.Width(); ++x) // forward East across the full row, incl. 63.
    ASSERT_TRUE(g.SetRoute(x, 0, c, Dir::West, Dir::East));
  ASSERT_TRUE(g.Inject(0, 0, c, Dir::East, MakeWavelet(0x0009, 0x0)));

  // 63 hops reach column 63; the 64th step forwards it off-grid -> dropped.
  for (int i = 0; i < 63; ++i)
    g.Step();
  EXPECT_EQ(InputX(g, 0, c), 63);
  EXPECT_EQ(g.DroppedOffGrid(), 0u);
  g.Step(); // column 63 forwards east -> off-grid.
  EXPECT_EQ(g.DroppedOffGrid(), 1u);
  EXPECT_EQ(InputX(g, 0, c), -1);
}

TEST(grid, SramBankModel) {
  Grid g;
  Sram &m = g.At(0, 0).Memory();
  EXPECT_EQ(m.kBanks, 8);
  EXPECT_EQ(m.kTotalBytes, 48 * 1024);

  // Word-interleaved banking spreads consecutive words across the 8 banks.
  EXPECT_EQ(m.BankOf(0), 0);
  EXPECT_EQ(m.BankOf(4), 1);
  EXPECT_EQ(m.BankOf(28), 7);
  EXPECT_EQ(m.BankOf(32), 0); // wraps after 8 words.

  m.BeginCycle();
  uint32_t v = 0;
  // 2 reads + 1 write per cycle; the 3rd read and 2nd write stall.
  EXPECT_TRUE(m.Write(0, 0xCAFEBABE));
  EXPECT_FALSE(m.Write(4, 0x11111111)); // write port exhausted.
  EXPECT_TRUE(m.Read(0, v));
  EXPECT_EQ(v, 0xCAFEBABEu);
  EXPECT_TRUE(m.Read(0, v)); // 2nd read port.
  EXPECT_FALSE(m.Read(0, v)); // read ports exhausted.
  EXPECT_EQ(m.ReadsThisCycle(), 2);
  EXPECT_EQ(m.WritesThisCycle(), 1);

  // A fresh cycle re-opens the ports; round-trip through a different bank.
  m.BeginCycle();
  EXPECT_TRUE(m.Write(4, 0x12345678)); // bank 1.
  EXPECT_TRUE(m.Read(4, v));
  EXPECT_EQ(v, 0x12345678u);

  // Out-of-bounds / misaligned accesses are rejected, not crashed.
  m.BeginCycle();
  EXPECT_FALSE(m.Read(48 * 1024, v));   // past the end.
  EXPECT_FALSE(m.Write(48 * 1024, 1));  // past the end.
  EXPECT_FALSE(m.Read(3, v));           // misaligned.
}

TEST(grid, CeFmacFusedMultiplyAdd) {
  Grid g;
  Ce &ce = g.At(0, 0).Compute();
  ASSERT_EQ(Ce::kLanes, 4);
  // FP16: 1.0 = 0x3C00, 2.0 = 0x4000, 3.0 = 0x4200, 5.0 = 0x4500.
  const Ce::LaneVec a = {0x3C00, 0x3C00, 0x3C00, 0x3C00}; // 1.0
  const Ce::LaneVec b = {0x4000, 0x4000, 0x4000, 0x4000}; // 2.0
  const Ce::LaneVec c = {0x4200, 0x4200, 0x4200, 0x4200}; // 3.0
  const auto out = ce.Fma(a, b, c);                       // 1*2 + 3 = 5.
  for (int i = 0; i < Ce::kLanes; ++i)
    EXPECT_EQ(out[i], 0x4500) << "lane " << i; // 5.0
}

} // namespace

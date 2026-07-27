//===- router_test.cpp - 5-port router unit tests -------------*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7). Suite `router` so
// `ctest -R 'router|color|grid'` discovers these.
//
// Asserts: 5 ports (RAMP + E/W/N/S) + 32b/link + single-cycle hop constant;
// multicast tx (one wavelet -> a SET of directions); lossless backpressure (a
// full VC stalls the sender, nothing is dropped); and per-color virtual-channel
// isolation at the router (congestion on one color does not block another on the
// same physical link).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Router.h"

#include "gtest/gtest.h"

namespace {

using polykernel::dataflow::Color;
using polykernel::dataflow::ColorConfig;
using polykernel::dataflow::Dir;
using polykernel::dataflow::FabinDsd;
using polykernel::dataflow::FaboutDsd;
using polykernel::dataflow::MakeWavelet;
using polykernel::dataflow::PopCount;
using polykernel::dataflow::Router;
using polykernel::dataflow::Wavelet;

TEST(router, FivePorts32bLinkSingleCycleHop) {
  EXPECT_EQ(Router::kNumPorts, 5); // RAMP + E/W/N/S
  EXPECT_EQ(Router::kLinkBits, 32);
  EXPECT_EQ(Router::kHopLatencyCycles, 1); // single-cycle hop
}

TEST(router, UnicastTx) {
  Router r;
  const Color c = *Color::Routable(3);
  const Wavelet w = MakeWavelet(0xBEEF, 0x0001);
  ASSERT_TRUE(r.Tx(c, Dir::East, w));
  EXPECT_EQ(r.OutputSize(Dir::East, c), 1u);
  EXPECT_EQ(r.OutputSize(Dir::West, c), 0u);
  EXPECT_EQ(r.PeekOutput(Dir::East, c), w);
}

TEST(router, MulticastTx) {
  Router r;
  const Color c = *Color::Routable(7);
  const Wavelet w = MakeWavelet(0x1234, 0x0002);
  // One wavelet transmitted to a SET of directions (East + North) = multicast.
  ASSERT_TRUE(r.Tx(c, Dir::East | Dir::North, w));
  EXPECT_EQ(r.OutputSize(Dir::East, c), 1u);
  EXPECT_EQ(r.OutputSize(Dir::North, c), 1u);
  EXPECT_EQ(r.OutputSize(Dir::West, c), 0u);
  EXPECT_EQ(r.OutputSize(Dir::South, c), 0u);
  // Both fan-out copies carry the same payload.
  EXPECT_EQ(r.PeekOutput(Dir::East, c), w);
  EXPECT_EQ(r.PeekOutput(Dir::North, c), w);
}

TEST(router, LosslessBackpressure) {
  Router r;
  const Color c = *Color::Routable(1);
  const Wavelet w = MakeWavelet(0xAAAA, 0x0003);
  // Fill the East VC to its finite depth.
  for (int i = 0; i < Router::kVcDepth; ++i)
    ASSERT_TRUE(r.Tx(c, Dir::East, w));
  EXPECT_TRUE(r.OutputFull(Dir::East, c));
  // The next tx is stalled (lossless): it returns false and drops nothing.
  EXPECT_FALSE(r.Tx(c, Dir::East, w));
  EXPECT_EQ(r.OutputSize(Dir::East, c),
            static_cast<std::size_t>(Router::kVcDepth));
  // Draining one slot re-opens the VC.
  EXPECT_TRUE(r.PopOutput(Dir::East, c).has_value());
  EXPECT_TRUE(r.Tx(c, Dir::East, w));
}

TEST(router, MulticastIsAllOrNothing) {
  Router r;
  const Color c = *Color::Routable(2);
  const Wavelet w = MakeWavelet(0x5555, 0x0004);
  // Saturate ONLY the North VC.
  for (int i = 0; i < Router::kVcDepth; ++i)
    ASSERT_TRUE(r.Tx(c, Dir::North, w));
  // A multicast East|North must now fail wholesale (no partial fan-out), and
  // East must remain empty (the wavelet was not half-committed).
  EXPECT_FALSE(r.Tx(c, Dir::East | Dir::North, w));
  EXPECT_EQ(r.OutputSize(Dir::East, c), 0u);
}

TEST(router, ColorIsolationOnSharedLink) {
  Router r;
  const Color busy = *Color::Routable(0);
  const Color free = *Color::Routable(1);
  const Wavelet w = MakeWavelet(0x0001, 0x0005);
  // Congest color 0's East VC completely.
  for (int i = 0; i < Router::kVcDepth; ++i)
    ASSERT_TRUE(r.Tx(busy, Dir::East, w));
  ASSERT_TRUE(r.OutputFull(Dir::East, busy));
  // Color 1 shares the SAME physical East link but a different virtual channel:
  // it is NOT blocked by color 0's congestion.
  EXPECT_TRUE(r.Tx(free, Dir::East, w));
  EXPECT_EQ(r.OutputSize(Dir::East, free), 1u);
}

TEST(router, RxBackpressure) {
  Router r;
  const Color c = *Color::Routable(4);
  const Wavelet w = MakeWavelet(0x0009, 0x0006);
  for (int i = 0; i < Router::kVcDepth; ++i)
    ASSERT_TRUE(r.Rx(Dir::West, c, w));
  EXPECT_FALSE(r.Rx(Dir::West, c, w)); // input VC full -> lossless stall.
  EXPECT_EQ(r.InputSize(c), static_cast<std::size_t>(Router::kVcDepth));
}

TEST(router, FabinFaboutDsd) {
  Router r;
  const Color c = *Color::Routable(5);
  const Wavelet w = MakeWavelet(0xCAFE, 0x0007);
  // fabin_dsd: CE writes a wavelet INTO the fabric (multicast East+South).
  ASSERT_TRUE(r.Fabin(FabinDsd{c, Dir::East | Dir::South}, w));
  EXPECT_EQ(r.OutputSize(Dir::East, c), 1u);
  EXPECT_EQ(r.OutputSize(Dir::South, c), 1u);
  // fabout_dsd: CE reads a received wavelet FROM the fabric.
  ASSERT_TRUE(r.Rx(Dir::North, c, w));
  const auto got = r.Fabout(FaboutDsd{c});
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(*got, w);
}

TEST(router, SetColorConfigRxTx) {
  Router r;
  const Color c = *Color::Routable(9);
  // @set_color_config: rx = one direction, tx = a SET of directions.
  ASSERT_TRUE(r.SetColorConfig(ColorConfig{c, Dir::West, Dir::East | Dir::North}));
  const auto cfg = r.GetColorConfig(c);
  ASSERT_TRUE(cfg.has_value());
  EXPECT_EQ(cfg->rx, Dir::West);
  EXPECT_EQ(PopCount(cfg->tx), 2); // multicast tx set.
  // A non-routable color (ID 24) is rejected.
  const Color bad = *Color::FromBits(24);
  EXPECT_FALSE(r.SetColorConfig(ColorConfig{bad, Dir::West, Dir::East}));
}

TEST(router, ApplyForwardingHonorsConfig) {
  Router r;
  const Color c = *Color::Routable(6);
  const Wavelet w = MakeWavelet(0x00FF, 0x0008);
  ASSERT_TRUE(r.SetColorConfig(ColorConfig{c, Dir::West, Dir::East}));
  ASSERT_TRUE(r.Rx(Dir::West, c, w)); // arrives on the rx direction.
  EXPECT_EQ(r.ApplyForwarding(), 1);  // routed input -> output East.
  EXPECT_EQ(r.InputSize(c), 0u);
  EXPECT_EQ(r.OutputSize(Dir::East, c), 1u);
  EXPECT_EQ(r.PeekOutput(Dir::East, c), w);
}

} // namespace

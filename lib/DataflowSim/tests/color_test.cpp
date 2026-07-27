//===- color_test.cpp - Fabric color / virtual-channel tests ---*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7). Suite `color` so
// `ctest -R 'router|color|grid'` discovers these.
//
// Asserts: the color is a 5-bit tag with 24 routable colors (IDs 0-23; 24-31
// encodable but not routable); `@set_color_config` rx/tx semantics; and - the
// key fabric property - virtual-channel isolation across all 24 colors sharing
// one physical link (congestion on one color never blocks another).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Color.h"
#include "PolyKernel/Dataflow/Router.h"

#include "gtest/gtest.h"

namespace {

using polykernel::dataflow::Color;
using polykernel::dataflow::ColorConfig;
using polykernel::dataflow::Dir;
using polykernel::dataflow::DirName;
using polykernel::dataflow::HasDir;
using polykernel::dataflow::MakeWavelet;
using polykernel::dataflow::PopCount;
using polykernel::dataflow::Router;

TEST(color, FiveBitTagTwentyFourRoutable) {
  EXPECT_EQ(Color::kBits, 5);
  EXPECT_EQ(Color::kNumRoutable, 24);
  EXPECT_EQ(Color::kMaxRoutable, 23);
  EXPECT_EQ(Color::kMaxEncodable, 31); // 2^5 - 1
}

TEST(color, RoutableRange) {
  ASSERT_TRUE(Color::Routable(0).has_value());
  EXPECT_EQ(Color::Routable(0)->Id(), 0);
  ASSERT_TRUE(Color::Routable(23).has_value());
  EXPECT_TRUE(Color::Routable(23)->IsRoutable());
  EXPECT_FALSE(Color::Routable(24).has_value()); // 24 is not routable.

  // 24..31 are encodable in 5 bits but NOT routable.
  const auto c31 = Color::FromBits(31);
  ASSERT_TRUE(c31.has_value());
  EXPECT_FALSE(c31->IsRoutable());
  // 32 does not fit in 5 bits -> rejected (parse-don't-validate).
  EXPECT_FALSE(Color::FromBits(32).has_value());
}

TEST(color, DirMaskAndNames) {
  const auto mask = Dir::East | Dir::North | Dir::South;
  EXPECT_EQ(PopCount(mask), 3);
  EXPECT_TRUE(HasDir(mask, Dir::East));
  EXPECT_TRUE(HasDir(mask, Dir::North));
  EXPECT_FALSE(HasDir(mask, Dir::West));
  EXPECT_STREQ(DirName(Dir::Ramp), "ramp");
  EXPECT_STREQ(DirName(Dir::East), "east");
  EXPECT_STREQ(DirName(Dir::West), "west");
  EXPECT_STREQ(DirName(Dir::North), "north");
  EXPECT_STREQ(DirName(Dir::South), "south");
}

TEST(color, SetColorConfigRxTx) {
  // @set_color_config binds ONE rx direction and a tx direction SET.
  ColorConfig cfg{*Color::Routable(11), Dir::West, Dir::East | Dir::North};
  EXPECT_EQ(cfg.rx, Dir::West);
  EXPECT_EQ(PopCount(cfg.tx), 2); // multicast.
  EXPECT_TRUE(HasDir(cfg.tx, Dir::East));
  EXPECT_TRUE(HasDir(cfg.tx, Dir::North));
}

// The fabric property: 24 colors share ONE physical link as independent virtual
// channels. Saturating every VC of color 0 on the East link must NOT block any
// of colors 1..23 on that same link.
TEST(color, TwentyFourColorIsolation) {
  Router r;
  const auto w = MakeWavelet(0x0042, 0x0001);

  // Fully congest color 0 on the East link.
  const Color congested = *Color::Routable(0);
  for (int i = 0; i < Router::kVcDepth; ++i)
    ASSERT_TRUE(r.Tx(congested, Dir::East, w));
  ASSERT_TRUE(r.OutputFull(Dir::East, congested));
  EXPECT_FALSE(r.Tx(congested, Dir::East, w)); // color 0 is stalled.

  // Every other routable color still flows on the SAME physical East link.
  for (unsigned id = 1; id < Color::kNumRoutable; ++id) {
    const Color c = *Color::Routable(id);
    EXPECT_TRUE(r.Tx(c, Dir::East, w)) << "color " << id << " was blocked";
    EXPECT_EQ(r.OutputSize(Dir::East, c), 1u);
  }
}

// Isolation is symmetric: congesting an arbitrary color K never blocks J != K.
TEST(color, PairwiseIsolation) {
  const auto w = MakeWavelet(0x0077, 0x0002);
  for (unsigned k = 0; k < Color::kNumRoutable; ++k) {
    Router r;
    const Color ck = *Color::Routable(k);
    for (int i = 0; i < Router::kVcDepth; ++i)
      ASSERT_TRUE(r.Tx(ck, Dir::North, w));
    ASSERT_TRUE(r.OutputFull(Dir::North, ck));
    const Color other = *Color::Routable((k + 1) % Color::kNumRoutable);
    EXPECT_TRUE(r.Tx(other, Dir::North, w))
        << "color " << other.Id() << " blocked by color " << k;
  }
}

} // namespace

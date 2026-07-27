//===- Color.h - Dataflow fabric color (virtual channel) -------*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7).
//
// THIS IS A SIMULATOR. It is a GPU-free, cycle-level *functional model* of a
// Cerebras-style wafer-scale dataflow fabric. It does NOT compile real CSL,
// does NOT invoke cslc, and does NOT run on Cerebras hardware - it models the
// fabric's routing behaviour in portable C++20 so the rest of Wave 7 (task
// model + scheduler, SUMMA mapping, traffic metrics, viz) has something to
// drive.
//
// A *color* is the fabric's virtual-channel tag. The CS-3 fabric carries a
// 5-bit color field; 24 of the 32 encodable values (IDs 0-23) are routable
// colors. Multiple colors share ONE physical link as independent virtual
// channels, so congestion (backpressure) on one color does NOT block another
// color on the same link.
//
// Terminology (matches the Cerebras CSL fabric model, NOT legacy nicknames):
//   - The compute unit is the CE (compute element) with FMAC units - there is
//     no "FMU"/"PMU" here.
//   - A color's routing is configured with `@set_color_config`, which takes an
//     `rx` direction (one receive direction) and a `tx` direction SET (the set
//     of transmit directions = multicast). It is NOT called "set_color".
//   - Fabric data moves through `fabin_dsd` / `fabout_dsd` (fabric-in /
//     fabric-out data-stream descriptors) - NOT "fdata"/"fcast"/"fmove".
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DATAFLOW_COLOR_H
#define POLYKERNEL_DATAFLOW_COLOR_H

#include <cstdint>
#include <optional>

namespace polykernel::dataflow {

//===----------------------------------------------------------------------===//
// Router directions (the 5 router ports).
//===----------------------------------------------------------------------===//

// A router has 5 ports: the local RAMP (CE <-> router) plus the four mesh
// neighbours East / West / North / South.
enum class Dir : uint8_t {
  Ramp = 0, // local port to the PE's CE + SRAM (no grid neighbour).
  East = 1,
  West = 2,
  North = 3,
  South = 4,
  Count = 5,
};

// Number of physical router ports (RAMP + E/W/N/S).
inline constexpr int kNumDirs = static_cast<int>(Dir::Count);

// A set of directions (one bit per Dir). A single Dir converts implicitly, so
// both `Dir::East` (unicast) and `Dir::East | Dir::North` (multicast) are valid
// wherever a DirMask is expected.
struct DirMask {
  uint8_t bits = 0;

  constexpr DirMask() = default;
  // Implicit: a lone direction is a one-element direction set.
  constexpr DirMask(Dir d)
      : bits(static_cast<uint8_t>(1u << static_cast<unsigned>(d))) {}

  static constexpr DirMask FromBits(uint8_t b) {
    DirMask m;
    m.bits = b;
    return m;
  }

  constexpr bool operator==(const DirMask &) const = default;
};

inline constexpr DirMask DirBit(Dir d) { return DirMask(d); }

inline constexpr DirMask operator|(DirMask a, DirMask b) {
  return DirMask::FromBits(static_cast<uint8_t>(a.bits | b.bits));
}
inline constexpr DirMask operator|(Dir a, Dir b) { return DirMask(a) | DirMask(b); }
inline constexpr DirMask operator|(DirMask a, Dir b) { return a | DirMask(b); }
inline constexpr bool HasDir(DirMask mask, Dir d) {
  return (mask.bits & DirMask(d).bits) != 0;
}

// Invoke `fn(dir)` for every direction bit set in `mask` (ascending order).
template <typename Fn>
constexpr void ForEachDir(DirMask mask, Fn &&fn) {
  for (unsigned d = 0; d < static_cast<unsigned>(Dir::Count); ++d)
    if ((mask.bits & (1u << d)) != 0)
      fn(static_cast<Dir>(d));
}

// Number of direction bits set in a mask (1 = unicast, >1 = multicast).
int PopCount(DirMask mask);

// Lower-case name of a direction ("ramp", "east", "west", "north", "south").
const char *DirName(Dir d);

//===----------------------------------------------------------------------===//
// Color: a 5-bit virtual-channel tag (24 routable colors, IDs 0-23).
//===----------------------------------------------------------------------===//

class Color {
public:
  // Width of the fabric color field, in bits.
  static constexpr int kBits = 5;
  // Number of routable colors (IDs 0..kNumRoutable-1).
  static constexpr int kNumRoutable = 24;
  // Largest routable color ID.
  static constexpr uint8_t kMaxRoutable = kNumRoutable - 1; // 23
  // Largest value encodable in kBits bits (2^5 - 1 = 31). Values 24..31 are
  // encodable but NOT routable.
  static constexpr uint8_t kMaxEncodable = (1u << kBits) - 1; // 31

  constexpr Color() = default;

  // Build a color from a raw 5-bit value. Returns nullopt if `id` does not fit
  // in 5 bits (> 31) - parse-don't-validate. Values 24..31 ARE accepted here
  // (they are encodable) but IsRoutable() will report false for them.
  static constexpr std::optional<Color> FromBits(unsigned id) {
    if (id > kMaxEncodable)
      return std::nullopt;
    Color c;
    c.id_ = static_cast<uint8_t>(id);
    return c;
  }

  // Build a routable color (ID 0..23). Returns nullopt if `id >= 24`.
  static constexpr std::optional<Color> Routable(unsigned id) {
    if (id >= kNumRoutable)
      return std::nullopt;
    Color c;
    c.id_ = static_cast<uint8_t>(id);
    return c;
  }

  constexpr uint8_t Id() const { return id_; }
  // True iff this color is one of the 24 routable colors (ID 0..23).
  constexpr bool IsRoutable() const { return id_ <= kMaxRoutable; }

  constexpr bool operator==(const Color &) const = default;

private:
  uint8_t id_ = 0;
};

//===----------------------------------------------------------------------===//
// ColorConfig: the `@set_color_config` rx/tx routing descriptor.
//===----------------------------------------------------------------------===//

// Models CSL `@set_color_config`: bind a color to ONE receive direction (`rx`)
// and a SET of transmit directions (`tx`). A non-singleton `tx` set is a
// multicast route. The router applies this config when forwarding wavelets that
// arrive on the color.
struct ColorConfig {
  Color color;        // the color being configured (must be routable).
  Dir rx = Dir::Ramp; // single receive direction.
  DirMask tx;         // set of transmit directions (multicast when > 1 bit).
};

} // namespace polykernel::dataflow

#endif // POLYKERNEL_DATAFLOW_COLOR_H

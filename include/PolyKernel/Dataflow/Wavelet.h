//===- Wavelet.h - 32-bit dataflow wavelet (flit) --------------*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7).
//
// THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
// hardware).
//
// A *wavelet* is the fabric's unit of data movement - a 32-bit flit split into
// a 16-bit data field and a 16-bit control/index field. Wavelets travel between
// PEs over the router fabric on a *color* (virtual channel); the color is the
// channel a wavelet is sent on (see Color.h / Router.h), not part of the 32-bit
// payload.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DATAFLOW_WAVELET_H
#define POLYKERNEL_DATAFLOW_WAVELET_H

#include <cstdint>

namespace polykernel::dataflow {

// A 32-bit wavelet: 16 bits of data + 16 bits of control/index.
struct Wavelet {
  // Total wavelet width, and its two field widths, in bits.
  static constexpr int kBits = 32;
  static constexpr int kDataBits = 16;
  static constexpr int kControlBits = 16;

  uint16_t data = 0;    // 16-bit data field.
  uint16_t control = 0; // 16-bit control/index field.

  // Pack into a single 32-bit word: control in the high half, data in the low.
  constexpr uint32_t Pack() const {
    return (static_cast<uint32_t>(control) << kDataBits) |
           static_cast<uint32_t>(data);
  }

  // Unpack a 32-bit word back into a wavelet.
  static constexpr Wavelet Unpack(uint32_t word) {
    Wavelet w;
    w.data = static_cast<uint16_t>(word & 0xFFFFu);
    w.control = static_cast<uint16_t>(word >> kDataBits);
    return w;
  }

  constexpr bool operator==(const Wavelet &) const = default;
};

// The wavelet is exactly 32 bits (4 bytes) on the wire.
static_assert(sizeof(Wavelet) == 4, "wavelet must be a 32-bit (4-byte) flit");
static_assert(Wavelet::kDataBits + Wavelet::kControlBits == Wavelet::kBits,
              "wavelet fields must sum to the wavelet width");

// Construct a wavelet from its 16-bit data and 16-bit control/index fields.
Wavelet MakeWavelet(uint16_t data, uint16_t control);

} // namespace polykernel::dataflow

#endif // POLYKERNEL_DATAFLOW_WAVELET_H

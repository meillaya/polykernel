//===- Router.h - 5-port dataflow router (virtual channels) ----*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7).
//
// THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
// hardware).
//
// Each PE owns ONE 5-port router: the local RAMP port (to the CE + SRAM) plus
// the four mesh links East / West / North / South. Model parameters (CS-3):
//   - 32 bits per link (one wavelet wide),
//   - single-cycle hop (a wavelet crosses exactly one link per cycle),
//   - lossless backpressure (a full virtual channel stalls the sender; the
//     wavelet is NEVER dropped inside the router),
//   - per-color virtual channels: every (direction, color) pair is an
//     independent finite-depth FIFO, so backpressure on one color does NOT
//     block a different color on the SAME physical link.
//
// Routing verbs (CSL fabric terminology):
//   - rx  = receive a wavelet from ONE direction (single direction),
//   - tx  = transmit a wavelet to a SET of directions (a direction set with
//           more than one bit is a multicast),
//   - `@set_color_config` binds a color to an rx direction and a tx direction
//     set; the router applies that config when it forwards the color,
//   - `fabin_dsd` / `fabout_dsd` are the fabric-in / fabric-out data-stream
//     descriptors the CE uses to write wavelets into / read wavelets from the
//     fabric.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DATAFLOW_ROUTER_H
#define POLYKERNEL_DATAFLOW_ROUTER_H

#include "PolyKernel/Dataflow/Color.h"
#include "PolyKernel/Dataflow/Wavelet.h"

#include <array>
#include <deque>
#include <optional>

namespace polykernel::dataflow {

// Fabric-in data-stream descriptor (`fabin_dsd`): the CE writes a wavelet INTO
// the fabric on `color`, transmitted to the direction set `dirs` (multicast
// when more than one bit is set).
struct FabinDsd {
  Color color;
  DirMask dirs;
};

// Fabric-out data-stream descriptor (`fabout_dsd`): the CE reads a wavelet FROM
// the fabric on `color`.
struct FaboutDsd {
  Color color;
};

class Router {
public:
  // Bits per physical link (one wavelet wide).
  static constexpr int kLinkBits = Wavelet::kBits; // 32
  // Number of router ports (RAMP + E/W/N/S).
  static constexpr int kNumPorts = kNumDirs; // 5
  // A wavelet crosses exactly one link per cycle.
  static constexpr int kHopLatencyCycles = 1;
  // Depth (capacity) of each per-color virtual channel. Finite depth is what
  // produces lossless backpressure.
  static constexpr int kVcDepth = 4;

  Router();

  // --- @set_color_config: bind a color to an rx direction + tx direction set.
  // Returns false (and changes nothing) if the color is not routable.
  bool SetColorConfig(const ColorConfig &cfg);
  std::optional<ColorConfig> GetColorConfig(Color c) const;

  // --- tx: transmit `w` on `color` to the direction SET `dirs` (multicast).
  // Lossless: if ANY target virtual channel is full, NOTHING is committed and
  // false is returned (the sender must retry). Returns true once the wavelet is
  // enqueued on every requested direction.
  bool Tx(Color color, DirMask dirs, const Wavelet &w);

  // --- rx: receive `w` from the single direction `from` on `color` into this
  // router's input virtual channel for the color. Returns false if that input
  // VC is full (lossless backpressure).
  bool Rx(Dir from, Color color, const Wavelet &w);

  // --- fabin_dsd / fabout_dsd: the CE-facing fabric data-stream descriptors.
  // Fabin == tx into the fabric; Fabout == pop the next received wavelet.
  bool Fabin(const FabinDsd &dsd, const Wavelet &w) {
    return Tx(dsd.color, dsd.dirs, w);
  }
  std::optional<Wavelet> Fabout(const FaboutDsd &dsd) {
    return PopInput(dsd.color);
  }

  // --- forwarding: apply each color's `@set_color_config` tx set, draining its
  // input VC into the named output VCs. Stops a color when an output VC fills
  // (lossless). Returns the number of wavelets moved input->output. Called by
  // the Grid once per cycle before the hop phase.
  int ApplyForwarding();

  // --- output-VC accessors (used by the Grid hop phase + tests). ---
  std::size_t OutputSize(Dir d, Color c) const;
  bool OutputFull(Dir d, Color c) const;
  // Front of an output VC without removing it (nullopt if empty). Lets the Grid
  // attempt a lossless hop and leave the wavelet buffered on backpressure.
  std::optional<Wavelet> PeekOutput(Dir d, Color c) const;
  std::optional<Wavelet> PopOutput(Dir d, Color c);

  // --- input-VC accessors. ---
  std::size_t InputSize(Color c) const;
  std::optional<Wavelet> PopInput(Color c);

  // Total wavelets buffered anywhere in this router (input + output VCs).
  std::size_t Buffered() const;

private:
  bool ValidColor(Color c) const { return c.IsRoutable(); }
  std::deque<Wavelet> &OutVc(Dir d, Color c) {
    return out_vc_[static_cast<unsigned>(d)][c.Id()];
  }
  const std::deque<Wavelet> &OutVc(Dir d, Color c) const {
    return out_vc_[static_cast<unsigned>(d)][c.Id()];
  }
  std::deque<Wavelet> &InVc(Color c) { return in_vc_[c.Id()]; }
  const std::deque<Wavelet> &InVc(Color c) const { return in_vc_[c.Id()]; }

  // Per-(direction, color) output virtual channels: out_vc_[dir][color_id].
  std::array<std::array<std::deque<Wavelet>, Color::kNumRoutable>, kNumDirs>
      out_vc_;
  // Per-color input virtual channels: in_vc_[color_id].
  std::array<std::deque<Wavelet>, Color::kNumRoutable> in_vc_;
  // Per-color `@set_color_config` state.
  std::array<ColorConfig, Color::kNumRoutable> config_;
  std::array<bool, Color::kNumRoutable> has_config_;
};

} // namespace polykernel::dataflow

#endif // POLYKERNEL_DATAFLOW_ROUTER_H

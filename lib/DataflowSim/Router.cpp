//===- Router.cpp - 5-port dataflow router ---------------------*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7). See Router.h.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Router.h"

namespace polykernel::dataflow {

Router::Router() { has_config_.fill(false); }

bool Router::SetColorConfig(const ColorConfig &cfg) {
  if (!ValidColor(cfg.color))
    return false;
  config_[cfg.color.Id()] = cfg;
  has_config_[cfg.color.Id()] = true;
  return true;
}

std::optional<ColorConfig> Router::GetColorConfig(Color c) const {
  if (!ValidColor(c) || !has_config_[c.Id()])
    return std::nullopt;
  return config_[c.Id()];
}

bool Router::Tx(Color color, DirMask dirs, const Wavelet &w) {
  if (!ValidColor(color) || dirs.bits == 0)
    return false;
  // Lossless backpressure: commit ONLY if every target VC has room, so a
  // multicast is all-or-nothing (never a partial fan-out).
  bool room = true;
  ForEachDir(dirs, [&](Dir d) {
    if (OutVc(d, color).size() >= static_cast<std::size_t>(kVcDepth))
      room = false;
  });
  if (!room)
    return false;
  ForEachDir(dirs, [&](Dir d) { OutVc(d, color).push_back(w); });
  return true;
}

bool Router::Rx(Dir /*from*/, Color color, const Wavelet &w) {
  if (!ValidColor(color))
    return false;
  if (InVc(color).size() >= static_cast<std::size_t>(kVcDepth))
    return false; // lossless: stall the sender, drop nothing.
  InVc(color).push_back(w);
  return true;
}

int Router::ApplyForwarding() {
  int forwarded = 0;
  for (unsigned id = 0; id < Color::kNumRoutable; ++id) {
    if (!has_config_[id])
      continue;
    const DirMask tx = config_[id].tx;
    if (tx.bits == 0)
      continue;
    Color c = *Color::Routable(id);
    auto &in = InVc(c);
    while (!in.empty()) {
      // Need room on EVERY transmit VC before consuming the input wavelet.
      bool room = true;
      ForEachDir(tx, [&](Dir d) {
        if (OutVc(d, c).size() >= static_cast<std::size_t>(kVcDepth))
          room = false;
      });
      if (!room)
        break; // lossless stall: leave the wavelet buffered for next cycle.
      const Wavelet w = in.front();
      in.pop_front();
      ForEachDir(tx, [&](Dir d) { OutVc(d, c).push_back(w); });
      ++forwarded;
    }
  }
  return forwarded;
}

std::size_t Router::OutputSize(Dir d, Color c) const {
  if (!ValidColor(c))
    return 0;
  return OutVc(d, c).size();
}

bool Router::OutputFull(Dir d, Color c) const {
  if (!ValidColor(c))
    return true;
  return OutVc(d, c).size() >= static_cast<std::size_t>(kVcDepth);
}

std::optional<Wavelet> Router::PeekOutput(Dir d, Color c) const {
  if (!ValidColor(c))
    return std::nullopt;
  const auto &vc = OutVc(d, c);
  if (vc.empty())
    return std::nullopt;
  return vc.front();
}

std::optional<Wavelet> Router::PopOutput(Dir d, Color c) {
  if (!ValidColor(c))
    return std::nullopt;
  auto &vc = OutVc(d, c);
  if (vc.empty())
    return std::nullopt;
  Wavelet w = vc.front();
  vc.pop_front();
  return w;
}

std::size_t Router::InputSize(Color c) const {
  if (!ValidColor(c))
    return 0;
  return InVc(c).size();
}

std::optional<Wavelet> Router::PopInput(Color c) {
  if (!ValidColor(c))
    return std::nullopt;
  auto &vc = InVc(c);
  if (vc.empty())
    return std::nullopt;
  Wavelet w = vc.front();
  vc.pop_front();
  return w;
}

std::size_t Router::Buffered() const {
  std::size_t n = 0;
  for (unsigned d = 0; d < static_cast<unsigned>(kNumDirs); ++d)
    for (unsigned id = 0; id < Color::kNumRoutable; ++id)
      n += out_vc_[d][id].size();
  for (unsigned id = 0; id < Color::kNumRoutable; ++id)
    n += in_vc_[id].size();
  return n;
}

} // namespace polykernel::dataflow

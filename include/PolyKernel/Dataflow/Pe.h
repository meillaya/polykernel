//===- Pe.h - Processing element (CE + SRAM + router) ----------*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7).
//
// THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
// hardware).
//
// A processing element (PE) is the CS-3 fabric node and bundles three things:
//   - the CE (compute element): 4 FMAC units on a 64-bit datapath (4 x 16-bit
//     FP16 lanes), each FMAC a fused multiply-add. (The compute unit is the CE
//     with FMACs - there is no "FMU"/"PMU".)
//   - 48 KB of SRAM organised as 8 banks x 6 KB, word-interleaved, with 2 read
//     ports + 1 write port per cycle,
//   - the 5-port router (RAMP + E/W/N/S) that moves wavelets (see Router.h).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DATAFLOW_PE_H
#define POLYKERNEL_DATAFLOW_PE_H

#include "PolyKernel/Dataflow/Router.h"

#include <array>
#include <cstdint>
#include <vector>

namespace polykernel::dataflow {

//===----------------------------------------------------------------------===//
// Ce: compute element (4x FP16 FMAC, 64-bit datapath).
//===----------------------------------------------------------------------===//

class Ce {
public:
  // Number of FMAC (fused multiply-add) units in the CE.
  static constexpr int kFmacCount = 4;
  // Datapath width in bits (4 x 16-bit FP16 lanes).
  static constexpr int kDatapathBits = 64;
  // Number of FP16 lanes = datapath / 16.
  static constexpr int kLanes = kDatapathBits / 16; // 4

  // A vector of FP16 lanes (each lane stored as its 16-bit half-precision bit
  // pattern).
  using LaneVec = std::array<uint16_t, kLanes>;

  // Fused multiply-add across all FMAC lanes: out[i] = a[i] * b[i] + c[i],
  // computed in FP16. One FMAC per lane, all in one cycle.
  LaneVec Fma(const LaneVec &a, const LaneVec &b, const LaneVec &c) const;
};

//===----------------------------------------------------------------------===//
// Sram: 48 KB = 8 banks x 6 KB, 2 reads + 1 write per cycle.
//===----------------------------------------------------------------------===//

class Sram {
public:
  static constexpr int kBanks = 8;
  static constexpr int kBankBytes = 6 * 1024;             // 6 KB per bank.
  static constexpr int kTotalBytes = kBanks * kBankBytes; // 48 KB.
  static constexpr int kReadPorts = 2;                    // 2 reads / cycle.
  static constexpr int kWritePorts = 1;                   // 1 write / cycle.
  static constexpr int kWordBytes = 4;                    // 32-bit word.

  Sram();

  // Reset the per-cycle read/write port accounting (call once per cycle).
  void BeginCycle();

  // Word-interleaved bank index for a byte address: (addr / word) mod kBanks.
  int BankOf(uint32_t addr) const;

  // Read a 32-bit word. Returns false (no side effects) if the address is out
  // of range, misaligned, or the 2 read ports for this cycle are exhausted.
  bool Read(uint32_t addr, uint32_t &out);

  // Write a 32-bit word. Returns false (no side effects) if the address is out
  // of range, misaligned, or the 1 write port for this cycle is exhausted.
  bool Write(uint32_t addr, uint32_t val);

  int ReadsThisCycle() const { return reads_; }
  int WritesThisCycle() const { return writes_; }

private:
  bool InRange(uint32_t addr) const {
    return addr + kWordBytes <= static_cast<uint32_t>(kTotalBytes) &&
           (addr % kWordBytes) == 0;
  }
  // Bank-local byte offset for a (valid, word-aligned) address.
  uint32_t BankOffset(uint32_t addr) const;

  std::array<std::vector<uint8_t>, kBanks> bank_;
  int reads_ = 0;
  int writes_ = 0;
};

//===----------------------------------------------------------------------===//
// Pe: CE + SRAM + 5-port router.
//===----------------------------------------------------------------------===//

class Pe {
public:
  Ce &Compute() { return ce_; }
  Sram &Memory() { return sram_; }
  Router &GetRouter() { return router_; }

  const Ce &Compute() const { return ce_; }
  const Sram &Memory() const { return sram_; }
  const Router &GetRouter() const { return router_; }

private:
  Ce ce_;
  Sram sram_;
  Router router_;
};

} // namespace polykernel::dataflow

#endif // POLYKERNEL_DATAFLOW_PE_H

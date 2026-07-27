//===- Pe.cpp - Processing element (CE + SRAM + router) --------*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7). See Pe.h.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Pe.h"

#include <bit>
#include <cmath>

namespace polykernel::dataflow {

namespace {

// IEEE-754 half (1-5-10) -> float. Functional-model precision is ample for the
// simulator's golden checks (which use exactly-representable values).
float HalfToFloat(uint16_t h) {
  const uint32_t sign = (uint32_t{h} & 0x8000u) << 16;
  uint32_t exp = (uint32_t{h} >> 10) & 0x1Fu;
  uint32_t mant = uint32_t{h} & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign; // signed zero
    } else {
      // Subnormal: normalise into a float exponent.
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (mant << 13); // inf / nan
  } else {
    f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  return std::bit_cast<float>(f);
}

// float -> IEEE-754 half (1-5-10), round-toward-zero.
uint16_t FloatToHalf(float value) {
  const uint32_t f = std::bit_cast<uint32_t>(value);
  const uint32_t sign = (f >> 16) & 0x8000u;
  const uint32_t fexp = (f >> 23) & 0xFFu;
  uint32_t mant = f & 0x7FFFFFu;
  if (fexp == 0xFFu) // inf / nan
    return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
  const int32_t exp = static_cast<int32_t>(fexp) - 127 + 15;
  if (exp >= 0x1F)
    return static_cast<uint16_t>(sign | 0x7C00u); // overflow -> inf
  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign); // underflow -> signed zero
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    return static_cast<uint16_t>(sign | (mant >> shift));
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               (mant >> 13));
}

} // namespace

Ce::LaneVec Ce::Fma(const LaneVec &a, const LaneVec &b, const LaneVec &c) const {
  LaneVec out;
  for (int i = 0; i < kLanes; ++i) {
    const float r = HalfToFloat(a[i]) * HalfToFloat(b[i]) + HalfToFloat(c[i]);
    out[i] = FloatToHalf(r);
  }
  return out;
}

Sram::Sram() {
  for (auto &b : bank_)
    b.assign(kBankBytes, 0);
}

void Sram::BeginCycle() {
  reads_ = 0;
  writes_ = 0;
}

int Sram::BankOf(uint32_t addr) const {
  return static_cast<int>((addr / kWordBytes) % kBanks);
}

uint32_t Sram::BankOffset(uint32_t addr) const {
  // Within a bank, words are strided by kBanks: word w lives in bank
  // (w mod kBanks) at bank-local word (w / kBanks).
  const uint32_t word = addr / kWordBytes;
  return (word / kBanks) * kWordBytes;
}

bool Sram::Read(uint32_t addr, uint32_t &out) {
  if (!InRange(addr) || reads_ >= kReadPorts)
    return false;
  const int bank = BankOf(addr);
  const uint32_t off = BankOffset(addr);
  uint32_t v = 0;
  for (int i = 0; i < kWordBytes; ++i)
    v |= uint32_t{bank_[bank][off + i]} << (8 * i);
  out = v;
  ++reads_;
  return true;
}

bool Sram::Write(uint32_t addr, uint32_t val) {
  if (!InRange(addr) || writes_ >= kWritePorts)
    return false;
  const int bank = BankOf(addr);
  const uint32_t off = BankOffset(addr);
  for (int i = 0; i < kWordBytes; ++i)
    bank_[bank][off + i] = static_cast<uint8_t>((val >> (8 * i)) & 0xFFu);
  ++writes_;
  return true;
}

} // namespace polykernel::dataflow

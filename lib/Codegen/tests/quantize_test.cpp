//===- quantize_test.cpp - Quantization codegen emitter tests ----*- C++ -*-===//
//
// PolyKernel quantization codegen lowering unit tests (Todo 41 / Wave 8 elite).
//
//===----------------------------------------------------------------------===//
//
// Validates the lib/Codegen/Quantize.cpp source emitter: the int8 weight-only
// emitter produces a complete portable kernel (kernel + checked launch entry, the
// dequant-by-per-channel-scale inner loop), the fp8 e4m3/e5m2 paths emit a
// clearly-labeled SIMULATION note (never a hardware fp8 claim), and the QuantPath
// dispatch is consistent. GPU-free: this only checks the emitted TEXT.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Codegen/Quantize.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>

using namespace polykernel::codegen;

namespace {

std::string emitInt8() {
  std::string s;
  llvm::raw_string_ostream os(s);
  emitMatMulInt8Cuda(os);
  return s;
}

std::string emitPath(QuantPath path) {
  std::string s;
  llvm::raw_string_ostream os(s);
  emitQuantizeCuda(os, path);
  return s;
}

TEST(quantize, int8_emits_complete_portable_kernel) {
  const std::string k = emitInt8();
  EXPECT_NE(k.find("kernel_common.h"), std::string::npos) << "portable template";
  EXPECT_NE(k.find("matmul_int8_kernel"), std::string::npos) << "kernel symbol";
  EXPECT_NE(k.find("launch_matmul_int8"), std::string::npos) << "launch entry";
  EXPECT_NE(k.find("int8_t"), std::string::npos) << "int8 weights";
  EXPECT_NE(k.find("pk_bf16"), std::string::npos) << "bf16 activations";
  EXPECT_NE(k.find("scale[col]"), std::string::npos) << "per-channel dequant scale";
  EXPECT_NE(k.find("PK_LAUNCH"), std::string::npos) << "portable launch macro";
  EXPECT_NE(k.find("PK_CHECK"), std::string::npos) << "launch is error-checked";
}

TEST(quantize, int8_dispatch_is_byte_identical_to_direct_emitter) {
  EXPECT_EQ(emitPath(QuantPath::Int8WeightOnly), emitInt8())
      << "emitQuantizeCuda(Int8WeightOnly) must dispatch to emitMatMulInt8Cuda";
}

TEST(quantize, fp8_sim_paths_are_labeled_simulation_not_hardware) {
  for (QuantPath p : {QuantPath::Fp8SimE4M3, QuantPath::Fp8SimE5M2}) {
    const std::string s = emitPath(p);
    EXPECT_NE(s.find("fp8"), std::string::npos) << "names fp8";
    EXPECT_NE(s.find("SIMULATION"), std::string::npos) << "labeled simulation";
    EXPECT_NE(s.find("NOT a hardware fp8 kernel"), std::string::npos)
        << "never claims fp8 hardware";
    EXPECT_NE(s.find("float8"), std::string::npos) << "points at ml_dtypes float8";
    EXPECT_EQ(s.find("matmul_int8_kernel"), std::string::npos)
        << "fp8-sim path is a note, not the int8 kernel";
  }
  EXPECT_NE(emitPath(QuantPath::Fp8SimE4M3).find("e4m3fn"), std::string::npos);
  EXPECT_NE(emitPath(QuantPath::Fp8SimE5M2).find("e5m2"), std::string::npos);
}

TEST(quantize, path_names_are_stable_contract_h_annotations) {
  EXPECT_STREQ(quantPathName(QuantPath::Int8WeightOnly), "int8_weight_only");
  EXPECT_STREQ(quantPathName(QuantPath::Fp8SimE4M3), "fp8_sim_e4m3");
  EXPECT_STREQ(quantPathName(QuantPath::Fp8SimE5M2), "fp8_sim_e5m2");
}

} // namespace

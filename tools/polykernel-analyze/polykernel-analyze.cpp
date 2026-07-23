//===- polykernel-analyze.cpp - Compile-time analyzer CLI -------*- C++ -*-===//
//
// PolyKernel compile-time analyzer (Todo 11 / Wave 2). GPU-FREE standalone CLI:
// reads a captured `ptxas -v` log + a GEMM shape + arch and emits the contract H
// per-kernel report (JSON or text). Wrapped by tools/polykernel-bench/analyze.py
// (which runs nvcc/ptxas to produce the log). A malformed ptxas log is reported
// as an explicit parse error on stderr with exit 1 - never a crash, never zeros.
//
//===----------------------------------------------------------------------===//

#include "KernelReport.h"
#include "Occupancy.h"
#include "Roofline.h"
#include "PolyKernel/Analysis/PtxasParser.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using polykernel::analysis::Arch;
using polykernel::analysis::Backend;
using polykernel::analysis::GemmShape;
using polykernel::analysis::Path;

struct Args {
  std::string kernel = "matmul";
  std::string backend = "cuda";
  std::string arch = "sm_90";
  std::string shape = "128,128,128";
  std::string path = "scalar";
  std::string ptxas_log = "-"; // "-" => stdin.
  std::string format = "json";
  int dtype_bytes = 2;
  int threads_per_block = 256;
};

void PrintUsage(const char *prog) {
  std::cerr
      << "usage: " << prog
      << " --kernel NAME --backend cuda|hip --arch sm_80|sm_90\n"
         "       --shape M,N,K --dtype-bytes N --threads-per-block N\n"
         "       --path scalar|wmma|mma --ptxas-log FILE|- --format json|text\n";
}

// Map a CLI string to its enum; returns false on an unknown value.
bool ParseEnums(const Args &a, Backend &backend, Arch &arch, Path &path) {
  if (a.backend == "cuda")
    backend = Backend::cuda;
  else if (a.backend == "hip")
    backend = Backend::hip;
  else
    return false;

  const auto parsed_arch = polykernel::analysis::ParseArch(a.arch);
  if (!parsed_arch)
    return false;
  arch = *parsed_arch;

  if (a.path == "scalar")
    path = Path::scalar;
  else if (a.path == "wmma")
    path = Path::wmma;
  else if (a.path == "mma")
    path = Path::mma;
  else
    return false;
  return true;
}

// Parse "M,N,K" into a GemmShape (with the element width). Returns false on a
// malformed shape (wrong field count or a non-integer component).
bool ParseShape(const std::string &text, int dtype_bytes, GemmShape &shape) {
  std::vector<std::int64_t> vals;
  std::stringstream ss(text);
  std::string field;
  while (std::getline(ss, field, ',')) {
    try {
      std::size_t consumed = 0;
      const std::int64_t v = std::stoll(field, &consumed);
      if (consumed != field.size())
        return false;
      vals.push_back(v);
    } catch (const std::exception &) {
      return false;
    }
  }
  if (vals.size() != 3)
    return false;
  shape = GemmShape{vals[0], vals[1], vals[2], dtype_bytes};
  return true;
}

std::string ReadLog(const std::string &source) {
  if (source == "-") {
    std::stringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
  }
  std::ifstream f(source);
  if (!f)
    return "";
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

} // namespace

int main(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "error: missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--kernel")
      args.kernel = next("--kernel");
    else if (arg == "--backend")
      args.backend = next("--backend");
    else if (arg == "--arch")
      args.arch = next("--arch");
    else if (arg == "--shape")
      args.shape = next("--shape");
    else if (arg == "--dtype-bytes")
      args.dtype_bytes = std::stoi(next("--dtype-bytes"));
    else if (arg == "--threads-per-block")
      args.threads_per_block = std::stoi(next("--threads-per-block"));
    else if (arg == "--path")
      args.path = next("--path");
    else if (arg == "--ptxas-log")
      args.ptxas_log = next("--ptxas-log");
    else if (arg == "--format")
      args.format = next("--format");
    else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "error: unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return 2;
    }
  }

  Backend backend;
  Arch arch;
  Path path;
  if (!ParseEnums(args, backend, arch, path)) {
    std::cerr << "error: invalid --backend/--arch/--path value\n";
    return 2;
  }

  GemmShape shape;
  if (!ParseShape(args.shape, args.dtype_bytes, shape)) {
    std::cerr << "error: --shape must be M,N,K integers (got '" << args.shape
              << "')\n";
    return 2;
  }

  const std::string log = ReadLog(args.ptxas_log);
  const auto parsed = polykernel::analysis::ParsePtxasLog(log);
  if (!parsed.ok()) {
    std::cerr << parsed.error << "\n";
    return 1;
  }

  polykernel::analysis::ReportInputs inputs;
  inputs.kernel = args.kernel;
  inputs.backend = backend;
  inputs.arch = arch;
  inputs.ptxas = *parsed.stats;
  inputs.threads_per_block = args.threads_per_block;
  inputs.shape = shape;
  inputs.path = path;

  const auto report = polykernel::analysis::BuildReport(inputs);
  if (args.format == "text")
    std::cout << polykernel::analysis::ToText(report);
  else
    std::cout << polykernel::analysis::ToJson(report);
  return 0;
}

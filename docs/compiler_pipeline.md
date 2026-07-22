# PolyKernel Compiler Pipeline — Toolchain & Build Wiring

> W1 spike (Todo 2): out-of-tree MLIR toolchain. This is a **GATE** — it proves
> the out-of-tree MLIR-21 build + dialect registration + `polykernel-opt` tool +
> `lit` pipeline before any dialect/op work (Todo 3+). **GATE PASSED.**

## Status

| Check | Result |
|---|---|
| CMake configures against nix MLIR-21 | ✅ exit 0 |
| Dialect lib `libMLIRPolyKernel.a` builds | ✅ |
| `polykernel-opt` builds + links | ✅ exit 0 |
| `polykernel-opt --help` lists the dialect | ✅ `Available Dialects: builtin, polykernel` |
| `lit` (`check-polykernel`) | ✅ `Passed: 1 (100.00%)`, deterministic across 2 runs |
| C++ standard | ✅ **C++20** (fallback to C++17 not needed) |

Evidence: `reports/w1_spike.log`, `reports/w1_cppstd.log`.

## Build model (Pinned contract A)

Build **inside `nix develop`** (the devenv devShell) with manual `cmake` + `ninja`
against the prebuilt `llvmPackages_21` MLIR from the nix cache. We do **NOT** do a
sandboxed `nix build` of PolyKernel.

```bash
nix develop --impure --accept-flake-config -c bash -c '
  cmake -B build -G Ninja \
    -DMLIR_DIR=/nix/store/2lzr6hy0s5cikj7lxwpl0qhh2idmqfc1-mlir-21.1.8-dev/lib/cmake/mlir \
    -DLLVM_DIR=/nix/store/4768r0pa7fswbi0mdjalqc9pm56fh2vm-llvm-21.1.8-dev/lib/cmake/llvm \
    -DMLIR_TABLEGEN_EXE=$(which mlir-tblgen) \
    -DLLVM_EXTERNAL_LIT=/nix/store/mnrhk9ghglznss3ds1mhd9pz4vlnk0d7-python3.14-lit-18.1.8/bin/.lit-wrapped
  cmake --build build
  cmake --build build --target check-polykernel
'
```

## C++ standard: C++20

`CMAKE_CXX_STANDARD 20` (with `_REQUIRED ON`, extensions OFF) project-wide. MLIR
21.1.8 headers compile cleanly under `-std=c++20` with `llvmPackages_21.clang`
(verified: `-std=c++20` reaches both `polykernel-opt` and `MLIRPolyKernel` compile
commands). The upstream `mlir/examples/standalone` defaults to C++17; had MLIR-21
headers rejected C++20 we would have dropped to 17 — **not needed**. See
`reports/w1_cppstd.log`.

## Resolving the nix store paths (method used)

`mlir-tblgen`, `lit`, `FileCheck` are on `PATH` inside the shell, so
`MLIR_TABLEGEN_EXE=$(which mlir-tblgen)` and `FileCheck` need no hardcoding. For
`MLIR_DIR`/`LLVM_DIR` we need the `.dev` outputs. The robust, reproducible method
that matches the **locked** flake revision is to query the flake's own nixpkgs
input (not `nixpkgs#...`, which may resolve to a different rev):

```bash
nix eval --raw --impure --expr \
  '(builtins.getFlake "/home/mei/projects/polykernel").inputs.nixpkgs.legacyPackages.x86_64-linux.llvmPackages_21.mlir.dev.outPath'
# -> /nix/store/2lzr6hy0s5cikj7lxwpl0qhh2idmqfc1-mlir-21.1.8-dev   (MLIR_DIR = .../lib/cmake/mlir)

nix eval --raw --impure --expr \
  '(builtins.getFlake "/home/mei/projects/polykernel").inputs.nixpkgs.legacyPackages.x86_64-linux.llvmPackages_21.llvm.dev.outPath'
# -> /nix/store/4768r0pa7fswbi0mdjalqc9pm56fh2vm-llvm-21.1.8-dev   (LLVM_DIR = .../lib/cmake/llvm)
```

Cross-check used: the `mlir-21.1.8.drv` references exactly
`4768r0pa7fswbi0mdjalqc9pm56fh2vm-llvm-21.1.8-dev`, confirming that llvm-dev is the
one MLIR was built against (a second, stale `238i…-llvm-21.1.8-dev` from an earlier
evaluation also exists in the store and must NOT be used).

`LLVM_EXTERNAL_LIT`: the nix `lit` package ships a wrapped entry point; use
`${lit}/bin/.lit-wrapped` (the nix lit also provides the `lit.llvm` Python module
that `test/lit.cfg.py` imports). `FileCheck`/`count`/`not` live in
`LLVM_TOOLS_BINARY_DIR` (`.../llvm-21.1.8/bin`), which `LLVMConfig.cmake` exports and
the lit site config substitutes.

## CMake wiring

Modeled on `mlir/examples/standalone`. Top-level `CMakeLists.txt`:

- `cmake_minimum_required(VERSION 3.20.0)`, `project(PolyKernel LANGUAGES CXX C)`
  (C is enabled because LLVM/MLIR CMake modules probe the C compiler; matches the
  upstream example).
- **nix gotcha (llvm/llvm-project#150986):** nixpkgs patches `MLIRConfig.cmake`, so
  `MLIR_TABLEGEN_EXE` MUST be set (via `-DMLIR_TABLEGEN_EXE=…`) **before**
  `find_package(MLIR REQUIRED CONFIG)`. The top-level CMake `FATAL_ERROR`s if it is
  unset.
- `list(APPEND CMAKE_MODULE_PATH …)` for `MLIR_CMAKE_DIR` + `LLVM_CMAKE_DIR`, then
  `include(TableGen) include(AddLLVM) include(AddMLIR) include(HandleLLVMOptions)`.
- Include dirs: `LLVM_INCLUDE_DIRS`, `MLIR_INCLUDE_DIRS`, source `include/`, binary
  `include/` (for generated `.inc`).
- Subdirs: `include/PolyKernel`, `lib`, `tools/polykernel-opt`, `test`.
- We deliberately do **not** set `LLVM_RUNTIME_OUTPUT_INTDIR`, so `polykernel-opt`
  lands at `build/tools/polykernel-opt/polykernel-opt` (the gate's expected path).

### Dialect (TableGen)

`include/PolyKernel/IR/CMakeLists.txt` uses direct `mlir_tablegen` (sanctioned by
the brief alongside `add_mlir_dialect`) so the generated files are named
`PolyKernelDialect.{h,cpp}.inc`:

```cmake
set(LLVM_TARGET_DEFINITIONS PolyKernelDialect.td)
mlir_tablegen(PolyKernelDialect.h.inc   -gen-dialect-decls -dialect=polykernel)
mlir_tablegen(PolyKernelDialect.cpp.inc -gen-dialect-defs  -dialect=polykernel)
add_public_tablegen_target(MLIRPolyKernelDialectIncGen)
```

`PolyKernelDialect.td` declares **only** the dialect (`name = "polykernel"`,
`cppNamespace = "::mlir::polykernel"`) — **no ops/types/attrs** (those are Todo 3).

### Dialect library

`lib/IR/CMakeLists.txt`:

```cmake
add_mlir_dialect_library(MLIRPolyKernel
  ${POLYKERNEL_SOURCE_DIR}/include/PolyKernel/IR/PolyKernelDialect.cpp
  DEPENDS MLIRPolyKernelDialectIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSupport)
```

### `polykernel-opt`

`tools/polykernel-opt/`: `DialectRegistry registry; registry.insert<mlir::polykernel::PolyKernelDialect>();`
then `mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "polykernel-opt\n", registry))`.
Linked with `add_llvm_executable` + `target_link_libraries(... MLIRPolyKernel MLIROptLib MLIRIR MLIRSupport)`
+ `mlir_check_all_link_libraries(polykernel-opt)`.

### lit

`test/lit.site.cfg.py.in` substitutes `LLVM_TOOLS_BINARY_DIR` (FileCheck/count/not),
`MLIR_BINARY_DIR`, `POLYKERNEL_BINARY_DIR/SOURCE_DIR`; `test/lit.cfg.py` (ShTest,
`.mlir` suffix) adds tool substitutions for `polykernel-opt` (from
`build/tools/polykernel-opt`) + `FileCheck/count/not` (from the LLVM tools dir) via
`lit.llvm`'s `add_tool_substitutions`. `test/CMakeLists.txt` wires `check-polykernel`
with `add_lit_testsuite(... DEPENDS polykernel-opt)` using `LLVM_EXTERNAL_LIT`.
`test/smoke.mlir` round-trips a builtin `module {}`.

## nix-specific deviations from the upstream standalone example (important for Todo 3+)

1. **No `registerAllDialects`/`registerAllPasses`.** nixpkgs builds MLIR with
   `MLIR_INSTALL_AGGREGATE_OBJECTS=1` and does **not** install
   `libMLIRRegisterAll{Dialects,Passes}.a` (verified absent from every mlir-21.1.8
   store output). Linking them fails with `cannot find -lMLIRRegisterAllDialects`.
   The spike therefore registers only the PolyKernel dialect. When upstream
   dialects/passes are needed (Todo 3+), link the **individual** libs nix DOES ship
   (e.g. `MLIRArithDialect`, `MLIRFuncDialect`, specific pass libs) rather than the
   aggregate `RegisterAll*`.
2. **`registerAllPasses()` takes no arguments** in LLVM 21 (the brief's
   `registerAllPasses(registry)` does not match the signature); `registerAllDialects`
   takes the registry. Moot for now per (1), but recorded for later.
3. **`FileCheck`/`count`/`not` are not CMake targets** out-of-tree, so they must not
   appear in the lit `DEPENDS` (only `polykernel-opt` does); they are resolved at lit
   runtime from `LLVM_TOOLS_BINARY_DIR`.

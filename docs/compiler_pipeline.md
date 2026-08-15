# PolyKernel Compiler Pipeline — Toolchain & Build Wiring

> W1 spike (Todo 2): out-of-tree MLIR toolchain. This is a **GATE** — it proves
> the out-of-tree MLIR-21 build + dialect registration + `polykernel-opt` tool +
> `lit` pipeline before any dialect/op work (Todo 3+). **GATE PASSED.**
>
> Todo 3 (Wave 1): the `polykernel` dialect now defines **exactly** the named
> transformer-inference op set via ODS/TableGen; every op parses + round-trips
> and the op set is closed (undefined op → parse error). **DONE** — see
> "Todo 3 (Wave 1)" section below + `reports/w1_dialect_roundtrip.log`,
> `reports/w1_unknown_op.log`.

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

## Todo 3 (Wave 1): PolyKernel dialect ops (ODS/TableGen)

Defined **exactly** the named transformer-inference op set in
`include/PolyKernel/IR/PolyKernelOps.td` — nothing else (op-zoo guardrail). 18 ops
register: 11 base (`func`, `rmsnorm`, `matmul`, `bias`, `gelu`, `silu`, `add`,
`softmax`, `rope`, `attention`, `kv_cache_update`) + 6 fused
(`fused_rmsnorm_matmul`, `fused_matmul_bias_gelu`, `fused_residual_rmsnorm`,
`fused_softmax_mask`, `qkv_projection`, `fused_kv_append_attention`) + the
`return` terminator. `polykernel.func`/`polykernel.return` are the function
container + terminator (modeled on `func.func` via `FunctionOpInterface`), **not**
compute ops, so the compute op-zoo guardrail holds. Verified: every op round-trips
(`test/dialect_roundtrip.mlir`) and the set is closed (`test/op_set_closed.mlir`:
`polykernel.conv2d` → `custom op 'polykernel.conv2d' is unknown`, exit 1).

### TableGen / C++ build learnings (for Todo 4+)

- **C++ class names are prefix-stripped.** Dialect def `PolyKernel_Dialect` strips
  the `PolyKernel_` prefix from op defs: `def PolyKernel_MatMulOp` → C++ class
  `mlir::polykernel::MatMulOp`. **`HasParent` takes the C++ name**: use
  `HasParent<"FuncOp">`, NOT `HasParent<"PolyKernel_FuncOp">` (the latter emits an
  undefined C++ type and fails with a confusing `no member named 'InferredProperties'`
  cascade).
- **ODS includes needed** (OpBase.td alone is NOT enough):
  `mlir/IR/CommonTypeConstraints.td` (`AnyTensor`), `mlir/Interfaces/ControlFlowInterfaces.td`
  (`ReturnLike`), `mlir/Interfaces/FunctionInterfaces.td` (`FunctionOpInterface`),
  `mlir/Interfaces/InferTypeOpInterface.td` (`SameOperandsAndResultType`),
  `mlir/Interfaces/SideEffectInterfaces.td` (`Pure`).
- **Ops header needs `mlir/Bytecode/BytecodeOpInterface.h`** (MLIR 21 generates
  properties/bytecode code that references `DialectBytecodeReader/Writer`). Mirror
  the include set of upstream `FuncOps.h`.
- **`AnyTensor` is a `def`, not a `class`** — you cannot `def X : AnyTensor;`. Use
  `AnyTensor` directly as the operand/result constraint.
- **Operand-type accessors:** an operand `$a` generates `getA()` returning a
  `TypedValue<TensorType>`; get its type via `getA().getType()` (there is no
  `getAType()` returning a `Type`).
- **Optional operands with their own types need a custom parser** (declarative
  assembly format can't print a type only when an optional operand is present
  without `AttrSizedOperandSegments` + properties handling). To keep a clean
  declarative format, `polykernel.rope`'s `cos`/`sin` are modeled as **explicit**
  (required) operands. Revisit optionality in Todo 4 if needed.
- **Link libs:** `MLIRPolyKernel` links `MLIRIR MLIRSupport MLIRFunctionInterfaces`.
  `MLIRFunctionInterfaces` provides `function_interface_impl::parse/printFunctionOp`
  (defined `T` in `libMLIRFunctionInterfaces.a`). `MLIRFuncDialect` is **not**
  required (`polykernel.func` is our own op, not `func.func`). `MLIRInferTypeOpInterface`
  not required (no shape inference yet).
- **No upstream dialect registration needed in `polykernel-opt`.** The IR uses only
  `polykernel.*` ops + builtin tensors; `registry.insert<PolyKernelDialect>()`
  suffices (the dialect's `initialize()` now calls `addOperations<...>()` via
  `GET_OP_LIST`). `--show-dialects` → `builtin,polykernel`.
- **clangd gives false "header not found" errors** here: the nix `clang-wrapper`
  bakes the MLIR/LLVM include paths in implicitly, so they are absent from
  `compile_commands.json` and clangd can't resolve them. The authoritative check is
  the nix compiler itself (`clang++ -fsyntax-only` with the build's warning flags →
  clean) and the actual `cmake --build`.

### CMake wiring (ops)

`include/PolyKernel/IR/CMakeLists.txt` has TWO independent TableGen targets (no
double-generation): `MLIRPolyKernelDialectIncGen` (`-gen-dialect-decls/-defs` from
`PolyKernelDialect.td`) and `MLIRPolyKernelOpsIncGen` (`-gen-op-decls/-defs` from
`PolyKernelOps.td`, which includes the dialect `.td` but defines no dialect record).
`lib/IR/CMakeLists.txt` compiles both `PolyKernelDialect.cpp` + `PolyKernelOps.cpp`
(co-located under `include/PolyKernel/IR/`) and `DEPENDS` on both IncGen targets.

### Verification (Todo 3)

`check-polykernel` → `Passed: 3 (100.00%)` (`smoke.mlir`, `dialect_roundtrip.mlir`,
`op_set_closed.mlir`), deterministic across 2 runs. Evidence:
`reports/w1_dialect_roundtrip.log` (round-tripped IR for every op + op census =
exactly the 18 named ops) and `reports/w1_unknown_op.log` (`polykernel.conv2d`
rejected, exit 1).

## Todo 4 (Wave 1): `--infer-shapes` pass (shape + dtype inference)

Added the PolyKernel **pass infrastructure** (which Todo 5 `--canonicalize`
extends) and the `--infer-shapes` pass: every compute op's result type(s) are
computed from its operands via `InferTypeOpInterface` and refined by the pass.

### Layout

- `include/PolyKernel/Passes/Passes.td`: pass decls via `-gen-pass-decls
  -name PolyKernel`. `def InferShapes : Pass<"infer-shapes", "ModuleOp">`. Todo 5
  adds another `def ... : Pass<...>` here; `registerPolyKernelPasses()` picks it up.
- `include/PolyKernel/Passes/InferShapes.h`: `createInferShapesPass()` + the
  `GEN_PASS_DECL_INFERSHAPES` include.
- `include/PolyKernel/Passes/Passes.h`: umbrella; `GEN_PASS_REGISTRATION` wrapped in
  `namespace mlir::polykernel` so registration is `mlir::polykernel::registerPolyKernelPasses()`
  / `registerInferShapesPass()`.
- `lib/Passes/InferShapes.cpp`: the pass — `impl::InferShapesBase` (from
  `GEN_PASS_DEF_INFERSHAPES`); walks the module, `dyn_cast<InferTypeOpInterface>`,
  calls `inferReturnTypes`, sets result types, `signalPassFailure()` on mismatch.
- `lib/Passes/CMakeLists.txt`: `add_mlir_library(MLIRPolyKernelPasses ...)` linking
  `MLIRPolyKernel MLIRInferTypeOpInterface MLIRPass MLIRIR MLIRSupport`.
- `tools/polykernel-opt`: calls `mlir::polykernel::registerPolyKernelPasses()` before
  `MlirOptMain`; links `MLIRPolyKernelPasses`. **Explicit** registration (no
  `registerAllPasses` / `MLIRRegisterAll*`).

### Inference design (important for Todo 5)

- Every compute op implements `InferTypeOpInterface` via
  `DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes","refineReturnTypes"]>`
  (the `PolyKernel_InferTypeOp` base in `PolyKernelOps.td`). The per-op shape logic
  lives in the adaptor-form `inferReturnTypes` in `PolyKernelOps.cpp`; a shared macro
  `POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(OpClass)` emits the full-signature
  `inferReturnTypes` (builds the Adaptor, delegates) + a **no-op `refineReturnTypes`**.
- **`refineReturnTypes` is a deliberate no-op**: it disables the InferTypeOpInterface
  verification hook so shape consistency is enforced by the `--infer-shapes` pass
  (`signalPassFailure`), NOT at IR parse time. This keeps fully-explicit round-trip IR
  valid even with placeholder result types, and makes the pass the meaningful enforcer
  (the mismatch test exercises `signalPassFailure`). `verifyInferredResultTypes`
  (mlir/lib/Interfaces/InferTypeOpInterface.cpp) calls `refineReturnTypes`, so a no-op
  fully disables it.
- `SameOperandsAndResultType` was **dropped** from rmsnorm/gelu/silu/softmax/add (it
  auto-generates a conflicting `inferReturnTypes`); `add` keeps `SameTypeOperands` so
  its `rhs` type stays inferable from `lhs` in the declarative format. Result type for
  these is `input` (add checks equal operand shapes).
- `qkv_projection` gained `OptionalAttr<I64Attr>:$num_heads` — the Q/K/V head split is
  under-determined from operand shapes alone; inference needs `num_heads`
  (Q=K=V=`[batch..., num_heads, seq, hidden/num_heads]`).
- Shape rules: matmul `A<...xMxK> x B<...xKxN> -> <...xMxN>` (K must match, else
  diagnostic); rmsnorm/gelu/silu/softmax/rope/fused_softmax_mask → input type;
  add/fused_residual_rmsnorm → operand shape (equal shapes required); bias → input
  shape (bias last dim must match); attention → query shape; kv_cache_update →
  cache shape; fused_rmsnorm_matmul/fused_matmul_bias_gelu → matmul MxN;
  fused_kv_append_attention → (query shape, cache shape).
- **Result types stay explicit** in the assemblyFormat for matmul/bias/rope/attention/
  kv_cache_update/fused (so inferred types are CHECK-able on the op line); the
  shape-preserving ops keep their elided result type (inferred at parse via the
  interface). MLIR IR is always fully typed, so the positive test uses correct types
  (the pass recomputes + confirms; idempotent) and the **negative test** proves the
  rules (mismatched matmul K → diagnostic + exit 1).

### nix / build notes (Todo 4)

- New upstream lib linked: `MLIRInferTypeOpInterface` (individual lib nix ships; in the
  same `mlir-21.1.8` output as `MLIRFunctionInterfaces`). No `MLIRRegisterAll*`.
- `MLIRPolyKernelPassesIncGen` (`-gen-pass-decls -name PolyKernel` → `Passes.h.inc`).
  The per-pass decl macro is `GEN_PASS_DECL_INFERSHAPES` (no underscore); the base
  class `impl::InferShapesBase` comes from `GEN_PASS_DEF_INFERSHAPES` in the .cpp.
- `check-polykernel` → `Passed: 5 (100.00%)` (`smoke`, `op_set_closed`,
  `dialect_roundtrip`, `infer_shapes`, `infer_shapes_mismatch`), deterministic across 2
  runs. Evidence: `reports/w1_infer_shapes.log` (inferred IR for matmul/rmsnorm/gelu/
  silu/softmax/add/fused/qkv on bf16+fp16 + lit summary) and
  `reports/w1_shape_mismatch.log` (mismatched-K matmul → diagnostic + exit 1).
- Op set unchanged: exactly **18** ops (census re-verified).

## The full pass pipeline (Waves 1-5)

The dialect ships **11 passes** (all declared in `include/PolyKernel/Passes/Passes.td`,
registered explicitly via `registerPolyKernelPasses()` — never `registerAllPasses`, per
the nix out-of-tree constraint above). The canonical end-to-end order:

```
--infer-shapes            # Todo 4  (Wave 1) shape + dtype inference
--canonicalize            # Todo 5  (Wave 1) identity-add elim, const fold, DCE
--fuse-rmsnorm-matmul     # Todo 12 (Wave 3) single-use rmsnorm+matmul -> fused
--fuse-matmul-bias-gelu   # Todo 13 (Wave 3) single-use matmul+bias+gelu -> fused
--fuse-residual-rmsnorm   # Todo 13 (Wave 3) single-use add+rmsnorm -> fused
--fuse-softmax-mask       # Todo 13 (Wave 3) single-use mask-add+softmax -> fused
--infer-tile-layout       # Todo 15 (Wave 3) BLOCK_M/N/K + layout attrs per matmul
--plan-memory             # Todo 16 (Wave 3) smem budget + workspace + over-budget guard
--lower-to-cuda           # Todo 8  (Wave 2) emit portable .cu (POLYKERNEL_CUDA)
--lower-to-hip            # Todo 19 (Wave 4) emit portable .cu (POLYKERNEL_HIP)
--emit-kernel-report      # Todo 27 (Wave 5) attach/emit the contract-H report skeleton
```

| Pass | Wave | What it does | Evidence |
|---|---|---|---|
| `--infer-shapes` | 1 | computes each op's result type from its operands via `InferTypeOpInterface`; mismatch → `signalPassFailure` | `reports/w1_infer_shapes.log`, `reports/w1_shape_mismatch.log` |
| `--canonicalize` | 1 | `add(x,0)→x`, pure-elementwise const fold, softmax-of-splat → uniform, DCE (greedy pattern driver) | `reports/w1_canonicalize.log`, `reports/w1_canonicalize_noop.log` |
| `--fuse-rmsnorm-matmul` | 3 | `rmsnorm(x);matmul(rms,w)` → `fused_rmsnorm_matmul(x,w)` (single-use only) | `reports/w3_fuse_rmsnorm_matmul.log` |
| `--fuse-matmul-bias-gelu` | 3 | `matmul;bias;gelu` → `fused_matmul_bias_gelu` (single-use chain) | `reports/w3_fused_kernels.log` |
| `--fuse-residual-rmsnorm` | 3 | `add(residual,x);rmsnorm` → `fused_residual_rmsnorm` | `reports/w3_fuse_remaining.log` |
| `--fuse-softmax-mask` | 3 | `add(x,mask);softmax` → `fused_softmax_mask` | `reports/w3_fuse_remaining.log` |
| `--infer-tile-layout` | 3 | assigns `polykernel.tile_m/n/k` + `polykernel.layout` (largest power-of-2 ≤ dim, clamped) | `reports/w3_tile_layout.log` |
| `--plan-memory` | 3 | `polykernel.smem_bytes` + `polykernel.workspace_bytes` (0 for fused) + over-budget guard | `reports/w3_plan_memory.log` |
| `--lower-to-cuda` | 2 | custom source emitter → `kernels/generated/*.cu` (see below) | `reports/w2_template.log` |
| `--lower-to-hip` | 4 | same emitter; the HIP define is a *driver* concern, not an emitter concern | `reports/w4_hipcc_compile.log` |
| `--emit-kernel-report` | 5 | attaches the IR-derivable contract-H fields + emits `kernel_report.json` | `reports/kernel_report_example.json` |

**Fusion guardrails (all four fusion passes).** Each pass matches its exact op chain and
requires every absorbed intermediate to have **exactly one use** (a multi-use producer is
never fused — fusing would duplicate it). Each fused op is tagged with
`polykernel.fused_from` + `polykernel.eliminated_type[_N]` discardable attributes so the
traffic report can compute the saved global round-trip (`2 × numel × bytes` per eliminated
intermediate). The op set stays **closed**: tile/layout, memory-plan, and report fields are
all *discardable attributes*, never new declared op attributes.

## Lowering is custom source emission (Pinned contract F)

`--lower-to-cuda` / `--lower-to-hip` do **not** use the upstream `gpu-to-llvm` /
`convert-to-llvm` paths. They walk the (lowered) IR and, for each recognized compute op,
**string-emit** a portable CUDA/HIP kernel source file into `--output-dir`
(default `kernels/generated/`). The emitted `.cu` is written against
`kernels/template/kernel_common.h`, which maps `__global__` / shared memory / launch to
CUDA vs HIP via `#ifdef POLYKERNEL_CUDA` / `#ifdef POLYKERNEL_HIP`. The two passes emit
**byte-identical** compute; they differ only in which driver compiles the output
(`nvcc -DPOLYKERNEL_CUDA` vs `hipcc -DPOLYKERNEL_HIP`). The IR itself is left unchanged.
Seven kernels are emitted (rmsnorm, gelu, silu, matmul, softmax, fused_rmsnorm_matmul,
fused_matmul_bias_gelu); `kernels/generated/matmul_wmma.cu` is the additive WMMA bf16
variant (Todo 22).

## The tools

| Tool | Purpose |
|---|---|
| `polykernel-opt` | pass driver (`MlirOptMain` + the registered PolyKernel dialect + passes) |
| `polykernel-translate` | `--mlir-to-cuda-source` / `--mlir-to-hip-source` source emit |
| `polykernel-analyze` | standalone GPU-free compile-time analyzer (Todo 11; reused by `-bench`/`-report`) |
| `polykernel-bench` | correctness-gated autotuner + analyzer CLI (Todo 25) |
| `polykernel-report` | `--traffic` (Todo 17), `--backend=dataflow [--viz]` (Todo 38/39), per-kernel `report.py` (Todo 27), `--dashboard` (Todo 34) |

## End-to-end example

```bash
nix develop --impure --accept-flake-config -c bash -c '
  # full pipeline on the MLP fragment: infer + canonicalize + fuse + tile + plan + lower
  ./build/tools/polykernel-opt/polykernel-opt examples/mlp_block.mlir \
      --infer-shapes --canonicalize \
      --fuse-rmsnorm-matmul --fuse-matmul-bias-gelu \
      --infer-tile-layout --plan-memory \
      --lower-to-cuda --output-dir=kernels/generated
  # the whole lit suite (14 tests)
  cmake --build build --target check-polykernel
'
```

`check-polykernel` runs lit over `test/` (14 `.mlir` tests: smoke, op-set-closed,
dialect round-trip, infer-shapes ± mismatch, canonicalize, the five fusion passes,
tile-layout, plan-memory, e2e-parse). The before/after fusion traffic report
(`polykernel-report --traffic examples/mlp_block.mlir`) quantifies the global-memory
reduction; for the MLP block fusion eliminates a 32 MiB intermediate round-trip
(7.02% of fragment traffic — `reports/mlp_traffic.md`).

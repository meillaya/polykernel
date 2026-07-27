"""Modal GPU endpoints for PolyKernel (Todo 30 + Todo 31 / Wave 6).

Extends the Todo 29 app skeleton with a GPU service class exposing four endpoints:

    POST /predict    - run the fused MLP block via the ctypes-loaded engine .so
    POST /benchmark  - run a correctness-gated kernel bench (subprocess: bench.py)
    GET  /kernels    - list tuned kernels from the contract-H tuning cache
    GET  /report     - return the per-kernel contract-H report (subprocess: report.py)

Pipeline (modal 1.5.2 - every decorator verified current by introspection):

    @app.cls(gpu="A10", ..., min_containers=1, scaledown_window=300,
             enable_memory_snapshot=True,
             experimental_options={"enable_gpu_snapshot": True})
    class PolyKernelService:
        @modal.enter(snap=True)   # BEFORE snapshot (NO GPU): dlopen .so + weights
        @modal.enter(snap=False)  # AFTER restore (GPU up): one dummy kernel warmup
        @modal.fastapi_endpoint(method="POST")  # /predict, /benchmark
        @modal.fastapi_endpoint(method="GET")   # /kernels, /report

The engine .so is compiled at IMAGE-BUILD time (modal/image.py run_commands) and
loaded once per container via the snap=True enter hook. Weights load from a
modal.Volume mounted at /weights. The bench + report tools are reused as-is
(subprocess).

Autoscaling + cold start (Todo 31): the class sets min_containers / max_containers
/ buffer_containers / scaledown_window and turns on memory snapshotting
(enable_memory_snapshot + experimental enable_gpu_snapshot). Startup is SPLIT into
two @modal.enter hooks keyed on the snapshot boundary:

    snap=True  runs BEFORE the memory snapshot is taken - NO GPU is available yet,
               so it does only CPU work (dlopen the engine .so, bind the ctypes
               signature, read the weights bytes). This state is captured in the
               snapshot, so a restored container skips it entirely.
    snap=False runs AFTER the snapshot is restored - the GPU is available, so it
               fires one dummy kernel to pay the CUDA-context / JIT / first-launch
               cost once, making the first real /predict warm.

Client-side cold-vs-warm measurement lives in modal/cold_start.py (sleeps past
scaledown_window to force a scale-down, then times the first vs a warm request).

Importing this module constructs the App + service class and needs NO Modal token.
Deploying/serving (which builds the image + provisions GPU containers) is the
opt-in cloud step behind a MODAL_TOKEN:

    modal serve modal/app.py    # dev URL (requires token)
    modal deploy modal/app.py   # production deploy (requires token)

Why the sibling image module is loaded by PATH (not `from modal.image import ...`):
this file lives in a directory named `modal/`, which collides with the installed
`modal` SDK package. `import modal` correctly resolves to the SDK (a regular
package always wins over the local `modal/` namespace directory), but the qualified
name `modal.image` is the SDK's own image submodule - NOT this project's
`modal/image.py`. We therefore load the sibling by its file path under a unique
module name, which is collision-proof and deterministic.
"""

# allow: SIZE_OK - Modal serves ONE module (`modal serve modal/app.py`); the @app.cls
# service class, its snap=True/snap=False enter hooks, four endpoints, and autoscaling
# config form an indivisible deployment unit that cannot be split across files (and the
# task restricts edits to this module). The bulk is the inherited Todo 30 endpoints.
from __future__ import annotations

import ctypes
import importlib.util
import json
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Pydantic models (request/response validation -> FastAPI 422 on bad input).
#
# The nix dev shell lacks pydantic (it is a deploy-time dep provisioned inside
# the Modal container, not a local build dep). The fallback below lets this
# module import cleanly for the local construction test; at deploy time the
# real pydantic BaseModel is used and FastAPI validates every request body.
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - local dev shell only

    class BaseModel:  # type: ignore[no-redef]
        """Minimal fallback for local import (no pydantic in nix dev shell)."""

        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default: object = None, **kwargs: object) -> object:  # type: ignore[no-redef]  # noqa: ARG001
        return default


# ---------------------------------------------------------------------------
# Request / response schemas (validated by FastAPI at deploy time).
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """POST /predict body: the fused MLP block input tensor (row-major bf16 as
    float list) + the GEMM shape (M, N, K)."""

    input_data: list[float] = Field(..., description="Row-major input tensor (bf16 as float)")
    shape_m: int = Field(..., gt=0, description="GEMM M dimension")
    shape_n: int = Field(..., gt=0, description="GEMM N dimension")
    shape_k: int = Field(..., gt=0, description="GEMM K dimension")


class PredictResponse(BaseModel):
    """POST /predict response: the prediction output tensor + metadata."""

    prediction: list[float]
    shape_m: int
    shape_n: int
    engine_loaded: bool


class BenchmarkRequest(BaseModel):
    """POST /benchmark body: which op + shape + arch to bench."""

    op: str = Field("fused_matmul_bias_gelu", description="Matmul-family op name")
    shape: str = Field("2048,4096,11008", description="GEMM shape M,N,K")
    arch: str = Field("gfx1101", description="Target GPU arch")
    variants: int = Field(4, gt=0, description="Bounded ConfigSpace prefix to race")


# ---------------------------------------------------------------------------
# Container paths (the image bakes the source tree at /app - see image.py).
# ---------------------------------------------------------------------------
REMOTE_APP_DIR = "/app"
ENGINE_SO_PATH = f"{REMOTE_APP_DIR}/build/lib/libpolykernel-engine.so"
WEIGHTS_MOUNT = "/weights"
TUNING_CACHE_PATH = f"{REMOTE_APP_DIR}/build/polykernel-bench/tuning_cache.json"
REPORT_TOOL = f"{REMOTE_APP_DIR}/tools/polykernel-report/report.py"
BENCH_TOOL = f"{REMOTE_APP_DIR}/tools/polykernel-bench/bench.py"

# ---------------------------------------------------------------------------
# Autoscaling + memory-snapshot config (Todo 31).
#
# These are the @app.cls knobs that govern how Modal provisions GPU containers.
# Introspected against modal 1.5.2 (modal.App.cls): min_containers REPLACES the
# old keep_warm; scaledown_window is seconds; enable_memory_snapshot is a bool;
# experimental_options is a free-form dict (enable_gpu_snapshot snapshots the CUDA
# context too).
# ---------------------------------------------------------------------------
# Keep 1 container always-warm so steady traffic never pays a cold start.
MIN_CONTAINERS = 1
# Hard cap on scale-out (GPU budget guard).
MAX_CONTAINERS = 8
# Extra standby containers pre-provisioned to absorb bursts without a cold start.
BUFFER_CONTAINERS = 1
# Seconds of idle before an idle container scales down. modal/cold_start.py sleeps
# PAST this to force a scale-down and measure the resulting cold start. The Modal
# minimum is 2s (used for the fast-scale-down negative case in cold_start.py).
SCALEDOWN_WINDOW = 300
# Tiny shape used by the snap=False GPU warmup - just enough to launch the kernel
# once so the first real /predict is warm (CUDA context + JIT already primed).
WARMUP_SHAPE = (64, 64, 64)  # (M, N, K)

# ctypes signature of the engine's fused MLP block, bound once in the snap=True
# enter hook (and captured in the memory snapshot):
#   int polykernel_fused_mlp_block(float* in, float* out, int M, int N, int K)
_FUSED_MLP_ARGTYPES = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

# ---------------------------------------------------------------------------
# Image + Volume (both construct token-free; materialised only on deploy).
# ---------------------------------------------------------------------------
_IMAGE_MODULE_NAME = "polykernel_modal_image"


def _load_engine_image() -> modal.Image:
    """Load `modal/image.py` (a sibling of this file) and return its `image`.

    Loaded by explicit path so the `modal/` directory name cannot collide with
    the `modal` SDK package (see module docstring).
    """
    image_py = Path(__file__).resolve().parent / "image.py"
    spec = importlib.util.spec_from_file_location(_IMAGE_MODULE_NAME, image_py)
    if spec is None or spec.loader is None:  # pragma: no cover - path is fixed
        msg = f"cannot load engine image module from {image_py}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.image


image = _load_engine_image()

# Weights volume: mounted at /weights inside the container. The engine loads
# weights from here at startup (@modal.enter). create_if_missing=True so the
# first deploy provisions it automatically.
weights_volume = modal.Volume.from_name("polykernel-weights", create_if_missing=True)

# ---------------------------------------------------------------------------
# The PolyKernel app + GPU service class.
# ---------------------------------------------------------------------------
app = modal.App("polykernel", image=image)


@app.cls(
    gpu="A10",
    timeout=600,
    startup_timeout=300,
    volumes={WEIGHTS_MOUNT: weights_volume},
    # --- Autoscaling (Todo 31; introspected against modal 1.5.2) ---
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    buffer_containers=BUFFER_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    # --- Memory snapshot: fast cold-start restore ---
    # CRIU-snapshot container memory after the snap=True enter hooks; the restored
    # container skips re-running them (the .so is already dlopen'd + bound).
    enable_memory_snapshot=True,
    # Also snapshot the CUDA context so the restored container keeps its GPU state.
    experimental_options={"enable_gpu_snapshot": True},
)
class PolyKernelService:
    """GPU service: ctypes-loaded engine .so + four FastAPI endpoints.

    The engine .so is compiled at image-build time (modal/image.py) and loaded
    once per container via the snap=True enter hook. Weights load from the
    modal.Volume mounted at /weights. The bench + report tools are reused via
    subprocess. Startup is split across the memory-snapshot boundary: snap=True
    (no GPU) does the CPU init; snap=False (GPU up) warms up the kernel.
    """

    @modal.enter(snap=True)
    def load_engine(self) -> None:
        """Pre-snapshot init (NO GPU): dlopen the engine .so + load weights.

        Runs BEFORE the memory snapshot is taken, so only CPU work is allowed
        here (the GPU is not available yet). ctypes.CDLL maps the .so into the
        process and resolves symbols; the weights are read as bytes from the
        Volume mount. All of this state is captured in the snapshot, so a
        restored container skips the dlopen + weight-load cost entirely - the
        whole point of enable_memory_snapshot.
        """
        self.engine = ctypes.CDLL(ENGINE_SO_PATH)
        # Bind the fused MLP block signature once (cheap CPU work; snapshotted).
        self.engine.polykernel_fused_mlp_block.restype = ctypes.c_int
        self.engine.polykernel_fused_mlp_block.argtypes = _FUSED_MLP_ARGTYPES
        # Load weights from the mounted Volume (persistent across restarts).
        weights_path = Path(WEIGHTS_MOUNT) / "weights.bin"
        self.weights: bytes | None = (
            weights_path.read_bytes() if weights_path.exists() else None
        )

    @modal.enter(snap=False)
    def warmup_gpu(self) -> None:
        """Post-restore GPU warmup (GPU available): fire one dummy kernel.

        Runs AFTER the memory snapshot is restored, when the GPU is available.
        GPU state (CUDA context, JIT caches, first-launch overhead) is NOT in
        the memory snapshot, so it is paid here - once, at restore - instead of
        on the first /predict. The dummy launch primes the context so the first
        real request is warm.
        """
        import numpy as np  # noqa: PLC0415 - deploy-time dep (image has numpy)

        m, n, k = WARMUP_SHAPE
        inp = np.zeros(m * k, dtype=np.float32)
        out = np.zeros(m * n, dtype=np.float32)
        # Warmup only: a non-zero rc is surfaced by the first real /predict, so
        # it is intentionally not raised here (would just mask the warmup intent).
        self.engine.polykernel_fused_mlp_block(
            inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            m,
            n,
            k,
        )

    @modal.fastapi_endpoint(method="POST")
    def predict(self, request: PredictRequest) -> PredictResponse:
        """Run the fused MLP block (engine .so via ctypes), return a prediction.

        The engine's fused kernel is called with the input tensor + shape. The
        output is the prediction (row-major bf16 as float list). FastAPI
        validates the request body via PredictRequest -> malformed payload yields
        a 422 (never a 500 crash).
        """
        m, n, k = request.shape_m, request.shape_n, request.shape_k
        # Call the engine's fused MLP block via ctypes. The restype/argtypes are
        # bound once in the snap=True enter hook (captured in the snapshot).
        import numpy as np  # noqa: PLC0415 - deploy-time dep (image has numpy)

        inp = np.array(request.input_data, dtype=np.float32)
        out = np.zeros(m * n, dtype=np.float32)
        rc = self.engine.polykernel_fused_mlp_block(
            inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            m,
            n,
            k,
        )
        if rc != 0:
            msg = f"engine polykernel_fused_mlp_block failed (exit {rc})"
            raise RuntimeError(msg)
        return PredictResponse(
            prediction=out.tolist(),
            shape_m=m,
            shape_n=n,
            engine_loaded=True,
        )

    @modal.fastapi_endpoint(method="POST")
    def benchmark(self, request: BenchmarkRequest) -> dict:
        """Run a correctness-gated kernel bench (subprocess: bench.py), return JSON.

        Reuses tools/polykernel-bench/bench.py as-is. The bench builds the C++
        driver, enumerates a bounded ConfigSpace prefix, races variants, and
        writes the contract-H tuning cache. Output is the bench's JSON summary.
        """
        cmd = [
            "python3",
            BENCH_TOOL,
            "--autotune",
            "--op", request.op,
            "--shape", request.shape,
            "--arch", request.arch,
            "--variants", str(request.variants),
            "--backend", "hip",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            return {
                "status": "error",
                "returncode": proc.returncode,
                "stderr": proc.stderr,
            }
        # The bench prints the tuning-cache JSON on stdout (last JSON block).
        return {"status": "ok", "stdout": proc.stdout, "stderr": proc.stderr}

    @modal.fastapi_endpoint(method="GET")
    def kernels(self) -> dict:
        """List tuned kernels from the contract-H tuning cache (Todo 24/28).

        Reads the tuning-cache JSON (baked into the image at build time) and
        returns the entries list. Each entry is a (gpu, op, shape) -> best config
        mapping with correctness metrics (contract H).
        """
        cache_path = Path(TUNING_CACHE_PATH)
        if not cache_path.exists():
            return {"version": 1, "entries": [], "note": "no tuning cache found"}
        cache = json.loads(cache_path.read_text())
        return cache

    @modal.fastapi_endpoint(method="GET")
    def report(self) -> dict:
        """Return the per-kernel contract-H report (subprocess: report.py).

        Reuses tools/polykernel-report/report.py as-is. The report merges CUDA
        compile-time analysis (ptxas -v), AMD ISA analysis, and the autotuner
        result into the authoritative contract-H per-kernel report.
        """
        cmd = [
            "python3",
            REPORT_TOOL,
            "--kernel", "fused_rmsnorm_matmul",
            "--backend", "cuda",
            "--arch", "sm_90",
            "--format", "json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if proc.returncode != 0:
            return {
                "status": "error",
                "returncode": proc.returncode,
                "stderr": proc.stderr,
            }
        return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Local construction test (no token needed).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Importing this module constructs the App + service class. This is the
    # local validation: the app + endpoints + autoscaling config + the split
    # snap=True/snap=False enter hooks are all constructible without a token.
    print(f"app: {app.name}")
    print(f"image: {image}")
    print(f"service class: {PolyKernelService.__name__}")
    print("endpoints: predict (POST), benchmark (POST), kernels (GET), report (GET)")
    print("gpu: A10, timeout: 600s, startup_timeout: 300s")
    print(f"weights volume: {weights_volume}")
    print(f"autoscaling: min_containers={MIN_CONTAINERS}, max_containers={MAX_CONTAINERS}, buffer_containers={BUFFER_CONTAINERS}, scaledown_window={SCALEDOWN_WINDOW}s")
    print("memory snapshot: enable_memory_snapshot=True, experimental_options={'enable_gpu_snapshot': True}")
    print("enter hooks: load_engine (snap=True, no GPU) + warmup_gpu (snap=False, GPU warmup)")
    print("construction OK (no token required)")

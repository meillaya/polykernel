"""Modal GPU endpoints for PolyKernel (Todo 30 / Wave 6).

Extends the Todo 29 app skeleton with a GPU service class exposing four endpoints:

    POST /predict    - run the fused MLP block via the ctypes-loaded engine .so
    POST /benchmark  - run a correctness-gated kernel bench (subprocess: bench.py)
    GET  /kernels    - list tuned kernels from the contract-H tuning cache
    GET  /report     - return the per-kernel contract-H report (subprocess: report.py)

Pipeline (modal 1.5.2 - every decorator verified current by introspection):

    @app.cls(gpu="A10", timeout=600, startup_timeout=300, volumes={...})
    class PolyKernelService:
        @modal.enter()          # ctypes.CDLL the engine .so + load Volume weights
        @modal.fastapi_endpoint(method="POST")  # /predict, /benchmark
        @modal.fastapi_endpoint(method="GET")   # /kernels, /report

The engine .so is compiled at IMAGE-BUILD time (modal/image.py run_commands) and
loaded once per container via @modal.enter(). Weights load from a modal.Volume
mounted at /weights. The bench + report tools are reused as-is (subprocess).

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
)
class PolyKernelService:
    """GPU service: ctypes-loaded engine .so + four FastAPI endpoints.

    The engine .so is compiled at image-build time (modal/image.py) and loaded
    once per container via @modal.enter(). Weights load from the modal.Volume
    mounted at /weights. The bench + report tools are reused via subprocess.
    """

    @modal.enter()
    def load_engine(self) -> None:
        """Startup hook: ctypes-load the engine .so + load weights from Volume.

        Runs once when the container starts (before any endpoint is served).
        The engine .so exports the fused MLP block kernel; weights are loaded
        from the Volume mount so they persist across container restarts.
        """
        self.engine = ctypes.CDLL(ENGINE_SO_PATH)
        # Load weights from the mounted Volume (persistent across restarts).
        weights_path = Path(WEIGHTS_MOUNT) / "weights.bin"
        self.weights: bytes | None = (
            weights_path.read_bytes() if weights_path.exists() else None
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
        # Call the engine's fused MLP block via ctypes. The engine exports
        # polykernel_fused_mlp_block(input, output, M, N, K) -> int (0 = ok).
        import numpy as np  # noqa: PLC0415 - deploy-time dep (image has numpy)

        inp = np.array(request.input_data, dtype=np.float32)
        out = np.zeros(m * n, dtype=np.float32)
        # ctypes call: int polykernel_fused_mlp_block(float* in, float* out, int M, int N, int K)
        self.engine.polykernel_fused_mlp_block.restype = ctypes.c_int
        self.engine.polykernel_fused_mlp_block.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
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
    # local validation: the app + endpoints are constructible without a token.
    print(f"app: {app.name}")
    print(f"image: {image}")
    print(f"service class: {PolyKernelService.__name__}")
    print("endpoints: predict (POST), benchmark (POST), kernels (GET), report (GET)")
    print("gpu: A10, timeout: 600s, startup_timeout: 300s")
    print(f"weights volume: {weights_volume}")
    print("construction OK (no token required)")

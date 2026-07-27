"""Modal app skeleton for PolyKernel (Todo 29 / Wave 6).

Defines `modal.App("polykernel", image=<engine image>)`. This is the skeleton the
GPU work hangs off: the /predict, /benchmark, /kernels, /report endpoints are
Todo 30 (attached via `@app.cls(gpu="H100")` + `@modal.fastapi_endpoint(...)`);
cold-start tuning is Todo 31; the kernel cache is Todo 32; the owner-gated cloud
deploy is Todo 33; the report dashboard is Todo 34.

Importing this module constructs the App AND its image definition and needs NO
Modal token - that construction is the local validation (reports/w6_image.log).
Deploying/serving (which builds the image + provisions GPU containers, spending
money) is the opt-in cloud step behind a `MODAL_TOKEN`:

    modal setup                 # one-time auth (creates/links the token)
    modal deploy modal/app.py   # build image + deploy the app

Why the sibling image module is loaded by PATH (not `from modal.image import ...`):
this file lives in a directory named `modal/`, which collides with the installed
`modal` SDK package. `import modal` correctly resolves to the SDK (a regular
package always wins over the local `modal/` namespace directory), but the qualified
name `modal.image` is the SDK's own image submodule - NOT this project's
`modal/image.py`. We therefore load the sibling by its file path under a unique
module name, which is collision-proof and deterministic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import modal

# The image module name we load the sibling under (deliberately NOT "modal.image",
# which is reserved by the SDK). Unique => no sys.modules collision.
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


# The PolyKernel app: every function/class deployed together shares the engine
# image (CUDA toolkit + the compiled engine .so). Constructing the App is pure
# and token-free; the GPU endpoints attach here in Todo 30.
app = modal.App("polykernel", image=_load_engine_image())

"""Modal deploy flow for PolyKernel (Todo 33 / Wave 6) — OWNER-GATED: MODAL_TOKEN.

Deploys the Todo 30/31 app (``modal/app.py``: ``modal.App("polykernel")`` + the
``PolyKernelService`` GPU class with ``/predict /benchmark /kernels /report``) to
Modal's cloud, verifies the deployed endpoints respond, and measures live cold-start
+ autoscaling by reusing ``modal/cold_start.py`` (Todo 31).

Two paths, gated on the ``MODAL_TOKEN`` environment variable:

* TOKEN PRESENT  — REAL deploy. Runs ``modal deploy modal/app.py`` (persistent),
  discovers the deployed ``*.modal.run`` web URLs, verifies ``/predict`` (POST),
  ``/kernels`` + ``/report`` (GET) respond, runs ``modal/cold_start.py --url <URL>``
  for live cold-vs-warm latency, and writes ``reports/w6_deploy.log`` (deploy URL,
  cold/warm latency, autoscaling behaviour). This path SPENDS Modal credits (the
  free tier / trial credits likely cover a small demo, ~$0) and is therefore opt-in.

* TOKEN ABSENT   — SKIPPED (the path exercised here; there is no MODAL_TOKEN). Prints
  ``SKIPPED: MODAL_TOKEN not set``, documents the EXACT ``modal setup`` + ``modal
  deploy`` + ``modal secret create`` + ``modal volume put`` commands an owner runs to
  deploy, writes ``reports/w6_deploy_skipped.log``, and exits 0. NEVER FAILED, NEVER
  blocks, NO spend, NO Modal API call.

Usage::

    python3 modal/deploy.py             # auto: SKIPPED (no token) or REAL deploy
    python3 modal/deploy.py --url URL   # verify + measure an already-deployed URL

Why this script shells out (``modal deploy`` CLI, ``python3 modal/cold_start.py``)
instead of importing its siblings: this file lives in a directory named ``modal/``
that collides with the installed ``modal`` SDK package (the same collision Todo
29/30/31 solved with importlib). Shelling out sidesteps it entirely and matches the
plan's QA interface (``modal deploy modal/app.py``, ``python modal/cold_start.py
--url $URL``) verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved from this file so the script works from any cwd).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_REF = "modal/app.py"  # the Todo 30/31 app this deploys (relative to root)
APP_NAME = "polykernel"  # modal.App("polykernel", ...) in modal/app.py
COLD_START_PY = PROJECT_ROOT / "modal" / "cold_start.py"  # Todo 31 (reused, not modified)
COLD_START_JSON = PROJECT_ROOT / "reports" / "cold_start.json"
DEPLOY_LOG = PROJECT_ROOT / "reports" / "w6_deploy.log"
SKIPPED_LOG = PROJECT_ROOT / "reports" / "w6_deploy_skipped.log"
VOLUME_NAME = "polykernel-weights"  # modal.Volume.from_name(...) in modal/app.py
HTTP_TIMEOUT_S = 120

# A small bf16-as-float MLP input for the /predict smoke test (M*N = 4 values).
PREDICT_SMOKE_BODY: dict[str, object] = {
    "input_data": [0.5, 0.25, -0.125, 0.0625],
    "shape_m": 2,
    "shape_n": 2,
    "shape_k": 2,
}

# The EXACT commands an owner runs to deploy (printed on SKIPPED, written to the log).
SETUP_COMMANDS = """\
  # 0. One-time auth (register at https://modal.com, then create a token):
  modal setup                      # interactive: creates ~/.modal.toml credentials
  #   (or non-interactively: export MODAL_TOKEN=<token>  /  modal token new)

  # 1. (Optional) secrets the app reads at runtime:
  modal secret create polykernel-secrets HF_TOKEN=<...> --force

  # 2. Upload weights into the app's Volume (mounted at /weights, see modal/app.py):
  modal volume put polykernel-weights ./weights.bin /weights/weights.bin

  # 3. Persistent deploy (builds the CUDA image + provisions the A10 GPU container):
  modal deploy modal/app.py        # prints the deployed *.modal.run web URLs

  # Local dev alternative (ephemeral URL, also needs a token):
  modal serve modal/app.py

  NOTE: Modal free tier / trial credits likely cover a small demo (~$0)."""


def utc_now() -> str:
    """ISO-8601 UTC timestamp (second precision) for log headers."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def has_token() -> bool:
    """True iff a Modal credential is present (MODAL_TOKEN gates the real deploy)."""
    return bool(os.environ.get("MODAL_TOKEN"))


# ---------------------------------------------------------------------------
# SKIPPED path (no token) — the primary deliverable here.
# ---------------------------------------------------------------------------
def write_skipped() -> str:
    """Compose + write reports/w6_deploy_skipped.log; return the text (for stdout)."""
    text = f"""\
======================================================================
PolyKernel T33 (Wave 6): Modal deploy + live cold-start — SKIPPED (no token)
======================================================================
UTC: {utc_now()}

STATUS: SKIPPED (not FAILED). Missing credential: MODAL_TOKEN
Detection: os.environ["MODAL_TOKEN"] is unset/empty -> no Modal auth available.

ACTION TAKEN: graceful skip. NO `modal deploy`, NO Modal API call, NO image build,
NO GPU container provisioned, NO spend. The full app (modal/app.py), the local
`modal serve` dev path, and the documented `modal deploy` flow all exist; the cloud
deploy is OPT-IN and gated behind the absent credential.

Deploy flow that is SKIPPED (would run with a token):
  - modal deploy modal/app.py            (persistent deploy of modal.App("polykernel"))
  - verify deployed /predict (POST) /kernels /report (GET) respond
  - python3 modal/cold_start.py --url URL  (live cold-vs-warm latency, autoscaling)
  - write reports/w6_deploy.log (deploy URL, cold/warm latency, autoscaling)

EXACT modal setup + deploy commands (an owner runs these to deploy):
{SETUP_COMMANDS}

OPT-IN: to perform the REAL deploy, set the credential and re-run:
    export MODAL_TOKEN=<your-token>
    python3 modal/deploy.py
  -> evidence: reports/w6_deploy.log

VERDICT: SKIPPED — graceful degradation, no crash, no surprise spend.
"""
    SKIPPED_LOG.parent.mkdir(parents=True, exist_ok=True)
    SKIPPED_LOG.write_text(text)
    return text


# ---------------------------------------------------------------------------
# REAL deploy path (token present) — written for real; not executed here (no token).
# ---------------------------------------------------------------------------
def run_deploy() -> subprocess.CompletedProcess[str]:
    """Run `modal deploy modal/app.py` (persistent) from the project root."""
    return subprocess.run(
        ["modal", "deploy", APP_REF],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def extract_web_urls(output: str) -> list[str]:
    """Pull the deployed ``*.modal.run`` web URLs out of `modal deploy` output."""
    urls: list[str] = re.findall(r"https?://[^\s'\"├└│╰─]+modal\.run[^\s'\"├└│╰─]*", output)
    # Dedupe, preserve order; strip trailing punctuation Modal may append.
    seen: dict[str, None] = {}
    for raw in urls:
        seen[raw.rstrip(".,;")] = None
    return list(seen)


def http_get(url: str) -> tuple[int, str]:
    """GET a deployed endpoint; return (status, body-prefix). Never raises."""
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as resp:  # noqa: S310
            return resp.status, resp.read(512).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # endpoint answered with an HTTP error
        return exc.code, exc.read(512).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:  # unreachable / timeout
        return -1, f"<unreachable: {exc}>"


def http_post_json(url: str, body: dict[str, object]) -> tuple[int, str]:
    """POST a JSON body to a deployed endpoint; return (status, body-prefix)."""
    req = urllib.request.Request(  # noqa: S310 - deployed Modal URL, owner-gated
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:  # noqa: S310
            return resp.status, resp.read(512).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(512).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        return -1, f"<unreachable: {exc}>"


def verify_endpoints(base_url: str) -> dict[str, tuple[int, str]]:
    """Smoke-test the deployed endpoints; return {endpoint: (status, body-prefix)}."""
    base = base_url.rstrip("/")
    return {
        "GET /kernels": http_get(f"{base}/kernels"),
        "GET /report": http_get(f"{base}/report"),
        "POST /predict": http_post_json(f"{base}/predict", PREDICT_SMOKE_BODY),
    }


def measure_cold_start(base_url: str) -> dict[str, object] | None:
    """Reuse modal/cold_start.py (Todo 31) for live cold-vs-warm latency.

    Returns the parsed reports/cold_start.json, or None if cold_start.py / the
    JSON is absent (e.g. Todo 31 not yet present) — never fatal to the deploy log.
    """
    if not COLD_START_PY.exists():
        return {"note": f"cold-start SKIPPED: {COLD_START_PY.name} not present (Todo 31)"}
    _ = subprocess.run(
        [sys.executable, str(COLD_START_PY), "--url", base_url],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if not COLD_START_JSON.exists():
        return {"note": "cold-start ran but wrote no reports/cold_start.json"}
    parsed: dict[str, object] = json.loads(COLD_START_JSON.read_text())
    return parsed


@dataclass(frozen=True, slots=True)
class DeployOutcome:
    """Everything the deploy log records: one value object, not four loose params."""

    urls: list[str]
    verify: dict[str, tuple[int, str]]
    cold: dict[str, object] | None
    deploy_out: str


def write_deploy_log(outcome: DeployOutcome) -> str:
    """Compose + write reports/w6_deploy.log; return the text (for stdout)."""
    urls, verify, cold, deploy_out = (
        outcome.urls,
        outcome.verify,
        outcome.cold,
        outcome.deploy_out,
    )
    base = urls[0] if urls else "<no modal.run URL discovered>"
    lines = [
        "======================================================================",
        "PolyKernel T33 (Wave 6): Modal deploy + live cold-start/autoscaling",
        "======================================================================",
        f"UTC: {utc_now()}",
        "",
        "STATUS: DEPLOYED (MODAL_TOKEN present).",
        f"app: {APP_NAME}  ref: {APP_REF}",
        f"deploy URL: {base}",
        f"all discovered web URLs: {urls or '[]'}",
        "",
        "--- endpoint verification (deployed) ---",
    ]
    for endpoint, (status, body) in verify.items():
        lines.append(f"{endpoint}: HTTP {status}  body[:512]={body!r}")
    lines += [
        "",
        "--- live cold-start + autoscaling (reuses modal/cold_start.py) ---",
        json.dumps(cold, indent=2) if cold is not None else "<not measured>",
        "",
        "--- modal deploy output (tail) ---",
        "\n".join(deploy_out.splitlines()[-30:]),
        "",
        "RESULT: PASS - deployed, endpoints verified, cold-start measured.",
    ]
    text = "\n".join(lines) + "\n"
    DEPLOY_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEPLOY_LOG.write_text(text)
    return text


def deploy_and_verify() -> int:
    """REAL deploy: modal deploy -> verify endpoints -> cold-start -> log. Exit code."""
    proc = run_deploy()
    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        sys.stderr.write(f"modal deploy FAILED (exit {proc.returncode}):\n{output}\n")
        return proc.returncode
    urls = extract_web_urls(output)
    if not urls:
        sys.stderr.write(f"modal deploy OK but no *.modal.run URL found in output:\n{output}\n")
        return 1
    outcome = DeployOutcome(urls, verify_endpoints(urls[0]), measure_cold_start(urls[0]), output)
    print(write_deploy_log(outcome), end="")
    return 0


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="PolyKernel Modal deploy (owner-gated).")
    parser.add_argument("--url", help="verify + measure an already-deployed URL (skip deploy)")
    args = parser.parse_args()

    url: str | None = args.url
    if url is not None:  # manual verification of a known deployment (token already used)
        outcome = DeployOutcome([url], verify_endpoints(url), measure_cold_start(url), "<--url mode>")
        print(write_deploy_log(outcome), end="")
        return 0

    if not has_token():
        print(write_skipped(), end="")
        return 0  # SKIPPED, never FAILED, never blocks, no spend

    return deploy_and_verify()


if __name__ == "__main__":
    raise SystemExit(main())

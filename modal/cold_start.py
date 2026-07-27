"""Client-side cold-start measurement for the PolyKernel Modal service (Todo 31).

Measures cold-vs-warm request latency against a deployed/served app URL, entirely
from the CLIENT (no Modal SDK, no token - just HTTP timing):

    1. Sleep PAST scaledown_window so the idle container scales down (cold state).
    2. Time the FIRST request  -> cold_ms  (container must boot + restore + warm up).
    3. Time subsequent requests -> warm_ms (container already up; median of N).
    4. Write reports/cold_start.json ({cold_ms, warm_ms, ...}; expect cold > warm).

Usage (live; needs a deployed app -> modal serve / deploy behind a MODAL_TOKEN):

    python modal/cold_start.py --url https://<app>--<env>.modal.run

Usage (no token / no deployment -> documented placeholder, still writes the JSON):

    python modal/cold_start.py --dry-run

Fast scale-down NEGATIVE case (proves the measurement captures cold starts): set
scaledown_window to the Modal minimum (2s). The sleep is then ~7s, containers drop
almost immediately, and every probe after a >2s idle is a cold start - so cold_ms
dominates reliably:

    python modal/cold_start.py --url $URL --scaledown-window 2

The probe defaults to GET /kernels (a body-less endpoint) so the timing reflects
CONTAINER cold-start (boot + snapshot restore + GPU warmup), not request-body
overhead. Containers boot in ~1s; with enable_memory_snapshot the restore skips the
snap=True init, so the measured cold_ms is dominated by provisioning + restore.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Modal's minimum scaledown_window (seconds). Used for the fast-scale-down negative
# case: at 2s the container drops almost immediately after idle, so a cold start is
# observed on essentially every probe (proves the measurement captures cold starts).
MIN_SCALEDOWN_WINDOW_S = 2

# Matches SCALEDOWN_WINDOW in modal/app.py (the production idle-before-scaledown).
DEFAULT_SCALEDOWN_WINDOW_S = 300

DEFAULT_OUT = Path("reports/cold_start.json")


@dataclass(frozen=True, slots=True)
class ColdStartResult:
    """One cold-vs-warm measurement (or a documented dry-run placeholder)."""

    cold_ms: float | None
    warm_ms: float | None
    cold_gt_warm: bool | None
    warm_samples_ms: list[float] = field(default_factory=list)
    scaledown_window_s: float = DEFAULT_SCALEDOWN_WINDOW_S
    sleep_s: float = 0.0
    endpoint: str = "/kernels"
    url: str | None = None
    status: str = "measured"
    note: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )


def time_request(url: str, timeout_s: float) -> float:
    """Wall-clock latency (ms) of one GET to `url` (status line + full body read)."""
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
        resp.read()
    return (time.perf_counter() - start) * 1000.0


def measure(
    url: str,
    scaledown_window_s: float,
    margin_s: float,
    endpoint: str,
    repeat: int,
    timeout_s: float,
) -> ColdStartResult:
    """Sleep past the scaledown window, then time one cold + `repeat` warm requests."""
    probe = url.rstrip("/") + endpoint
    sleep_s = scaledown_window_s + margin_s
    # Force a scale-down: idle past the window so the warm container drops. This is
    # what makes the NEXT request a genuine cold start.
    print(f"sleeping {sleep_s:.0f}s (> scaledown_window={scaledown_window_s:.0f}s) to force scale-down...",
          file=sys.stderr)
    time.sleep(sleep_s)
    cold_ms = time_request(probe, timeout_s)
    warm_samples = [time_request(probe, timeout_s) for _ in range(repeat)]
    warm_ms = statistics.median(warm_samples)
    return ColdStartResult(
        cold_ms=round(cold_ms, 3),
        warm_ms=round(warm_ms, 3),
        cold_gt_warm=cold_ms > warm_ms,
        warm_samples_ms=[round(s, 3) for s in warm_samples],
        scaledown_window_s=scaledown_window_s,
        sleep_s=sleep_s,
        endpoint=endpoint,
        url=url,
        status="measured",
        note="cold = first request after scale-down; warm = median of subsequent",
    )


def dry_run_result(
    scaledown_window_s: float, margin_s: float, endpoint: str
) -> ColdStartResult:
    """Documented placeholder when no deployment URL is available (no token)."""
    return ColdStartResult(
        cold_ms=None,
        warm_ms=None,
        cold_gt_warm=None,
        scaledown_window_s=scaledown_window_s,
        sleep_s=scaledown_window_s + margin_s,
        endpoint=endpoint,
        url=None,
        status="skipped-no-deployment",
        note=(
            "Live measurement needs a deployed/served app (modal serve / deploy, "
            "behind MODAL_TOKEN). Expected behavior: cold_ms > warm_ms - the first "
            "request after scale-down pays container boot + snapshot restore + GPU "
            "warmup, while subsequent requests hit an already-warm container. "
            "Run: python modal/cold_start.py --url $URL"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cold_start",
        description="Client-side cold-vs-warm latency measurement for the PolyKernel "
                    "Modal service. Sleeps past scaledown_window to force a scale-down, "
                    "times the first (cold) vs subsequent (warm) requests, and writes "
                    "reports/cold_start.json. Needs a deployed app URL for a live run.",
    )
    parser.add_argument("--url", default=None,
                        help="deployed app base URL, e.g. https://<app>--<env>.modal.run "
                             "(omit with --dry-run for a documented placeholder)")
    parser.add_argument("--scaledown-window", type=float, default=DEFAULT_SCALEDOWN_WINDOW_S,
                        help="app scaledown_window in seconds; the script sleeps this + "
                             f"--margin to force scale-down (min {MIN_SCALEDOWN_WINDOW_S}s "
                             "for the fast negative case; default %(default)s)")
    parser.add_argument("--margin", type=float, default=5.0,
                        help="extra seconds slept past the scaledown window (default %(default)s)")
    parser.add_argument("--endpoint", default="/kernels",
                        help="probe path (body-less GET recommended; default %(default)s)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="number of warm samples; warm_ms is their median (default %(default)s)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-request HTTP timeout in seconds (default %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="write a documented placeholder JSON without hitting any URL "
                             "(no token / no deployment needed)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output JSON path (default %(default)s)")
    args = parser.parse_args(argv)

    if args.dry_run or args.url is None:
        if not args.dry_run:
            print("note: no --url given; writing a documented placeholder (use --dry-run to silence this).",
                  file=sys.stderr)
        result = dry_run_result(args.scaledown_window, args.margin, args.endpoint)
    else:
        result = measure(args.url, args.scaledown_window, args.margin,
                         args.endpoint, args.repeat, args.timeout)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

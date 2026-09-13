"""Live connectivity/data smoke tests for the official public providers.

These checks are an operational health probe, not a unit-test replacement.
A provider can legitimately be unavailable to GitHub-hosted runners because
of rate limits, bot controls, maintenance, or network policy. The smoke test
therefore fails only when every provider is unavailable or a provider returns
an unexpected non-request error; partial upstream availability is considered
a healthy enough signal for CI.
"""
from __future__ import annotations

import requests

from .bis import smoke_test as bis_test
from .eurostat import smoke_test as eurostat_test
from .imf import smoke_test as imf_test
from .oecd import smoke_test as oecd_test
from .world_bank import smoke_test as world_bank_test


def main() -> int:
    tests = (world_bank_test, imf_test, bis_test, eurostat_test, oecd_test)
    healthy = 0
    unavailable = 0
    hard_failures = 0

    for test in tests:
        try:
            result = test()
            ok = bool(result.get("ok"))
            status = "OK" if ok else "NO DATA"
            print(f"{result['source']}: {status}")
            print(f"  {result.get('endpoint', '')}")
            if ok:
                healthy += 1
            else:
                unavailable += 1
        except requests.RequestException as exc:
            unavailable += 1
            print(f"{test.__module__}: UNAVAILABLE — {exc}")
        except Exception as exc:  # noqa: BLE001
            hard_failures += 1
            print(f"{test.__module__}: FAILED — {exc}")

    if hard_failures:
        print(f"Smoke test failed: {hard_failures} provider adapter error(s).")
        return 1
    if healthy:
        print(f"Smoke test passed: {healthy} official provider(s) reachable; {unavailable} unavailable.")
        return 0
    print("Smoke test failed: no official provider was reachable.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

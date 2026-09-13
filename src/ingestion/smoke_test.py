"""Live connectivity/data smoke tests for the official public providers."""
from __future__ import annotations

import requests

from .bis import smoke_test as bis_test
from .eurostat import smoke_test as eurostat_test
from .imf import smoke_test as imf_test
from .oecd import smoke_test as oecd_test
from .world_bank import smoke_test as world_bank_test


def main() -> int:
    tests = (
        ("World Bank", world_bank_test),
        ("IMF", imf_test),
        ("BIS", bis_test),
        ("Eurostat", eurostat_test),
        ("OECD", oecd_test),
    )
    hard_failures = 0
    healthy = 0
    for name, test in tests:
        try:
            result = test()
            ok = bool(result.get("ok"))
            print(f"{name}: {'OK' if ok else 'NO CURRENT-YEAR DATA'}")
            print(f"  endpoint: {result.get('endpoint', '')}")
            if ok:
                healthy += 1
        except requests.RequestException as exc:
            print(f"{name}: UNAVAILABLE")
            print(f"  request error: {exc}")
        except Exception as exc:  # noqa: BLE001
            hard_failures += 1
            print(f"{name}: ADAPTER ERROR")
            print(f"  error: {exc}")

    # CI should fail on broken adapter code, but not merely because an upstream
    # provider is temporarily unreachable or has not published a 2026 value.
    if hard_failures:
        print(f"Smoke test failed: {hard_failures} adapter error(s).")
        return 1
    print(f"Smoke test completed: {healthy}/{len(tests)} providers returned current-year data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

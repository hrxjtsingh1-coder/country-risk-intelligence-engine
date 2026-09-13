"""Live connectivity/data smoke tests for the five official providers."""
from __future__ import annotations

from .bis import smoke_test as bis_test
from .eurostat import smoke_test as eurostat_test
from .imf import smoke_test as imf_test
from .oecd import smoke_test as oecd_test
from .world_bank import smoke_test as world_bank_test


def main() -> int:
    failures = 0
    for test in (world_bank_test, imf_test, bis_test, eurostat_test, oecd_test):
        try:
            result = test()
            ok = bool(result.get("ok"))
            print(f"{result['source']}: {'OK' if ok else 'NO DATA'}")
            print(f"  {result.get('endpoint', '')}")
            if not ok:
                failures += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{test.__module__}: FAILED — {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

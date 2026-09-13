"""Official-provider smoke tests with actionable diagnostics."""
from __future__ import annotations

from .bis import smoke_test as bis_test
from .eurostat import smoke_test as eurostat_test
from .imf import smoke_test as imf_test
from .oecd import smoke_test as oecd_test
from .world_bank import smoke_test as world_bank_test


def _run(name, fn):
    try:
        result = fn()
        ok = bool(result.get("ok"))
        print(f"{name}: {'OK' if ok else 'FAILED'}")
        if not ok:
            print(f"  endpoint: {result.get('endpoint', 'n/a')}")
            print(f"  detail: {result.get('detail', 'no usable data returned')}")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: FAILED")
        print(f"  error: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    checks = [
        ("World Bank", world_bank_test),
        ("IMF", imf_test),
        ("BIS", bis_test),
        ("Eurostat", eurostat_test),
        ("OECD", oecd_test),
    ]
    passed = sum(_run(name, fn) for name, fn in checks)
    print(f"\nProvider smoke tests: {passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

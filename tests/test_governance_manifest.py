import json

from src.governance import manifest as mf
from tests.test_analytics import panel as analytics_panel


def _write_inputs(tmp_path):
    a = tmp_path / "input_a.txt"
    a.write_text("alpha=1\n", encoding="utf-8")
    b = tmp_path / "input_b.txt"
    b.write_text("beta=2\n", encoding="utf-8")
    return [a, b]


def test_build_and_verify_consistent(tmp_path):
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "manifest.json"
    manifest = mf.build_manifest(inputs=inputs, outputs=[], panel=analytics_panel(), manifest_path=out)
    assert manifest["methodology_version"] == mf.METHODOLOGY_VERSION

    verification = mf.verify_manifest(out)
    assert verification.overall == "consistent"
    assert all(f.status == "ok" for f in verification.files)


def test_verify_detects_modified_input(tmp_path):
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "manifest.json"
    inputs[0].write_text("alpha=999\n", encoding="utf-8")  # built AFTER tamper? build first below
    # Rebuild after restoring, then re-tamper, to keep the baseline pristine.
    inputs[0].write_text("alpha=1\n", encoding="utf-8")
    mf.build_manifest(inputs=inputs, outputs=[], panel=analytics_panel(), manifest_path=out)
    inputs[0].write_text("alpha=999\n", encoding="utf-8")

    verification = mf.verify_manifest(out)
    assert verification.overall == "drifted"
    first = next(f for f in verification.files if f.path.endswith("input_a.txt"))
    assert first.status == "drifted"


def test_verify_detects_missing_file(tmp_path):
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "manifest.json"
    mf.build_manifest(inputs=inputs, outputs=[], panel=analytics_panel(), manifest_path=out)
    inputs[0].unlink()

    verification = mf.verify_manifest(out)
    assert verification.overall == "inconsistent"
    first = next(f for f in verification.files if f.path.endswith("input_a.txt"))
    assert first.status == "missing"


def test_manifest_json_schema(tmp_path):
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "manifest.json"
    mf.build_manifest(inputs=inputs, outputs=[], panel=analytics_panel(), signed_by="Tester", manifest_path=out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == mf.FORMAT_ID
    assert payload["schema_version"] == mf.SCHEMA_VERSION
    assert payload["methodology_version"] == "2.0.0"
    assert payload["dataset_mode"] == "demo"
    assert payload["env"]["python"]
    assert payload["signoff"]["signed_by"] == "Tester"
    assert payload["files"]
    assert payload["model"]["coverage"]["status"]
    assert set(payload["model"]["backtest"]) >= {"detection_rate", "flagged", "missed", "inconclusive"}


def test_cli_build_and_verify(tmp_path, capsys):
    out = tmp_path / "manifest.json"
    assert mf.main(["build", "--panel", "data/demo/panel_wide.csv", "--signed-by", "CLI", "--output", str(out)]) == 0
    assert "Built manifest" in capsys.readouterr().out
    assert mf.main(["verify", "--manifest", str(out)]) == 0
    assert "CONSISTENT" in capsys.readouterr().out

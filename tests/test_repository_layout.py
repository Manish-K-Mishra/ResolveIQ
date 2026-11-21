"""Repository structure the submission requires (Setup Guide §07/§08): entry points, register, CI."""
import json
import re
import subprocess
import sys
from pathlib import Path
import pytest
import cloudserve
from cloudserve.providers import PROMPT_DIR, load_prompt

ROOT = Path(__file__).resolve().parents[1]


def test_package_root_resolves_to_repository_with_src_layout():
    assert cloudserve.ROOT == ROOT
    assert Path(cloudserve.__file__).resolve().parent == ROOT / "src" / "cloudserve"
    assert PROMPT_DIR == ROOT / "prompts" / "build"


def test_runtime_artefacts_and_secrets_are_ignored():
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignore and "storage/" in ignore


def test_every_build_prompt_is_in_the_register_with_its_version():
    register = (ROOT / "prompts" / "README.md").read_text(encoding="utf-8")
    files = sorted(PROMPT_DIR.glob("*.md"))
    assert files
    for f in files:
        p = load_prompt(f.name)
        row = next((line for line in register.splitlines() if f"build/{f.name}" in line), None)
        assert row, f"{f.name} missing from prompts/README.md"
        assert re.search(rf"\|\s*{p['id']}\s*\|\s*{re.escape(p['version'])}\s*\|", row)


def test_ci_runs_tests_and_the_unattended_gate_without_secrets():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "python -m pytest -q" in ci
    assert ci.index("python scripts/fetch_models.py") < ci.index("python -m pytest -q")
    assert "python -m evaluation.harness --input data/validation_tickets.json --output storage/ci_eval" in ci
    assert "secrets." not in ci and "API_KEY" not in ci


@pytest.mark.parametrize("entry", [["-m", "evaluation.harness"], ["evaluation/harness.py"]])
def test_documented_harness_entry_points_run_from_repo_root(entry, tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, *entry, "--input", "examples/four_channels.json", "--output", str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    m = json.loads((out / "metrics.json").read_text())
    assert m["volume"]["processed"] == 4 and m["governance"]["decision_log_reconciles"]


@pytest.mark.parametrize("module", ["cloudserve.evaluation", "cloudserve.cli"])
def test_reconciliation_failure_exits_non_zero(module, monkeypatch, tmp_path):
    import importlib
    mod = importlib.import_module(module)
    real = mod.run_batch

    def short_log(*args, **kwargs):
        results, logged = real(*args, **kwargs)
        return results, logged - 1
    monkeypatch.setattr(mod, "run_batch", short_log)
    args = ["--input", str(ROOT / "examples/one_ticket.json")]
    args += ["--output", str(tmp_path / "out")] if module.endswith("evaluation") else \
            ["--output", str(tmp_path / "o.json"), "--db", str(tmp_path / "o.sqlite")]
    with pytest.raises(SystemExit, match="(?i)reconciliation failed|integrity check failed"):
        mod.main(args)

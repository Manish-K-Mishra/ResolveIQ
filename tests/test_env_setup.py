from pathlib import Path

def test_env_example_contains_blank_claude_key():
    config = Path(__file__).resolve().parents[1] / '.env.example'
    rows = config.read_text().splitlines()
    assert 'ANTHROPIC_API_KEY=' in rows
    assert 'CLOUDSERVE_PROVIDER=anthropic' in rows

def test_local_env_is_ignored():
    ignore = (Path(__file__).resolve().parents[1] / '.gitignore').read_text().splitlines()
    assert '.env' in ignore


def test_requirements_are_exactly_pinned():
    rows = [r for r in (Path(__file__).resolve().parents[1] / 'requirements.txt').read_text().splitlines()
            if r.strip() and not r.startswith('#')]
    assert rows and all('==' in r for r in rows)


def test_routing_policy_reproduces_from_development_data(tmp_path):
    import json, subprocess, sys
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / 'p.json'
    subprocess.run([sys.executable, str(root / 'scripts/derive_routing_policy.py'), '--out', str(out)],
                   check=True, capture_output=True)
    strip = lambda p: {k: v for k, v in json.loads(p.read_text()).items() if k != 'derived_on'}
    assert strip(out) == strip(root / 'config/routing_policy.json')

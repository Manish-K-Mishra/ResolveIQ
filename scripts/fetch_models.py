"""Fetch the embedding model used for retrieval (explicit, documented setup step).

    python scripts/fetch_models.py            # into storage/models/all-MiniLM-L6-v2/

Downloads the ONNX export of sentence-transformers/all-MiniLM-L6-v2 and its tokenizer from the
Hugging Face Hub at a pinned revision, and checks each file's SHA-256 so every machine embeds
with exactly the same weights. Nothing is downloaded at run time: if these files are missing,
retrieval falls back to lexical search and every decision record says so.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILES = {   # repo path -> (local name, sha256)
    "onnx/model.onnx": ("model.onnx", "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"),
    "tokenizer.json": ("tokenizer.json", "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=str(ROOT / "storage" / "models" / "all-MiniLM-L6-v2"))
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for remote, (local, expected) in FILES.items():
        target = dest / local
        if not (target.exists() and (expected is None or sha256(target) == expected)):
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
            tmp = target.with_suffix(target.suffix + ".part")
            print(f"downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            tmp.replace(target)
        digest = sha256(target)
        if expected is not None and digest != expected:
            target.unlink()
            print(f"checksum mismatch for {local}: got {digest}", file=sys.stderr)
            return 1
        print(f"ok {target} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

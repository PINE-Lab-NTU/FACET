#!/usr/bin/env python3
"""Create a deterministic source archive without environments or generated files."""

import argparse
import gzip
import hashlib
import io
from pathlib import Path
import tarfile

from check_release import ROOT, check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True, help="archive path outside the source tree")
    args = ap.parse_args()
    output = args.output.resolve()
    if ROOT == output.parent or ROOT in output.parents:
        ap.error("write the archive outside the source tree")
    check()
    skip = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store"}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for p in sorted(ROOT.rglob("*")):
            relative = p.relative_to(ROOT)
            if not p.is_file() or any(x in skip or x.endswith(".egg-info") for x in relative.parts):
                continue
            if p.is_symlink():
                raise ValueError(f"symlinks are not release inputs: {relative}")
            data = p.read_bytes()
            info = tarfile.TarInfo("facet-online-adaptation/" + relative.as_posix())
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as compressed:
            compressed.write(buffer.getvalue())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    print(f"Wrote {output.name}, {output.stat().st_size} bytes, SHA256 {digest}")


if __name__ == "__main__":
    main()

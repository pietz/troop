#!/usr/bin/env python3
"""Release helper for CI.

Commands:
  detect  - Outputs version and whether a release is needed.
  build   - Build distributions using uv.
  publish - Publish to PyPI using uv and PYPI_TOKEN.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple


def read_version(pyproject_path: Path = Path("pyproject.toml")) -> str:
    import tomllib

    data = tomllib.loads(pyproject_path.read_text())
    version = data.get("project", {}).get("version")
    if not version:
        print("::error::[project].version missing in pyproject.toml", file=sys.stderr)
        raise SystemExit(1)
    return str(version)


def tag_exists(tag: str) -> bool:
    try:
        subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def write_output(**kwargs: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as fh:
        for k, v in kwargs.items():
            fh.write(f"{k}={v}\n")


def cmd_detect() -> int:
    version = read_version()
    tag = f"v{version}"
    exists = tag_exists(tag)
    release_needed = "false" if exists else "true"
    print(f"Detected version: {version}")
    print(f"Git tag {tag} exists: {exists}")
    write_output(version=version, release_needed=release_needed)
    # For local runs, also echo the outputs to help debugging
    print(f"::notice::release_needed={release_needed}")
    return 0


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def cmd_build() -> int:
    # Ensure dist/ is clean? Not strictly necessary; uv build overwrites.
    print("Building with uv build...")
    run(["uv", "build"])
    # Verify dist artifacts exist
    dist = Path("dist")
    if not dist.exists() or not any(dist.iterdir()):
        print("::error::No artifacts in dist/ after build", file=sys.stderr)
        return 1
    return 0


def cmd_publish() -> int:
    token = os.environ.get("PYPI_TOKEN")
    if not token:
        print("::error::PYPI_TOKEN environment variable is required", file=sys.stderr)
        return 1
    print("Publishing to PyPI with uv publish...")
    # Use password token auth
    env = os.environ.copy()
    # uv publish supports --username/--password
    run(["uv", "publish", "--username", "__token__", "--password", token])
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "detect":
        return cmd_detect()
    if cmd == "build":
        return cmd_build()
    if cmd == "publish":
        return cmd_publish()
    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


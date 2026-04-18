#!/usr/bin/env python3
"""Publish the current workspace to the public GitHub archive from a clean temp repo.

This avoids relying on the local .git state by copying the intended public
archive contents into a temporary directory, initializing a fresh git
repository there, and pushing `main` plus the first archival tag.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "https://github.com/chandanacharitha/multi-timescale-memory-routing.git"
TAG = "paper-v1.0.0"
COPY_ITEMS = [
    ".gitignore",
    "ARCHIVE_CONTENTS.md",
    "BASELINE_MVP.md",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "artifacts",
    "data",
    "deliverables",
    "docs",
    "journalplan.txt",
    "multitimescale_memory",
    "paper",
    "pyproject.toml",
    "researchplan.txt",
    "scripts",
    "tests",
]
IGNORE_NAMES = {
    ".git",
    ".venv",
    ".venv-journal",
    ".cache",
    "__pycache__",
    ".DS_Store",
}


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> str | None:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture,
    )
    if capture:
        return result.stdout.strip()
    return None


def ignore(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORE_NAMES or name.endswith(".pyc")}


def copy_archive(dest: Path) -> None:
    for item in COPY_ITEMS:
        src = ROOT / item
        target = dest / item
        print(f"Copying {item}", flush=True)
        if src.is_dir():
            shutil.copytree(src, target, ignore=ignore, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        publish_root = Path(tmpdir) / "publish_repo"
        publish_root.mkdir(parents=True, exist_ok=True)
        print(f"Copying archive into {publish_root}", flush=True)
        copy_archive(publish_root)

        print("Initializing clean git repository", flush=True)
        run(["git", "init", "-b", "main"], cwd=publish_root)
        run(["git", "config", "user.name", "Chandana Charitha Peddinti"], cwd=publish_root)
        run(
            [
                "git",
                "config",
                "user.email",
                "181019324+chandanacharitha@users.noreply.github.com",
            ],
            cwd=publish_root,
        )
        run(["git", "add", "."], cwd=publish_root)
        run(
            ["git", "commit", "-m", "Prepare public archive and submission packages"],
            cwd=publish_root,
        )

        token = os.environ.get("GITHUB_TOKEN", "").strip()
        remote = REMOTE_REPO
        if token:
            remote = f"https://x-access-token:{token}@github.com/chandanacharitha/multi-timescale-memory-routing.git"
        run(["git", "remote", "add", "origin", remote], cwd=publish_root)
        print(f"Pushing main to {REMOTE_REPO}", flush=True)
        run(["git", "push", "-u", "origin", "main"], cwd=publish_root)
        print(f"Pushing tag {TAG}", flush=True)
        run(["git", "tag", TAG], cwd=publish_root)
        run(["git", "push", "origin", TAG], cwd=publish_root)
        head = run(["git", "rev-parse", "HEAD"], cwd=publish_root, capture=True)
        print(f"Published commit: {head}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MQUAKE_URLS = {
    "MQuAKE-CF.json": "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF.json",
    "MQuAKE-CF-3k.json": "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k.json",
    "MQuAKE-CF-3k-v2.json": "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k-v2.json",
    "MQuAKE-T.json": "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-T.json",
}

KNOWEDIT_URLS = {
    "README.md": "https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/README.md",
    "WikiBio/wikibio-test-all.json": "https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/WikiBio/wikibio-test-all.json",
    "WikiBio/wikibio-train-all.json": "https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/WikiBio/wikibio-train-all.json",
    "ZsRE/ZsRE-test-all.json": "https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/ZsRE/ZsRE-test-all.json",
    "wiki_counterfact/test_cf.json": "https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/wiki_counterfact/test_cf.json",
    "wiki_counterfact/train_cf.json": "https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/wiki_counterfact/train_cf.json",
    "wiki_recent/recent_test.json": "https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/wiki_recent/recent_test.json",
    "wiki_recent/recent_train.json": "https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/wiki_recent/recent_train.json",
}

UNIEDIT_URLS = {
    "README.md": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/README.md",
    "test/agronomy.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/agronomy.json",
    "test/art.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/art.json",
    "test/astronomy.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/astronomy.json",
    "test/biology.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/biology.json",
    "test/chemistry.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/chemistry.json",
    "test/civil engineering.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/civil%20engineering.json",
    "test/computer science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/computer%20science.json",
    "test/data science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/data%20science.json",
    "test/economics.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/economics.json",
    "test/environmental science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/environmental%20science.json",
    "test/geoscience.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/geoscience.json",
    "test/history.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/history.json",
    "test/jurisprudence.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/jurisprudence.json",
    "test/literature.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/literature.json",
    "test/material science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/material%20science.json",
    "test/mathematics.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/mathematics.json",
    "test/mechanical engineering.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/mechanical%20engineering.json",
    "test/medicine.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/medicine.json",
    "test/pedagogy.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/pedagogy.json",
    "test/philosophy.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/philosophy.json",
    "test/physics.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/physics.json",
    "test/political science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/political%20science.json",
    "test/psychology.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/psychology.json",
    "test/sociology.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/sociology.json",
    "test/sports science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/test/sports%20science.json",
    "train/agronomy.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/agronomy.json",
    "train/art.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/art.json",
    "train/astronomy.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/astronomy.json",
    "train/biology.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/biology.json",
    "train/chemistry.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/chemistry.json",
    "train/civil engineering.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/civil%20engineering.json",
    "train/computer science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/computer%20science.json",
    "train/data science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/data%20science.json",
    "train/economics.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/economics.json",
    "train/environmental science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/environmental%20science.json",
    "train/geoscience.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/geoscience.json",
    "train/history.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/history.json",
    "train/jurisprudence.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/jurisprudence.json",
    "train/literature.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/literature.json",
    "train/material science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/material%20science.json",
    "train/mathematics.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/mathematics.json",
    "train/mechanical engineering.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/mechanical%20engineering.json",
    "train/medicine.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/medicine.json",
    "train/pedagogy.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/pedagogy.json",
    "train/philosophy.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/philosophy.json",
    "train/physics.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/physics.json",
    "train/political science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/political%20science.json",
    "train/psychology.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/psychology.json",
    "train/sociology.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/sociology.json",
    "train/sports science.json": "https://huggingface.co/datasets/qizhou/UniEdit/resolve/main/train/sports%20science.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(url: str, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return {
            "path": str(destination.relative_to(ROOT)),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "source_url": url,
        }
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "20",
            "--max-time",
            "120",
            url,
            "-o",
            str(destination),
        ],
        check=True,
    )
    return {
        "path": str(destination.relative_to(ROOT)),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "source_url": url,
    }


def fetch_mquake(target_dir: Path) -> dict[str, object]:
    files = []
    for filename, url in MQUAKE_URLS.items():
        files.append(_download_url(url, target_dir / filename))
    manifest = {
        "benchmark": "MQuAKE",
        "source": "princeton-nlp/MQuAKE",
        "files": files,
    }
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"root": str(target_dir.relative_to(ROOT)), "manifest": str(manifest_path.relative_to(ROOT)), "files": files}


def fetch_knowedit(target_dir: Path) -> dict[str, object]:
    files = []
    failures = []
    for relative_path, url in KNOWEDIT_URLS.items():
        try:
            files.append(_download_url(url, target_dir / relative_path))
        except subprocess.CalledProcessError as exc:
            failures.append(
                {
                    "path": str((target_dir / relative_path).relative_to(ROOT)),
                    "source_url": url,
                    "returncode": exc.returncode,
                }
            )
    manifest = {
        "benchmark": "KnowEdit",
        "source": "zjunlp/KnowEdit",
        "source_urls": KNOWEDIT_URLS,
        "files": files,
        "failures": failures,
        "complete": not failures,
    }
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"root": str(target_dir.relative_to(ROOT)), "manifest": str(manifest_path.relative_to(ROOT)), "files": files}


def fetch_uniedit(target_dir: Path) -> dict[str, object]:
    files = []
    failures = []
    for relative_path, url in UNIEDIT_URLS.items():
        try:
            files.append(_download_url(url, target_dir / relative_path))
        except subprocess.CalledProcessError as exc:
            failures.append(
                {
                    "path": str((target_dir / relative_path).relative_to(ROOT)),
                    "source_url": url,
                    "returncode": exc.returncode,
                }
            )
    manifest = {
        "benchmark": "UniEdit",
        "source": "qizhou/UniEdit",
        "source_urls": UNIEDIT_URLS,
        "files": files,
        "failures": failures,
        "complete": not failures,
    }
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"root": str(target_dir.relative_to(ROOT)), "manifest": str(manifest_path.relative_to(ROOT)), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark inputs needed for the next TMLR evidence cycle.")
    parser.add_argument("--root", default=str(ROOT), help="Repository root.")
    parser.add_argument("--skip-mquake", action="store_true", help="Do not download MQuAKE.")
    parser.add_argument("--skip-knowedit", action="store_true", help="Do not download the KnowEdit fallback benchmark.")
    parser.add_argument("--skip-uniedit", action="store_true", help="Do not download the UniEdit benchmark snapshot.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if root != ROOT.resolve():
        raise SystemExit(f"Expected repository root {ROOT}, got {root}")

    outputs: dict[str, object] = {}
    benchmark_root = root / "data" / "benchmarks"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_mquake:
        outputs["mquake"] = fetch_mquake(benchmark_root / "mquake")
    if not args.skip_knowedit:
        outputs["knowedit"] = fetch_knowedit(benchmark_root / "knowedit")
    if not args.skip_uniedit:
        outputs["uniedit"] = fetch_uniedit(benchmark_root / "uniedit")

    summary_path = benchmark_root / "tmlr_input_manifest.json"
    summary_path.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path.relative_to(root)), "outputs": outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

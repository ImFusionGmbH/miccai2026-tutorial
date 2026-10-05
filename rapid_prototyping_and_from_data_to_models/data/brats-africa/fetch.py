"""Fill in the images of the BraTS-Africa Labels project next to this script.

The Labels project is committed without its images. Its descriptors reference the NIfTI files by
relative path, e.g. ../95_Glioma/BraTS-SSA-00002-000/BraTS-SSA-00002-000-t2w.nii.gz.

TCIA distributes the images as an IBM Aspera Faspex package (1.6 GB, CC BY 4.0). The download uses
the Aspera CLI, either a local `ascli` or its Docker image.

    python fetch.py                    download from TCIA, does nothing if all images exist
    python fetch.py --source <folder>  use an existing download
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
CHECKSUMS = ROOT / "SHA256SUMS"
DOWNLOAD = ROOT / ".download"

# Public package link from https://www.cancerimagingarchive.net/collection/brats-africa/
TCIA_PACKAGE = (
    "https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlw"
    "ZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6Ijk0OCIsInBhc3Njb2RlIjoiOTg2MzVlMGRmNzc3NWQ0NWJmZTQ2NjlhYzQwNjNmYj"
    "cxMjU0MzI1NyIsInBhY2thZ2VfaWQiOiI5NDgiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0=")
ASCLI_IMAGE = "martinlaurent/ascli:latest"


def referenced_images() -> list[Path]:
    """Return the image paths of all descriptors, relative to this folder."""
    tree = ET.parse(PROJECT / "project.xml")
    paths = {
        param.text
        for param in tree.iter("param")
        if param.get("name") == "path" and param.text and param.text.endswith(".nii.gz")
    }
    return sorted((PROJECT / p).resolve().relative_to(ROOT.resolve()) for p in paths)


def download() -> Path:
    """Download the TCIA package into .download and return that folder.

    Aspera resumes an interrupted transfer when the same command runs again.
    """
    DOWNLOAD.mkdir(exist_ok=True)
    receive = ["faspex5", "packages", "receive", f"--url={TCIA_PACKAGE}", "--progress-bar=no"]
    if shutil.which("ascli"):
        command = ["ascli", *receive, f"--to-folder={DOWNLOAD}"]
    elif shutil.which("docker"):
        user = subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()
        group = subprocess.run(["id", "-g"], capture_output=True, text=True).stdout.strip()
        command = [
            "docker", "run", "--rm", "--user", f"{user}:{group}", "-v", f"{DOWNLOAD.resolve()}:/work", ASCLI_IMAGE,
            *receive, "--to-folder=/work"
        ]
    else:
        sys.exit("Neither ascli nor docker is available.\n"
                 "Install the Aspera CLI (https://github.com/IBM/aspera-cli) or download the package in a browser\n"
                 "from https://www.cancerimagingarchive.net/collection/brats-africa/ and pass it with --source.")
    print("Downloading BraTS-Africa from TCIA (1.6 GB)...", flush=True)
    subprocess.run(command, check=True)
    return DOWNLOAD


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def place(images: list[Path], source: Path, move: bool) -> list[Path]:
    """Copy or move the images from `source` to the paths the project expects.

    Files are matched by name, so the folder structure of `source` does not matter.

    Returns:
        The images that were not found in `source`.
    """
    found = {p.name: p for p in source.rglob("*.nii.gz")}
    missing = []
    for image in images:
        target = ROOT / image
        if target.exists():
            continue
        if image.name not in found:
            missing.append(image)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(found[image.name], target)
        else:
            shutil.copy2(found[image.name], target)
    return missing


def verify() -> list[str]:
    """Return the images whose checksum differs from SHA256SUMS or that are missing."""
    failed = []
    for line in CHECKSUMS.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = ROOT / name
        if not path.exists() or sha256(path) != expected:
            failed.append(name)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="folder with an existing download, searched recursively")
    parser.add_argument("--keep-download", action="store_true", help="keep the .download folder")
    parser.add_argument("--verify", action="store_true", help="check the checksums even if all images exist")
    args = parser.parse_args()

    images = referenced_images()
    if not args.source and not args.verify and all((ROOT / image).exists() for image in images):
        return
    source = args.source if args.source else download()
    missing = place(images, source, move=args.source is None)
    if missing:
        sys.exit(f"{len(missing)} images not found in {source}, e.g. {missing[0].name}")

    print(f"Verifying {len(images)} images...")
    failed = verify()
    if failed:
        sys.exit(f"{len(failed)} images do not match SHA256SUMS, e.g. {failed[0]}")
    if args.source is None and not args.keep_download:
        shutil.rmtree(DOWNLOAD)
    print(f"Done. Open {PROJECT} in ImFusion Labels or run from_data_to_models.ipynb.")


if __name__ == "__main__":
    main()

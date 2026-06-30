"""Download and unpack the corpora used for Cadence training.

Corpora used:
  - CANDOR (Zenodo, ~45h dyadic conversational speech, free)
  - AMI Meeting Corpus (OpenSLR, ~100h meeting speech, free)

Run: make data-download
"""

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path("data/raw")

# Zenodo record for CANDOR corpus
CANDOR_URL = "https://zenodo.org/record/7121435/files/candor-corpus.zip"
# AMI headset mix (IHM) from OpenSLR
AMI_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


def download_file(url: str, dest: Path, desc: str) -> None:
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def unpack(archive: Path, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already unpacked")
        return
    print(f"  unpacking {archive.name} → {dest}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)


def main(corpora: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if "candor" in corpora:
        print("==> CANDOR corpus")
        archive = RAW_DIR / "candor-corpus.zip"
        download_file(CANDOR_URL, archive, "CANDOR")
        unpack(archive, RAW_DIR / "candor")

    if "ami" in corpora:
        print("==> AMI corpus annotations")
        archive = RAW_DIR / "ami_annotations.zip"
        download_file(AMI_URL, archive, "AMI annotations")
        unpack(archive, RAW_DIR / "ami")
        print(
            "  NOTE: AMI audio must be downloaded separately from https://groups.inf.ed.ac.uk/ami/download/"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Cadence training corpora")
    parser.add_argument(
        "--corpora",
        nargs="+",
        default=["candor", "ami"],
        choices=["candor", "ami"],
        help="Which corpora to download (default: all)",
    )
    args = parser.parse_args()
    main(args.corpora)

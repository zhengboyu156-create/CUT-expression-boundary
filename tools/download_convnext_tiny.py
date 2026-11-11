"""Download the official torchvision ImageNet-1K ConvNeXt-Tiny weights.

Usage: python tools/download_convnext_tiny.py [output.pth]
This downloads ImageNet weights, not the PASB HER2-finetuned classifier.
"""
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
SHA256 = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"

def main():
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help")):
        print(__doc__)
        return
    target = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("checkpoints/convnext_tiny-983f1562.pth")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit(f"Output already exists: {target}")
    fd, temp_name = tempfile.mkstemp(prefix=".convnext-", suffix=".download", dir=target.parent)
    os.close(fd)
    try:
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, temp_name)
        digest = hashlib.sha256()
        with open(temp_name, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != SHA256:
            raise RuntimeError("SHA-256 verification failed; downloaded file discarded.")
        os.replace(temp_name, target)
        print(f"Saved and verified: {target}")
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

if __name__ == "__main__":
    main()

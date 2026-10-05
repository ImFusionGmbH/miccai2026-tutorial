"""Build hook that fetches the Torch plugin from the libimfusion-torch wheel on the ImFusion package index.

The published wheel bundles libtorch and CUDA (2.7 GB). The plugin itself only consists of `__init__.py` and the plugin
library, which use the libraries of the `torch` package instead. `torch` must therefore be imported before `imfusion`.
Only these two files are downloaded, with HTTP range requests on the remote wheel, and checked against its RECORD.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

INDEX_URL = "https://pypi.imfusion.com/simple/libimfusion-torch/"
PLATFORM_TAGS = {"linux": "linux_x86_64", "win32": "win_amd64", "darwin": "macosx_15_0_universal2"}
PLUGIN_LIBRARY = re.compile(r"(lib)?TorchPlugin\.(so|dll|dylib)$")


class HttpRangeFile(io.RawIOBase):
    """Read-only file over HTTP that downloads only the byte ranges that are read."""

    def __init__(self, url: str):
        self.url = url
        self.position = 0
        response = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        start = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = start + offset
        return self.position

    def readinto(self, buffer: memoryview) -> int:
        if self.position >= self.size or len(buffer) == 0:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        data = urllib.request.urlopen(request).read()
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)


def wheel_url(version: str) -> str:
    """Return the URL of the published wheel for `version` and the current platform."""
    filename = f"libimfusion_torch-{version}-py3-none-{PLATFORM_TAGS[sys.platform]}.whl"
    index = urllib.request.urlopen(INDEX_URL).read().decode()
    for href in re.findall(r'href="([^"]+)"', index):
        if href.split("#")[0].endswith(filename):
            return urllib.parse.urljoin(INDEX_URL, href.split("#")[0])
    raise RuntimeError(f"{filename} is not available on {INDEX_URL}")


def fetch_plugin(version: str) -> dict[str, bytes]:
    """Download `__init__.py` and the plugin library of the published wheel.

    Returns:
        The file contents by file name.

    Raises:
        RuntimeError: If a file does not match the hash in the RECORD of the wheel.
    """
    package = f"libimfusion_torch-{version}.data/purelib/libimfusion_torch/"
    wheel = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(wheel_url(version)), buffer_size=1 << 20))
    members = [f"{package}__init__.py"]
    members += [name for name in wheel.namelist() if name.startswith(package) and PLUGIN_LIBRARY.search(name)]

    record = wheel.read(f"libimfusion_torch-{version}.dist-info/RECORD").decode()
    hashes = {row[0]: row[1] for row in csv.reader(record.splitlines()) if row}

    files = {}
    for member in members:
        data = wheel.read(member)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if hashes.get(member) != f"sha256={digest}":
            raise RuntimeError(f"{member} does not match the RECORD of the published wheel")
        files[Path(member).name] = data
    return files


class CustomBuildHook(BuildHookInterface):
    download_dir: Path | None = None

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        self.download_dir = Path(tempfile.mkdtemp(prefix="libimfusion-torch-"))
        for name, data in fetch_plugin(self.metadata.version).items():
            (self.download_dir / name).write_bytes(data)
            build_data["force_include"][str(self.download_dir / name)] = f"libimfusion_torch/{name}"
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        if self.download_dir:
            shutil.rmtree(self.download_dir, ignore_errors=True)

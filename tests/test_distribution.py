import json
from pathlib import Path
import subprocess
import sys
import zipfile
import hashlib


def test_clean_reproducible_release_with_valid_manifest():
    root = Path(__file__).resolve().parents[1]
    builder = root/"scripts/build_release.py"
    result = subprocess.run([sys.executable, str(builder)], check=True, capture_output=True, text=True)
    archive = Path(json.loads(result.stdout)["archive"])
    first = archive.read_bytes()
    checksum=archive.with_name(archive.name+".sha256").read_bytes()
    assert b"\r" not in checksum and checksum.endswith(b"\n")
    assert checksum.decode().split()==[hashlib.sha256(first).hexdigest(),archive.name]
    subprocess.run([sys.executable, str(builder)], check=True, capture_output=True)
    assert first == archive.read_bytes()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert "GPA/README.md" in names and "GPA/src/gpa/attention.py" in names
        assert not any(name.endswith((".pt", ".pth", ".safetensors", ".pdf", ".tex", ".pyc")) for name in names)
        assert not any("__pycache__" in name or "egg-info" in name for name in names)
        manifest = json.loads(z.read("GPA/MANIFEST.sha256.json"))
        assert len(manifest) == len(names)-1
        for name, digest in manifest.items():
            assert hashlib.sha256(z.read("GPA/"+name)).hexdigest() == digest

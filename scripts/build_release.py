"""Whitelist source/docs only; produce ZIP + SHA256 manifest, no weights/data."""
from pathlib import Path
import hashlib
import json
import re
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    top = ["pyproject.toml", "README.md", "README.zh-CN.md", "CITATION.cff", ".gitignore"]
    files = [root / name for name in top if (root / name).is_file()]
    for directory in ("src", "tests", "examples", "scripts", "docs", "ds_configs"):
        files.extend(p for p in (root/directory).rglob("*")
                     if p.is_file() and (p.suffix in (".py", ".sh", ".md", ".json")
                                         or (directory == "docs" and p.suffix == ".png"))
                     and "__pycache__" not in p.parts and p.name!="check_optimizer_replay.sh"
                     and not any(part.endswith(".egg-info") for part in p.parts))
    files = sorted(set(files))
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    version = re.search(r'^version\s*=\s*"([^"]+)"', (root/"pyproject.toml").read_text(), re.M).group(1)
    archive = dist / f"gpa-attention-{version}-source.zip"
    # Fixed timestamps and sorted members make repeated builds reproducible.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info = zipfile.ZipInfo("GPA/"+p.relative_to(root).as_posix(), (2026, 10, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
        info = zipfile.ZipInfo("GPA/MANIFEST.sha256.json", (2026, 10, 24, 0, 0, 0))
        z.writestr(info, json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    # Linux sha256sum treats CR as part of the filename. Keep LF on Windows too.
    (dist/(archive.name+".sha256")).write_bytes(f"{checksum}  {archive.name}\n".encode("utf-8"))
    print(json.dumps({"archive": str(archive), "files": len(files), "sha256": checksum}, indent=2))


if __name__ == "__main__":
    main()

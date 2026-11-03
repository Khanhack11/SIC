"""Download pinned public dataset archives; preserve originals and verify extraction."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
from urllib.request import Request, urlopen
import zipfile

from ml.download_dataset import fetch, blob_hash

ROOT = Path("data/raw/dataset_hub_20261103")
HF_REVISION = "527a72eb8f00c95e41698bb09d982c7d4625dce3"
PD_REVISION = "4730a233a555b30ee98e0879c63ad25d82407455"
PD_CLASSIFICATION_REVISION = "5467f6012d78d1c446145d5f582da6096f852ae8"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def checksum(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transfer(url, path, expected_size=None, expected_hash=None, algorithm="sha256"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"Existing archive has unexpected size: {path}")
        if expected_hash and checksum(path, algorithm) != expected_hash:
            raise ValueError(f"Existing archive checksum mismatch: {path}")
        return
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if expected_size is not None and offset == expected_size:
                break
            headers = {"User-Agent": "SIC-dataset-audit/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                resume = offset > 0 and response.status == 206
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an inconsistent byte range")
                total = offset if resume else 0
                reported = total
                with partial.open("ab" if resume else "wb") as stream:
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        stream.write(chunk)
                        total += len(chunk)
                        if total - reported >= 64 * 1024 * 1024:
                            print(f"{path.parent.name}: {total / 1e6:.0f} MB downloaded", flush=True)
                            reported = total
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise ValueError(f"Incomplete download: {path.name}")
            break
        except (OSError, TimeoutError, ValueError) as error:
            if attempt == 5:
                raise
            print(f"Retry {path.name}: {type(error).__name__}: {error}", flush=True)
            time.sleep(min(2 ** attempt, 16))
    if expected_hash and checksum(partial, algorithm) != expected_hash:
        raise ValueError(f"Downloaded checksum mismatch: {path}")
    partial.replace(path)


def safe_member(name):
    source = PurePosixPath(name)
    if source.is_absolute() or ".." in source.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    parts = []
    for part in source.parts:
        if any(c in part for c in '<>:"|?*') or part.endswith((" ", ".")) or part.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]
        }:
            suffix = PurePosixPath(part).suffix
            if not re.fullmatch(r"\.[A-Za-z0-9]{1,8}", suffix):
                suffix = PurePosixPath(part.split("?", 1)[0]).suffix
            if not re.fullmatch(r"\.[A-Za-z0-9]{1,8}", suffix):
                suffix = ""
            part = "__" + hashlib.sha256(part.encode()).hexdigest()[:24] + suffix
        parts.append(part)
    return Path(*parts)


def extract(archive, directory):
    directory.mkdir(parents=True, exist_ok=True)
    mapping = {}
    claimed = set()
    with zipfile.ZipFile(archive) as source:
        files = [item for item in source.infolist() if not item.is_dir()]
        if sum(item.file_size for item in files) > 12 * 1024 ** 3 or len(files) > 200_000:
            raise ValueError("Archive exceeds expected extraction budget")
        for number, item in enumerate(files, 1):
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Archive symlinks are not allowed")
            relative = safe_member(item.filename)
            if len(str((directory / relative).resolve())) > 240:
                relative = relative.with_name("__" + hashlib.sha256(item.filename.encode()).hexdigest()[:24] + relative.suffix)
            destination = directory / relative
            if not destination.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Archive path escaped output directory")
            key = str(relative).casefold()
            if key in claimed:
                relative = relative.with_name("__" + hashlib.sha256(item.filename.encode()).hexdigest()[:24] + relative.suffix)
                destination = directory / relative
                key = str(relative).casefold()
                if key in claimed:
                    raise ValueError(f"Duplicate archive destination: {relative}")
            claimed.add(key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Reading each member to EOF checks ZIP CRC even when resuming an extraction.
            with source.open(item) as incoming, destination.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            if destination.stat().st_size != item.file_size:
                raise ValueError("Extracted size mismatch")
            if str(relative).replace("\\", "/") != item.filename:
                mapping[item.filename] = relative.as_posix()
            if number % 5000 == 0:
                print(f"{archive.parent.name}: extracted {number}/{len(files)} files", flush=True)
    save_json(archive.parent / "extraction_map.json", mapping)
    return {"files": len(files), "uncompressed_bytes": sum(item.file_size for item in files),
            "renamed_windows_paths": len(mapping), "zip_crc_verified": True}


def acquire(root, name, url, metadata, size=None, digest=None, algorithm="sha256"):
    folder = root / name
    save_json(folder / "source.json", metadata | {"download_url": url})
    archive = folder / "original.zip"
    transfer(url, archive, size, digest, algorithm)
    report = extract(archive, folder / "original")
    report.update(archive_bytes=archive.stat().st_size, archive_sha256=checksum(archive),
                  upstream_checksum=f"{algorithm}:{digest}" if digest else None)
    tree_path = root / "sources" / f"{name}_tree.json"
    if tree_path.exists():
        report["git_blob_files_verified"] = verify_git_archive(archive, tree_path)
    save_json(folder / "download_verification.json", report)
    print(f"{name}: complete {report['files']} files", flush=True)
    return report


def verify_git_archive(archive_path, tree_path):
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    if tree.get("truncated"):
        raise ValueError("Incomplete upstream manifest")
    expected = {item["path"]: item["sha"] for item in tree["tree"] if item["type"] == "blob"}
    seen = set()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            relative = "/".join(PurePosixPath(item.filename).parts[1:])
            if relative not in expected or blob_hash(archive.read(item)) != expected[relative]:
                raise ValueError(f"Upstream Git archive mismatch: {relative}")
            seen.add(relative)
    if seen != set(expected):
        raise ValueError("Missing files in upstream Git archive")
    return len(seen)


def copy_plantvillage(root):
    source = Path("data/raw/plantvillage")
    if not (source / "source.json").exists():
        from ml.download_dataset import download as download_village
        download_village(source)
    meta = json.loads((source / "source.json").read_text(encoding="utf-8"))
    destination = root / "plantvillage/original"
    destination.mkdir(parents=True, exist_ok=True)
    for number, entry in enumerate(meta["files"], 1):
        content = (source / entry["path"]).read_bytes()
        if blob_hash(content) != entry["git_blob_sha1"]:
            raise ValueError(f"Existing PlantVillage blob mismatch: {entry['path']}")
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or blob_hash(target.read_bytes()) != entry["git_blob_sha1"]:
            target.write_bytes(content)
        if number % 3000 == 0:
            print(f"PlantVillage: verified/copied {number}/{len(meta['files'])}", flush=True)
    for name in ("source.json", "leaf-map.json"):
        shutil.copyfile(source / name, destination / name)
    save_json(destination.parent / "source.json", {"homepage": meta["repository"], "revision": meta["revision"],
        "task": "classification", "variant": "Selected 11-class RGB subset, max 1500/class; not the entire PlantVillage dataset",
        "reused_verified_local_source": str(source.resolve())})
    report = {"files": len(meta["files"]), "git_blob_hashes_verified": True, "reused_existing_data": True}
    save_json(destination.parent / "download_verification.json", report)
    return report


def download(root):
    root.mkdir(parents=True, exist_ok=True)
    hf = json.loads(fetch(f"https://huggingface.co/api/datasets/uqtwei2/PlantWild/tree/{HF_REVISION}?recursive=true"))
    hf_file = next(item for item in hf if item["path"] == "plantwild_v2.zip")
    zenodo = json.loads(fetch("https://zenodo.org/api/records/17719108"))
    seg_file = next(item for item in zenodo["files"] if item["key"] == "plantseg.zip")
    save_json(root / "sources" / "plantwild_hf_tree.json", hf)
    save_json(root / "sources" / "plantseg_zenodo_record.json", zenodo)
    for name, repository, revision in (
        ("plantdoc_detection", "pratikkayal/PlantDoc-Object-Detection-Dataset", PD_REVISION),
        ("plantdoc_classification", "pratikkayal/PlantDoc-Dataset", PD_CLASSIFICATION_REVISION),
    ):
        tree = json.loads(fetch(f"https://api.github.com/repos/{repository}/git/trees/{revision}?recursive=1"))
        if tree.get("truncated"):
            raise ValueError("Incomplete upstream Git tree")
        save_json(root / "sources" / f"{name}_tree.json", tree)
    tasks = [
        ("plantwild_v2", f"https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/{HF_REVISION}/plantwild_v2.zip?download=true",
         {"homepage": "https://tqwei05.github.io/PlantWild/", "revision": HF_REVISION,
          "license": "CC-BY-NC-ND-4.0", "task": "classification"},
         hf_file["size"], hf_file["lfs"]["oid"], "sha256"),
        ("plantseg", seg_file["links"]["self"],
         {"homepage": "https://github.com/tqwei05/PlantSeg", "record": zenodo["id"],
          "doi": "10.5281/zenodo.17719108", "license": zenodo["metadata"]["license"]["id"],
          "task": "diseased-region segmentation"},
         seg_file["size"], seg_file["checksum"].split(":", 1)[1], "md5"),
        ("plantdoc_detection", f"https://codeload.github.com/pratikkayal/PlantDoc-Object-Detection-Dataset/zip/{PD_REVISION}",
         {"homepage": "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset", "revision": PD_REVISION,
          "license": "CC-BY-4.0", "task": "object detection"}, None, None, "sha256"),
        ("plantdoc_classification", f"https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/{PD_CLASSIFICATION_REVISION}",
         {"homepage": "https://github.com/pratikkayal/PlantDoc-Dataset", "revision": PD_CLASSIFICATION_REVISION,
          "license": "CC-BY-4.0", "task": "classification"}, None, None, "sha256"),
    ]
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(acquire, root, *task): task[0] for task in tasks}
        pending[pool.submit(copy_plantvillage, root)] = "plantvillage"
        for future in as_completed(pending):
            name = pending[future]
            try:
                results[name] = {"status": "complete", **future.result()}
            except Exception as error:
                results[name] = {"status": "failed", "error": str(error)}
                print(f"{name}: FAILED: {error}", flush=True)
            save_json(root / "download_status.json", results)
    if any(item["status"] != "complete" for item in results.values()):
        raise RuntimeError("Some datasets are incomplete; rerun to resume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    download(parser.parse_args().root)

"""Download additional field datasets from Mendeley Data and Zenodo and verify file integrity."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
from urllib.request import Request, urlopen
import zlib

from ml.download_recommended_datasets import ROOT, checksum, extract, save_json, transfer

USER_AGENT = "SIC-dataset-audit/1.0"


def fetch_json(url, timeout=30):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download_verified_file(url, destination, expected_sha256=None, expected_size=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_size is not None and destination.stat().st_size != expected_size:
            destination.unlink()
        elif expected_sha256 and checksum(destination, "sha256") != expected_sha256:
            destination.unlink()
        else:
            return destination.stat().st_size
    temp = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=45) as resp:
                data = resp.read()
            if expected_size is not None and len(data) != expected_size:
                raise ValueError(f"Size mismatch for {destination.name}: {len(data)} != {expected_size}")
            if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
                raise ValueError(f"SHA256 mismatch for {destination.name}")
            temp.write_bytes(data)
            temp.replace(destination)
            return len(data)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
    return 0


def ensure_potato_holeta(root):
    folder = root / "potato_holeta_v2"
    rar_path = folder / "original.rar"
    expected_sha = "6928a86923c15059a838003b1113ac7fbac8c0790e64190b0c37acb4bbd5f5eb"
    if not rar_path.exists():
        transfer(
            "https://data.mendeley.com/public-files/datasets/v4w72bsts5/files/e5edcf03-1734-4f75-87e0-895f08727b19/file_downloaded",
            rar_path,
            expected_hash=expected_sha,
            algorithm="sha256",
        )
    actual_sha = checksum(rar_path, "sha256")
    if actual_sha != expected_sha:
        raise ValueError(f"potato_holeta_v2 checksum mismatch: {actual_sha}")
    out_dir = folder / "original"
    if not (out_dir / "Local").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["C:/Windows/System32/tar.exe", "-xf", str(rar_path), "-C", str(out_dir)], check=True)
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    save_json(folder / "extraction_map.json", {})
    report = {
        "status": "complete",
        "files": len(files),
        "uncompressed_bytes": sum(p.stat().st_size for p in files),
        "renamed_windows_paths": 0,
        "archive_bytes": rar_path.stat().st_size,
        "archive_sha256": actual_sha,
        "upstream_checksum": f"sha256:{expected_sha}",
    }
    save_json(folder / "download_verification.json", report)
    return report


def ensure_potato_iitkgp(root):
    folder = root / "potato_late_blight_iitkgp"
    if (folder / "download_verification.json").exists():
        existing = json.loads((folder / "download_verification.json").read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
    record = fetch_json("https://zenodo.org/api/records/22059910")
    save_json(root / "sources" / "potato_iitkgp_zenodo_record.json", record)
    file_entry = record["files"][0]
    save_json(
        folder / "source.json",
        {
            "homepage": "https://zenodo.org/records/22059910",
            "doi": record.get("doi", "10.5281/zenodo.22059910"),
            "record": record["id"],
            "title": record["metadata"]["title"],
            "license": record["metadata"]["license"]["id"],
            "task": "field potato corpus and stage-wise late blight detector benchmark",
            "note": "Authors deposited 1,561 field potato images and metric CSVs, but omitted per-image bounding-box/class annotations.",
            "download_url": file_entry["links"]["self"],
        },
    )
    archive = folder / "original.zip"
    transfer(file_entry["links"]["self"], archive, file_entry["size"], file_entry["checksum"].split(":", 1)[1], "md5")
    report = extract(archive, folder / "original")
    report.update(
        status="complete",
        archive_bytes=archive.stat().st_size,
        archive_sha256=checksum(archive),
        upstream_checksum=file_entry["checksum"],
    )
    save_json(folder / "download_verification.json", report)
    return report


def ensure_potato_field_indonesia(root):
    """Download Shabrina et al. (2024) Data in Brief 52:109955 (DOI 10.17632/ptz377bwb8.1)."""
    folder = root / "potato_field_indonesia"
    if (folder / "download_verification.json").exists():
        existing = json.loads((folder / "download_verification.json").read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("files", 0) >= 1096:
            return existing
    meta_path = root / "sources" / "potato_field_indonesia_mendeley.json"
    if meta_path.exists():
        dataset = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        dataset = fetch_json("https://data.mendeley.com/public-api/datasets/ptz377bwb8")
        save_json(meta_path, dataset)

    folder_map = {
        "4893a56f-8b14-45df-ba31-921b9bcce728": "Phytophthora",
        "90fa4a30-0306-491c-875b-29958782e9f8": "Healthy",
        "1e57eaed-21f4-4353-9998-32b2ebb8613a": "Fungi",
        "62801f1a-8513-4b37-a440-a09ed7adaa21": "Pest",
        "b3bfa6fe-e82c-4064-be3d-bafe93de4a65": "Bacteria",
        "bca5a215-4a28-43e2-93d7-5fa2ed0a1f8c": "Virus",
        "aebe9126-14db-4cb7-8e76-6286ae40979c": "Nematode",
    }
    # Download 100% of direct-target classes (Healthy=201, Phytophthora=347, Nematode=68)
    # and deterministic 120-image subsets of Fungi, Pest, Bacteria, Virus (all verified by SHA256).
    by_class = {name: [] for name in folder_map.values()}
    for item in dataset.get("files", []):
        fid = item.get("folder_id")
        if fid in folder_map:
            by_class[folder_map[fid]].append(item)
    selected = []
    for cls_name, items in sorted(by_class.items()):
        items_sorted = sorted(items, key=lambda x: x["filename"])
        limit = len(items_sorted) if cls_name in {"Healthy", "Phytophthora", "Nematode"} else 120
        for item in items_sorted[:limit]:
            selected.append((cls_name, item))

    save_json(
        folder / "source.json",
        {
            "homepage": "https://data.mendeley.com/datasets/ptz377bwb8/1",
            "doi": "10.17632/ptz377bwb8.1",
            "paper_doi": "10.1016/j.dib.2023.109955",
            "title": dataset.get("name"),
            "license": "CC-BY-4.0",
            "task": "uncontrolled field potato leaf classification",
            "total_upstream_files": len(dataset.get("files", [])),
            "downloaded_verified_files": len(selected),
            "folder_mapping": folder_map,
        },
    )

    out_dir = folder / "original"
    tasks = []
    for cls_name, item in selected:
        cd = item["content_details"]
        dest = out_dir / cls_name / item["filename"]
        tasks.append((cd["download_url"], dest, cd["sha256_hash"], cd["size"]))

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(download_verified_file, *t) for t in tasks]
        for idx, fut in enumerate(as_completed(futures), 1):
            total_bytes += fut.result()
            if idx % 200 == 0 or idx == len(tasks):
                print(f"potato_field_indonesia: verified {idx}/{len(tasks)} files ({total_bytes / 1e6:.1f} MB)", flush=True)

    save_json(folder / "extraction_map.json", {})
    report = {
        "status": "complete",
        "files": len(tasks),
        "uncompressed_bytes": total_bytes,
        "renamed_windows_paths": 0,
        "upstream_sha256_verified": len(tasks),
    }
    save_json(folder / "download_verification.json", report)
    return report


def ensure_maize_field_bd(root):
    """Download Bangladesh Multi-Format Maize Leaf Dataset (DOI 10.17632/sb4wwjs8bz.1)."""
    folder = root / "maize_field_bd"
    if (folder / "download_verification.json").exists():
        existing = json.loads((folder / "download_verification.json").read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("files", 0) >= 1001:
            return existing
    meta_path = root / "sources" / "maize_field_bd_mendeley.json"
    if meta_path.exists():
        dataset = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        dataset = fetch_json("https://data.mendeley.com/public-api/datasets/sb4wwjs8bz")
        save_json(meta_path, dataset)

    # Use the aspect-ratio-preserving field JPEG folders + the 7 bounding-box CSV files.
    # Avoid the 3 duplicate copies (Detection full-res duplicate, 224x224 squashed org_IMG, and offline aug_0_IMG).
    image_folders = {
        "536322d1-9d30-454e-bf40-de2717023a55": "Healthy_Leaf",
        "b198a9ca-f318-4eb2-b85e-16863cac31b9": "Common_Rust",
        "6294f985-3be1-4f99-92c5-014381037cb3": "Gray_Leaf_Spot",
        "2c470659-4769-44ab-9216-d0535b4b8929": "Corn_Leaf_Blight",
        "fbd2aed7-a4ce-4fcf-93a7-def8e80986d0": "Bacterial_Leaf_Streak",
        "7ebe8528-d4c8-4da9-b3a9-e157ff882f27": "Eyespot",
        "2a74fdf6-9f24-41f0-9fb9-af75f948720e": "Maize_Streak_Disease",
    }
    csv_folders = {
        "6c1b9440-cbac-4f36-8de2-c2d44749a809": "Healthy_Leaf",
        "62b16b09-cd62-453c-9083-4579b4aa4cd0": "Common_Rust",
        "6586cbfa-cb39-4724-a3b3-6ba4bde4ea94": "Gray_Leaf_Spot",
        "26256b90-5c3a-41cf-998e-74a359aa8a0d": "Corn_Leaf_Blight",
        "0adc02f5-69ad-4ad0-9cb3-35d45464eedb": "Bacterial_Leaf_Streak",
        "31bf82c2-e233-452d-8a2d-e7d848a81c68": "Eyespot",
        "f4a09c0f-0263-48b7-b248-70d953e82455": "Maize_Streak_Disease",
    }

    out_dir = folder / "original"
    tasks = []
    skipped_heic = []
    for item in dataset.get("files", []):
        fid = item.get("folder_id")
        fname = item.get("filename", "")
        cd = item.get("content_details", {})
        if fid in csv_folders and fname.lower().endswith(".csv"):
            dest = out_dir / "annotations" / fname
            tasks.append((cd["download_url"], dest, cd["sha256_hash"], cd["size"]))
        elif fid in image_folders:
            if fname.lower().endswith(".heic"):
                skipped_heic.append({"class": image_folders[fid], "filename": fname, "sha256": cd.get("sha256_hash")})
                continue
            dest = out_dir / image_folders[fid] / fname
            tasks.append((cd["download_url"], dest, cd["sha256_hash"], cd["size"]))

    save_json(
        folder / "source.json",
        {
            "homepage": "https://data.mendeley.com/datasets/sb4wwjs8bz/1",
            "doi": "10.17632/sb4wwjs8bz.1",
            "title": dataset.get("name"),
            "license": "CC-BY-4.0",
            "task": "field maize leaf classification and bounding-box detection",
            "total_upstream_files": len(dataset.get("files", [])),
            "deduplicated_raw_Download_files": len(tasks),
            "excluded_heic_files": skipped_heic,
            "audit_note": (
                "Upstream repository contains 4 redundant representations of the same ~1,000 field captures "
                "(Detection full-res, Original full-res, Resized aspect-preserving, and 224x224 squashed + offline aug_0_IMG). "
                "Only one aspect-preserving raw JPEG representation + the 7 bounding-box CSVs are downloaded to prevent cross-representation leakage."
            ),
        },
    )

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(download_verified_file, *t) for t in tasks]
        for idx, fut in enumerate(as_completed(futures), 1):
            total_bytes += fut.result()
            if idx % 200 == 0 or idx == len(tasks):
                print(f"maize_field_bd: verified {idx}/{len(tasks)} files ({total_bytes / 1e6:.1f} MB)", flush=True)

    save_json(folder / "extraction_map.json", {})
    report = {
        "status": "complete",
        "files": len(tasks),
        "uncompressed_bytes": total_bytes,
        "renamed_windows_paths": 0,
        "upstream_sha256_verified": len(tasks),
        "excluded_heic_count": len(skipped_heic),
    }
    save_json(folder / "download_verification.json", report)
    return report


def ensure_tomato_field_pakistan(root):
    """Select camera-named files from Mendeley 3mbnb82mxd.2 via HTTP Range and verify ZIP CRC32."""
    folder = root / "tomato_field_pakistan"
    url = "https://data.mendeley.com/public-files/datasets/3mbnb82mxd/files/4844fda3-b66b-4b2a-bd83-775c0da974ea/file_downloaded"
    total_size = 726463120
    offset = total_size - 2 * 1024 * 1024
    req = Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={offset}-{total_size - 1}"})
    with urlopen(req, timeout=30) as resp:
        tail = resp.read()

    pos = 0
    central_entries = []
    while True:
        idx = tail.find(b"PK\x01\x02", pos)
        if idx == -1 or idx + 46 > len(tail):
            break
        method, _, _, crc32, comp_sz, uncomp_sz, fn_len, ex_len, cm_len, _, _, _, lhoff = struct.unpack_from(
            "<HHHIIIHHHHHII", tail, idx + 10
        )
        fn = tail[idx + 46 : idx + 46 + fn_len].decode("utf-8", errors="ignore")
        central_entries.append(
            {
                "filename": fn,
                "method": method,
                "crc32": crc32,
                "compressed_size": comp_sz,
                "uncompressed_size": uncomp_sz,
                "local_header_offset": lhoff,
            }
        )
        pos = idx + 46 + fn_len + ex_len + cm_len

    save_json(root / "sources" / "tomato_field_pakistan_zip_index.json", central_entries)

    # Filter only Dataset (raw) camera/smartphone captures (IMG_* or _storage_emulated_* or 172345*)
    # to avoid re-downloading the 6,370 pre-split offline augmentations and web/PlantDoc copies.
    raw_entries = [e for e in central_entries if "Tomato Dataset/Dataset (raw)/" in e["filename"] and not e["filename"].endswith("/")]
    smartphone_entries = [
        e
        for e in raw_entries
        if any(tag in e["filename"].split("/")[-1] for tag in ("IMG_", "_storage_emulated_", "172345"))
    ]

    save_json(
        folder / "source.json",
        {
            "homepage": "https://data.mendeley.com/datasets/3mbnb82mxd/2",
            "doi": "10.17632/3mbnb82mxd.2",
            "title": "Tomato Leaf Disease Classification Dataset in Pakistan",
            "license": "CC-BY-4.0",
            "task": "field tomato leaf classification",
            "total_zip_entries": len(central_entries),
            "raw_folder_entries": len(raw_entries),
            "extracted_smartphone_entries": len(smartphone_entries),
            "audit_note": (
                f"Selected {len(smartphone_entries)} files by camera-name patterns from the raw folder; "
                "excluded offline augmentation folders. Filenames do not independently prove capture provenance. "
                "ZIP CRC32 verifies file integrity, not disease labels."
            ),
        },
    )

    out_dir = folder / "original"

    def fetch_zip_member(entry):
        cls_folder = entry["filename"].split("/")[2]
        base_name = entry["filename"].split("/")[-1]
        dest = out_dir / cls_folder / base_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == entry["uncompressed_size"]:
            if (zlib.crc32(dest.read_bytes()) & 0xFFFFFFFF) == entry["crc32"]:
                return entry["uncompressed_size"]
        start = entry["local_header_offset"]
        end = start + 30 + 512 + entry["compressed_size"]
        r = Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
        with urlopen(r, timeout=45) as response:
            chunk = response.read()
        if chunk[:4] != b"PK\x03\x04":
            raise ValueError("Invalid local ZIP header")
        fn_len, ex_len = struct.unpack_from("<HH", chunk, 26)
        data_start = 30 + fn_len + ex_len
        comp_data = chunk[data_start : data_start + entry["compressed_size"]]
        if entry["method"] == 0:
            raw_bytes = comp_data
        elif entry["method"] == 8:
            raw_bytes = zlib.decompress(comp_data, -zlib.MAX_WBITS)
        else:
            raise ValueError(f"Unsupported ZIP compression method: {entry['method']}")
        if len(raw_bytes) != entry["uncompressed_size"]:
            raise ValueError("Uncompressed size mismatch")
        if (zlib.crc32(raw_bytes) & 0xFFFFFFFF) != entry["crc32"]:
            raise ValueError("ZIP CRC32 mismatch")
        dest.write_bytes(raw_bytes)
        return len(raw_bytes)

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(fetch_zip_member, e) for e in smartphone_entries]
        for idx, fut in enumerate(as_completed(futures), 1):
            total_bytes += fut.result()
            if idx % 40 == 0 or idx == len(smartphone_entries):
                print(
                    f"tomato_field_pakistan: verified {idx}/{len(smartphone_entries)} smartphone files ({total_bytes / 1e6:.1f} MB)",
                    flush=True,
                )

    save_json(folder / "extraction_map.json", {})
    report = {
        "status": "complete",
        "files": len(smartphone_entries),
        "uncompressed_bytes": total_bytes,
        "renamed_windows_paths": 0,
        "zip_crc_verified": True,
    }
    save_json(folder / "download_verification.json", report)
    return report


def run_all(root=ROOT):
    status_path = root / "download_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    for name, fn in (
        ("potato_holeta_v2", ensure_potato_holeta),
        ("potato_late_blight_iitkgp", ensure_potato_iitkgp),
        ("potato_field_indonesia", ensure_potato_field_indonesia),
        ("maize_field_bd", ensure_maize_field_bd),
        ("tomato_field_pakistan", ensure_tomato_field_pakistan),
    ):
        print(f"=== Processing {name} ===", flush=True)
        status[name] = fn(root)
        save_json(status_path, status)
    print("All additional datasets verified and ready.", flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run_all(parser.parse_args().root)

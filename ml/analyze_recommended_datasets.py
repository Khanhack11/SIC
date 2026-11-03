"""Inventory the downloaded sources without changing training or evaluation splits."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import struct
import zipfile

import numpy as np
from PIL import Image, ImageOps

from backend.config import CLASS_NAMES
from ml.download_dataset import blob_hash
from ml.download_recommended_datasets import ROOT, save_json
from ml.labels import PLANTVILLAGE_LABELS
from ml.near_duplicate_audit import HashIndex

# Exact disease names are candidates, not expert validation of individual images.
DIRECT_MAPPING = {
    "corn gray leaf spot": "Corn_Gray_leaf_spot",
    "corn northern leaf blight": "Corn_Northern_Leaf_Blight",
    "corn common rust": "Corn_Common_rust",
    "corn healthy": "Corn_healthy",
    "potato early blight": "Potato_Early_blight",
    "potato late blight": "Potato_Late_blight",
    "potato lateblight": "Potato_Late_blight",
    "potato phytophthora": "Potato_Late_blight",
    "potato healthy": "Potato_healthy",
    "tomato early blight": "Tomato_Early_blight",
    "tomato late blight": "Tomato_Late_blight",
    "tomato septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "tomato healthy": "Tomato_healthy",
}


def map_label(label):
    if label in CLASS_NAMES:
        return label, "project_label"
    key = " ".join(label.lower().replace("_", " ").split())
    key = " ".join(word for word in key.split() if word != "leaf")
    # Remove 'leaf' only for lookup; common-rust species and generic blight stay ambiguous.
    normalized = {" ".join(w for w in k.split() if w != "leaf"): v for k, v in DIRECT_MAPPING.items()}
    if key in normalized:
        return normalized[key], "direct_name_candidate"
    if key in {"corn rust", "corn common rust rust"}:
        return "Corn_Common_rust", "review_rust_species"
    if key in {"corn blight", "corn corn blight"}:
        return "", "ambiguous_corn_blight"
    if key == "potato fungi":
        return "", "ambiguous_broad_fungal_label"
    if key == "unlabeled field potato mixed stages":
        return "", "unlabeled_corpus_no_per_image_ground_truth"
    if key in {"potato", "tomato", "corn"}:
        return {"potato": "Potato_healthy", "tomato": "Tomato_healthy", "corn": "Corn_healthy"}[key], "review_generic_healthy_label"
    return "", "outside_current_11_classes"


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def hashes(image):
    basis = np.cos(np.pi * np.arange(8)[:, None] * (2 * np.arange(32)[None, :] + 1) / 64)
    gray = image.convert("L")
    gray.thumbnail((256, 256), Image.Resampling.LANCZOS)
    variants = [gray, *[gray.transpose(operation) for operation in (
        Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270, Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE)]]
    values = []
    for variant in variants:
        pixels = np.asarray(variant.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        coefficients = (basis @ pixels @ basis.T).ravel()
        bits = coefficients > np.median(coefficients[1:])
        bits[0] = False
        small = np.asarray(variant.resize((9, 8), Image.Resampling.LANCZOS))
        values.append((int.from_bytes(np.packbits(bits).tobytes(), "big"),
                       int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")))
    return min(values)


def inspect_image(task):
    record, content = task
    result = dict(record, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    if record.get("expected_git_sha1") and blob_hash(content) != record["expected_git_sha1"]:
        raise ValueError(f"Upstream Git blob mismatch: {record['id']}")
    try:
        with Image.open(io.BytesIO(content)) as source:
            result.update(raw_width=source.width, raw_height=source.height, format=source.format,
                          exif_orientation=source.getexif().get(274, 1), frames=getattr(source, "n_frames", 1))
            if source.width * source.height > 30_000_000:
                raise ValueError("More than 30 megapixels: excluded from decoding budget")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
        result.update(width=image.width, height=image.height)
        result["pixel_sha256"] = hashlib.sha256(struct.pack(">II", *image.size) + image.tobytes()).hexdigest()
        phash, dhash = hashes(image)
        result.update(phash=f"{phash:016x}", dhash=f"{dhash:016x}", status="valid")
        if min(image.size) < 32:
            result["status"] = "too_small"
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        result.update(status="invalid_or_excluded", error=str(error))
    return result


def archive_records(root, name):
    directory = root / name
    renames = json.loads((directory / "extraction_map.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(directory / "original.zip") as archive:
        return [(item.filename, directory / "original" / renames.get(item.filename, item.filename))
                for item in archive.infolist() if not item.is_dir()]


def scan(root, name, records):
    output = root / "analysis" / f"{name}_inventory.jsonl"
    cache = {}
    if output.exists():
        cached = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        cache = {row["id"]: row for row in cached}
        if len(cached) == len(records) and all(r["id"] in cache and all(cache[r["id"]].get(k) == v for k, v in r.items()) for r in records):
            print(f"{name}: reused completed audit of {len(cached)} images", flush=True)
            return cached
    archive_path = root / name / "original.zip"
    archive = zipfile.ZipFile(archive_path) if archive_path.exists() else None
    def tasks():
        for record in records:
            if record.get("expected_git_sha1") and cache.get(record["id"], {}).get("expected_git_sha1") == record["expected_git_sha1"]:
                yield record, None
                continue
            yield record, archive.read(record["archive_member"]) if archive else Path(record["path"]).read_bytes()
    def inspect(task):
        record, content = task
        return cache[record["id"]] | record if content is None else inspect_image(task)
    # Bound queued image bytes: Executor.map eagerly consumes generators on older Python.
    results = []
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.partial")
    iterator = iter(tasks())
    try:
        with temporary.open("w", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                batch = []
                for _ in range(24):
                    try:
                        batch.append(next(iterator))
                    except StopIteration:
                        break
                if not batch:
                    break
                for result in pool.map(inspect, batch):
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    results.append(result)
                if len(results) % 480 == 0:
                    print(f"{name}: inspected {len(results)}/{len(records)} images", flush=True)
        temporary.replace(output)
    finally:
        if archive:
            archive.close()
    print(f"{name}: audit complete ({len(results)} images)", flush=True)
    return results


def record(name, path, label, split="unspecified", member=None, expected=None):
    target, status = map_label(label)
    result = {"id": name + ":" + (member or str(path)), "dataset": name, "path": str(path.resolve()),
              "source_label": label, "target_label": target, "mapping_status": status, "official_split": split}
    if member:
        result["archive_member"] = member
    if expected:
        result["expected_git_sha1"] = expected
    return result


def classification_records(root, name):
    records = []
    if name == "plantvillage":
        source = root / name / "original"
        meta = json.loads((source / "source.json").read_text(encoding="utf-8"))
        for item in meta["files"]:
            folder = PurePosixPath(item["path"]).parts[0]
            records.append(record(name, source / item["path"], PLANTVILLAGE_LABELS[folder],
                                  expected=item["git_blob_sha1"]))
        return records
    expected = {}
    if name == "plantdoc_classification":
        source_meta = json.loads((root / "sources/plantdoc_classification_tree.json").read_text(encoding="utf-8"))
        if source_meta.get("truncated"):
            raise ValueError("Incomplete PlantDoc classification Git tree")
        expected = {item["path"]: item["sha"] for item in source_meta["tree"] if item["type"] == "blob"}
    for member, path in archive_records(root, name):
        parts = PurePosixPath(member).parts
        if name == "plantdoc_classification":
            if len(parts) < 4 or parts[1] not in {"train", "test"}:
                continue  # Documentation illustrations are not dataset samples.
            label, split = parts[2], parts[1]
        else:
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            label, split = parts[1], "unspecified"
        records.append(record(name, path, label, split, member, expected.get("/".join(parts[1:]))))
    return records


def detection_records(root):
    name = "plantdoc_detection"
    files = archive_records(root, name)
    lookup = {"/".join(PurePosixPath(member).parts[1:]): (member, path) for member, path in files}
    tree = json.loads((root / "sources/plantdoc_detection_tree.json").read_text(encoding="utf-8"))
    if tree.get("truncated"):
        raise ValueError("Incomplete detection source manifest")
    expected = {item["path"]: item["sha"] for item in tree["tree"] if item["type"] == "blob"}
    boxes = []
    grouped = defaultdict(list)
    issues = []
    for split in ("train", "test"):
        csv_path = lookup[f"{split}_labels.csv"][1]
        with csv_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = split.upper() + "/" + row["filename"]
                item = dict(row, official_split=split)
                if key not in lookup:
                    item["annotation_status"] = "missing_image"
                    issues.append(item)
                grouped[key].append(item)
                boxes.append(item)
    records = []
    for key, (member, path) in lookup.items():
        if key.split("/")[0] not in {"TRAIN", "TEST"} or PurePosixPath(key).suffix.lower() == ".xml":
            continue
        rows = grouped.get(key, [])
        label = " | ".join(sorted({row["class"] for row in rows}))
        item = record(name, path, label, key.split("/")[0].lower(), member, expected[key])
        item.update(source_labels=sorted({row["class"] for row in rows}), annotation_key=key,
                    annotation_count=len(rows), mapping_status="bounding_boxes_not_image_class")
        records.append(item)
    # Verify source XML/CSV/license files too, not only the image subset.
    with zipfile.ZipFile(root / name / "original.zip") as archive:
        for key, (member, path) in lookup.items():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                if blob_hash(archive.read(member)) != expected[key]:
                    raise ValueError(f"Non-image Git blob mismatch: {key}")
    return records, boxes, issues


def audit_boxes(records, boxes):
    images = {row["annotation_key"]: row for row in records}
    counts = Counter()
    class_counts = Counter()
    for row in boxes:
        counts["total_boxes"] += 1
        class_counts[row["class"]] += 1
        image = images.get(row["official_split"].upper() + "/" + row["filename"])
        errors = []
        try:
            if not image:
                raise ValueError("missing_image")
            width, height = image["raw_width"], image["raw_height"]
            if image["status"] != "valid":
                errors.append("image_invalid_or_too_small")
            x1, y1, x2, y2 = (float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                errors.append("invalid_or_outside_image_box")
            if (int(row["width"]), int(row["height"])) != (width, height):
                errors.append("annotation_image_size_mismatch")
            if image["exif_orientation"] != 1:
                errors.append("requires_exif_coordinate_transform")
            row["image_id"] = image["id"]
            row["image_path"] = image["path"]
            row["area_ratio"] = round((x2-x1)*(y2-y1)/(width*height), 6)
        except (ValueError, KeyError) as error:
            errors.append(str(error))
        row["annotation_status"] = ";".join(errors) if errors else "valid_geometry"
        row["target_label"], row["mapping_status"] = map_label(row["class"])
        counts[row["annotation_status"]] += 1
    return {"counts": dict(counts), "classes": dict(sorted(class_counts.items())),
            "images_without_csv_boxes": sum(not row["annotation_count"] for row in records),
            "note": "Geometry checks do not verify disease identity or complete leaf coverage."}


def segmentation_records(root):
    name = "plantseg"
    files = archive_records(root, name)
    lookup = {member: path for member, path in files}
    with zipfile.ZipFile(root / name / "original.zip") as archive:
        metadata = list(csv.DictReader(io.StringIO(archive.read("plantseg/Metadata.csv").decode("utf-8-sig"))))
    by_name = {row["Name"]: row for row in metadata}
    if len(by_name) != len(metadata):
        raise ValueError("Duplicate PlantSeg metadata names")
    records = []
    for member, path in files:
        parts = PurePosixPath(member).parts
        if len(parts) != 4 or parts[1] != "images":
            continue
        meta = by_name[path.name]
        item = record(name, path, meta["Disease"], parts[2], member)
        mask_member = f"plantseg/annotations/{parts[2]}/{meta['Label file']}"
        item.update(mask_path=str(lookup[mask_member].resolve()) if mask_member in lookup else "",
                    mask_member=mask_member, class_index=int(meta["Index"]), mask_value=int(meta["Index"]) + 1,
                    source_url=meta["URL"], source_license=meta["License"],
                    metadata_split=meta["Split"], metadata_mask_ratio=float(meta["Mask ratio"]))
        records.append(item)
    save_json(root / "analysis/plantseg_categories.json", [
        {"id": index, "name": label, "mask_value": index + 1}
        for index, label in sorted({(int(row["Index"]), row["Disease"]) for row in metadata})])
    return records


def audit_masks(root, records):
    results = []
    counts = Counter()
    ratios = []
    with zipfile.ZipFile(root / "plantseg/original.zip") as archive:
        for number, record in enumerate(records, 1):
            row = {key: record[key] for key in ("id", "mask_path", "class_index", "mask_value", "official_split")}
            errors = []
            try:
                with Image.open(io.BytesIO(archive.read(record["mask_member"]))) as image:
                    mask = np.asarray(image)
                    if image.size != (record["raw_width"], record["raw_height"]):
                        errors.append("image_mask_size_mismatch")
                    if record["exif_orientation"] != 1:
                        errors.append("requires_exif_coordinate_transform")
                    if mask.ndim != 2:
                        raise ValueError("not_single_channel_mask")
                    values, frequency = np.unique(mask, return_counts=True)
                    row["values"] = values.tolist()
                    if set(values.tolist()) - {0, record["mask_value"]}:
                        errors.append("unexpected_class_values")
                    ratio = float(np.count_nonzero(mask)) / mask.size
                    row.update(foreground_ratio=ratio, metadata_ratio=record["metadata_mask_ratio"])
                    ratios.append(ratio)
                    if not ratio:
                        errors.append("empty_mask")
                    if abs(ratio - record["metadata_mask_ratio"]) > 0.0001:
                        errors.append("metadata_ratio_mismatch")
            except (OSError, KeyError, ValueError) as error:
                errors.append(str(error))
            row["status"] = ";".join(errors) if errors else "valid_mask"
            counts[row["status"]] += 1
            results.append(row)
            if number % 1500 == 0:
                print(f"PlantSeg: verified {number}/{len(records)} masks", flush=True)
        coco = {}
        for split in ("train", "val", "test"):
            data = json.loads(archive.read(f"plantseg/annotation_{split}.json"))
            coco[split] = {"images": len(data["images"]), "annotations": len(data["annotations"]),
                           "categories": len(data["categories"])}
    write_csv(root / "analysis/segmentation_masks.csv", results)
    return {"mask_counts": dict(counts), "foreground_ratio_percentiles":
            {str(p): float(np.percentile(ratios, p)) for p in (0, 25, 50, 75, 100)} if ratios else {},
            "coco": coco, "encoding": "PNG background=0, disease value=Metadata.Index+1; COCO category_id=Metadata.Index",
            "note": "Masks cover diseased parts, not the entire leaf. Missing COCO categories must be reconstructed from Metadata.csv."}


def duplicate_audit(root, records):
    valid = [r for r in records if r["status"] == "valid"]
    by_id = {r["id"]: r for r in valid}
    parent = {key: key for key in by_id}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    pairs = []
    pixels = defaultdict(list)
    files = defaultdict(list)
    for row in valid:
        pixels[row["pixel_sha256"]].append(row["id"])
        files[row["sha256"]].append(row["id"])
    for identifiers in pixels.values():
        for current in identifiers[1:]:
            union(identifiers[0], current)
            pairs.append({"left": identifiers[0], "right": current, "method": "exact_oriented_rgb_pixels"})
    index = HashIndex()
    for number, row in enumerate(valid, 1):
        phash, dhash = int(row["phash"], 16), int(row["dhash"], 16)
        for distance, (old_dhash, previous_id) in index.neighbors(phash, 3):
            gradient_distance = (dhash ^ old_dhash).bit_count()
            if gradient_distance <= 4 and by_id[previous_id]["pixel_sha256"] != row["pixel_sha256"]:
                union(row["id"], previous_id)
                pairs.append({"left": previous_id, "right": row["id"], "method": "perceptual_review_candidate",
                              "phash_distance": distance, "dhash_distance": gradient_distance})
        index.add(phash, (dhash, row["id"]))
        if number % 5000 == 0:
            print(f"Duplicate audit: indexed {number}/{len(valid)} images", flush=True)
    previous_splits = defaultdict(set)
    old_records = Path("data/processed/field_mobile_v1/records.json")
    for item in json.loads(old_records.read_text(encoding="utf-8")):
        previous_splits[item["sha256"]].add(item["split"])
    groups = defaultdict(list)
    for row in valid:
        row["duplicate_group"] = "g:" + hashlib.sha256(find(row["id"]).encode()).hexdigest()[:20]
        row["previous_model_splits"] = sorted(previous_splits[row["sha256"]])
        groups[row["duplicate_group"]].append(row)
    summaries = []
    for group, items in groups.items():
        previous = sorted({s for row in items for s in row["previous_model_splits"]})
        official = sorted({row["official_split"] for row in items})
        targets = sorted({row["target_label"] for row in items if row["mapping_status"] in {"project_label", "direct_name_candidate"}})
        protected = bool(set(previous) & {"validation", "test"} or set(official) & {"validation", "val", "test"})
        for row in items:
            row["group_protected_holdout"] = protected
            row["group_previous_train"] = "train" in previous
            row["group_label_conflict"] = len(targets) > 1
        if len(items) > 1:
            summaries.append({"group": group, "images": len(items), "datasets": sorted({r["dataset"] for r in items}),
                              "target_labels": targets, "source_labels": sorted({r["source_label"] for r in items}),
                              "previous_splits": previous, "official_splits": official, "protected_holdout": protected})
    pair_counts = Counter()
    for pair in pairs:
        names = sorted((by_id[pair["left"]]["dataset"], by_id[pair["right"]]["dataset"]))
        pair_counts[" / ".join(names) + ": " + pair["method"]] += 1
    write_csv(root / "analysis/duplicate_pairs.csv", pairs)
    write_csv(root / "analysis/duplicate_groups.csv", summaries)
    return {"exact_file_duplicate_groups": sum(len(v) > 1 for v in files.values()),
            "exact_pixel_duplicate_groups": sum(len(v) > 1 for v in pixels.values()),
            "duplicate_or_similarity_groups": len(summaries), "pairs": len(pairs),
            "label_conflict_groups": sum(len(g["target_labels"]) > 1 for g in summaries),
            "pair_counts": dict(sorted(pair_counts.items())),
            "method": "Dimension-aware EXIF-oriented RGB SHA256; canonical rotations/reflections of <=256px grayscale, pHash <=3 and dHash <=4.",
            "limitations": ["Perceptual pairs need visual review and can miss crops or different views of the same leaf.",
                            "No matches does not establish independence of a field test set.",
                            "Pair counts are graph edges, not independent sample counts; exact pairs use a star per group."]}


def additional_records(root, name):
    base = root / name / "original"
    records = []
    if name == "potato_holeta_v2":
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rel = path.relative_to(base).as_posix()
                records.append(record(name, path, f"Potato {path.parent.name}", "unspecified", rel))
    elif name == "potato_late_blight_iitkgp":
        for member, path in archive_records(root, name):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append(record(name, path, "unlabeled_field_potato_mixed_stages", "unspecified", member))
    elif name == "potato_field_indonesia":
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rel = path.relative_to(base).as_posix()
                records.append(record(name, path, f"Potato {path.parent.name}", "unspecified", rel))
    elif name == "maize_field_bd":
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.parent.name != "annotations" and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rel = path.relative_to(base).as_posix()
                records.append(record(name, path, f"Corn {path.parent.name}", "unspecified", rel))
    elif name == "tomato_field_pakistan":
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rel = path.relative_to(base).as_posix()
                records.append(record(name, path, path.parent.name, "unspecified", rel))
    return records


def audit_maize_bd_boxes(root, records):
    ann_dir = root / "maize_field_bd/original/annotations"
    if not ann_dir.exists():
        return {}
    by_name = {Path(r["path"]).name: r for r in records if r["dataset"] == "maize_field_bd"}
    csv_summary = {}
    all_boxes = []
    status_counts = Counter()
    for csv_path in sorted(ann_dir.glob("*.csv")):
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        csv_summary[csv_path.name] = len(rows)
        for row in rows:
            fname = row.get("image_name", "")
            img = by_name.get(fname)
            errors = []
            try:
                x = float(row["bbox_x"])
                y = float(row["bbox_y"])
                w = float(row["bbox_width"])
                h = float(row["bbox_height"])
                iw = float(row["image_width"])
                ih = float(row["image_height"])
                if not (0 <= x and 0 <= y and w > 0 and h > 0 and x + w <= iw + 1 and y + h <= ih + 1):
                    errors.append("invalid_or_outside_image_box")
                if not img:
                    errors.append("unmatched_resized_filename_or_heic")
            except (KeyError, ValueError) as err:
                errors.append(str(err))
            status = ";".join(errors) if errors else "valid_geometry"
            status_counts[status] += 1
            all_boxes.append(dict(row, csv_file=csv_path.name, status=status))
    write_csv(root / "analysis/maize_field_bd_boxes.csv", all_boxes)
    return {
        "csv_files": csv_summary,
        "total_boxes": len(all_boxes),
        "status_counts": dict(status_counts),
        "note": (
            "Common Rust.csv has 0 bounding-box rows (header only) in upstream Mendeley v1; "
            "other 6 CSVs annotate original 3000x4000 full-res coordinates while resized folders are ~1080x1916."
        ),
    }


def multi_source_candidates(root, records):
    new_sources = {"plantwild_v2", "potato_holeta_v2", "potato_field_indonesia", "maize_field_bd", "tomato_field_pakistan"}
    selected = []
    seen = set()
    by_source = defaultdict(Counter)
    for row in sorted(records, key=lambda r: r["id"]):
        if row["dataset"] not in new_sources or not row["target_label"]:
            continue
        status = "needs_visual_label_review"
        if row["mapping_status"] not in {"direct_name_candidate", "project_label"}:
            status = row["mapping_status"]
        elif row["status"] != "valid" or min(row.get("width", 0), row.get("height", 0)) < 64:
            status = "review_image_quality"
        elif row.get("group_label_conflict"):
            status = "review_conflicting_labels"
        elif row.get("group_protected_holdout"):
            status = "exclude_overlap_with_protected_holdout"
        elif row.get("group_previous_train"):
            status = "exclude_overlap_with_existing_train"
        elif row["duplicate_group"] in seen:
            status = "exclude_duplicate_within_new_sources"
        else:
            seen.add(row["duplicate_group"])
            status = "holdout_safe_candidate"
        selected.append(
            {
                key: row.get(key)
                for key in (
                    "id",
                    "dataset",
                    "path",
                    "source_label",
                    "target_label",
                    "mapping_status",
                    "status",
                    "sha256",
                    "duplicate_group",
                    "group_protected_holdout",
                    "group_previous_train",
                )
            }
            | {"selection_status": status}
        )
        by_source[row["dataset"]][status] += 1
    write_csv(root / "analysis/multi_source_candidates.csv", selected)
    return {k: dict(v) for k, v in sorted(by_source.items())}


def candidates(root, records):
    selected = []
    seen = set()
    for row in sorted(records, key=lambda r: r["id"]):
        if row["dataset"] != "plantwild_v2" or not row["target_label"]:
            continue
        status = "needs_visual_label_review"
        if row["mapping_status"] not in {"direct_name_candidate", "project_label"}:
            status = row["mapping_status"]
        elif row["status"] != "valid" or min(row.get("width", 0), row.get("height", 0)) < 64:
            status = "review_image_quality"
        elif row.get("group_label_conflict"):
            status = "review_conflicting_labels"
        elif row.get("group_protected_holdout"):
            status = "exclude_overlap_with_protected_holdout"
        elif row.get("group_previous_train"):
            status = "exclude_overlap_with_existing_train"
        elif row["duplicate_group"] in seen:
            status = "exclude_duplicate_within_new_source"
        else:
            seen.add(row["duplicate_group"])
        selected.append({key: row.get(key) for key in ("id", "path", "source_label", "target_label", "mapping_status",
                        "status", "sha256", "duplicate_group", "group_protected_holdout", "group_previous_train")}
                        | {"selection_status": status, "split_assigned": False})
    write_csv(root / "analysis/classification_candidates.csv", selected)
    return dict(Counter(row["selection_status"] for row in selected))


def analyze(root):
    records = []
    for name in ("plantwild_v2", "plantdoc_classification", "plantdoc_detection", "plantvillage", "plantseg"):
        if name == "plantdoc_detection":
            source, boxes, _ = detection_records(root)
        elif name == "plantseg":
            source = segmentation_records(root)
        else:
            source = classification_records(root, name)
        records.extend(scan(root, name, source))
    for extra in ("potato_holeta_v2", "potato_late_blight_iitkgp", "potato_field_indonesia", "maize_field_bd", "tomato_field_pakistan"):
        if (root / extra / "original").exists():
            source = additional_records(root, extra)
            if source:
                records.extend(scan(root, extra, source))
    detector = audit_boxes([r for r in records if r["dataset"] == "plantdoc_detection"], boxes)
    write_csv(root / "analysis/detection_boxes.csv", boxes)
    maize_bd_boxes = audit_maize_bd_boxes(root, records)
    segmentation = audit_masks(root, [r for r in records if r["dataset"] == "plantseg"])
    duplicates = duplicate_audit(root, records)
    candidate_counts = candidates(root, records)
    multi_source_counts = multi_source_candidates(root, records)
    class_counts = Counter((r["dataset"], r["source_label"], r["target_label"], r["mapping_status"]) for r in records)
    write_csv(root / "analysis/class_counts.csv", [dict(dataset=dataset, source_label=label, target_label=target,
               mapping_status=status, images=count) for (dataset, label, target, status), count in sorted(class_counts.items())])
    write_csv(root / "analysis/image_inventory.csv", records)
    summary = {"datasets": {}, "detection": detector, "maize_field_bd_detection": maize_bd_boxes, "segmentation": segmentation,
               "duplicates": duplicates, "plantwild_candidate_counts": candidate_counts,
               "multi_source_candidate_counts": multi_source_counts,
               "training_started": False, "existing_splits_modified": False, "serving_model_changed": False}
    for name in sorted({r["dataset"] for r in records}):
        subset = [r for r in records if r["dataset"] == name]
        valid = [r for r in subset if r["status"] == "valid"]
        summary["datasets"][name] = {"images": len(subset), "source_labels": len({r["source_label"] for r in subset}),
            "status": dict(Counter(r["status"] for r in subset)), "bytes": sum(r["bytes"] for r in subset),
            "dimensions": {"min_width": min(r["width"] for r in valid), "max_width": max(r["width"] for r in valid),
                           "min_height": min(r["height"] for r in valid), "max_height": max(r["height"] for r in valid)},
            "official_splits": dict(Counter(r["official_split"] for r in subset)),
            "candidate_target_labels": sorted({r["target_label"] for r in subset if r["target_label"]}),
            "direct_name_target_counts": dict(Counter(r["target_label"] for r in subset if r["mapping_status"] in {"direct_name_candidate", "project_label"})),
            "hash_verified_image_blobs": sum(bool(r.get("expected_git_sha1")) for r in subset)}
    save_json(root / "analysis/summary.json", summary)
    print(json.dumps({"datasets": {k: v["images"] for k, v in summary["datasets"].items()},
                      "plantwild_candidates": candidate_counts, "duplicate_pairs": duplicates["pairs"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    analyze(parser.parse_args().root)

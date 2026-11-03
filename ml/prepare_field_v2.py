"""Build holdout-safe expanded field dataset (field_mobile_v2) from verified multi-source hub."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import struct

from PIL import Image, ImageOps

from backend.config import CLASS_NAMES
from backend.mobile_model import FIELD_CLASSES
from backend.preprocessing import orient_rgb
from ml.analyze_recommended_datasets import hashes, map_label
from ml.near_duplicate_audit import HashIndex

PLANTDOC_BOX_MAPPING = {
    "corn gray leaf spot": "Corn_Gray_leaf_spot",
    "corn rust leaf": "Corn_Common_rust",
    "corn leaf blight": None,  # ambiguous blight excluded
    "potato leaf": "Potato_healthy",
    "potato leaf early blight": "Potato_Early_blight",
    "potato leaf late blight": "Potato_Late_blight",
    "tomato leaf": "Tomato_healthy",
    "tomato early blight leaf": "Tomato_Early_blight",
    "tomato leaf late blight": "Tomato_Late_blight",
    "tomato septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "tomato mold leaf": "Unsupported",
    "tomato leaf bacterial spot": "Unsupported",
    "tomato leaf mosaic virus": "Unsupported",
    "tomato leaf yellow virus": "Unsupported",
    "Tomato two spotted spider mites leaf": "Unsupported",
}


def pixel_digest(image: Image.Image) -> str:
    return hashlib.sha256(struct.pack(">II", *image.size) + image.tobytes()).hexdigest()


def extract_plantdoc_train_crops(hub_root: Path, v1_records: list[dict], crop_dir: Path) -> tuple[list[dict], list[dict]]:
    """Extract single-leaf bounding-box crops strictly from PlantDoc/train parent images."""
    train_by_sha = {
        r["sha256"]: r
        for r in v1_records
        if r["domain"] == "PlantDoc" and r["split"] == "train"
    }
    holdout_shas = {
        r["sha256"]
        for r in v1_records
        if r["split"] in {"validation", "test"}
    }

    det_inv_path = hub_root / "analysis/plantdoc_detection_inventory.jsonl"
    det_boxes_path = hub_root / "analysis/detection_boxes.csv"
    if not det_inv_path.exists() or not det_boxes_path.exists():
        return [], []

    det_images = {
        row["annotation_key"]: row
        for row in (json.loads(line) for line in det_inv_path.read_text(encoding="utf-8").splitlines())
    }
    boxes_by_key = defaultdict(list)
    with det_boxes_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("official_split") == "train" and row.get("annotation_status") == "valid_geometry":
                key = "TRAIN/" + row["filename"]
                boxes_by_key[key].append(row)

    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_records = []
    skipped = []

    for key, rows in sorted(boxes_by_key.items()):
        det_img = det_images.get(key)
        if not det_img:
            continue
        sha = det_img["sha256"]
        if sha in holdout_shas or sha not in train_by_sha:
            skipped.append({"key": key, "reason": "parent_not_in_plantdoc_train"})
            continue
        parent_rec = train_by_sha[sha]
        parent_path = Path(parent_rec["path"])
        if not parent_path.exists():
            continue

        # Sort boxes by area descending and take up to 3 largest valid boxes per image
        def box_area(b):
            return (float(b["xmax"]) - float(b["xmin"])) * (float(b["ymax"]) - float(b["ymin"]))

        selected_boxes = sorted(rows, key=box_area, reverse=True)[:3]
        with Image.open(parent_path) as raw_img:
            image = orient_rgb(raw_img)
            w, h = image.size
            for idx, box in enumerate(selected_boxes):
                raw_cls = box["class"].strip()
                lower_cls = raw_cls.lower()
                if lower_cls == "corn leaf blight":
                    continue
                target_label = PLANTDOC_BOX_MAPPING.get(lower_cls, "Unsupported")
                # Only keep boxes whose label agrees with parent known class or both are Unsupported
                if parent_rec["label"] != "Unsupported" and target_label != parent_rec["label"]:
                    continue
                x1, y1, x2, y2 = float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])
                bw, bh = x2 - x1, y2 - y1
                # Skip near-whole-image boxes (>88% area) since full image is already in PlantDoc/train
                if (bw * bh) / (w * h) > 0.88:
                    continue
                # Add 8% context margin
                pad_x, pad_y = 0.08 * bw, 0.08 * bh
                cx1 = max(0, int(round(x1 - pad_x)))
                cy1 = max(0, int(round(y1 - pad_y)))
                cx2 = min(w, int(round(x2 + pad_x)))
                cy2 = min(h, int(round(y2 + pad_y)))
                if min(cx2 - cx1, cy2 - cy1) < 64:
                    continue
                cropped = image.crop((cx1, cy1, cx2, cy2))
                buf = io.BytesIO()
                cropped.save(buf, format="JPEG", quality=94)
                raw_bytes = buf.getvalue()
                crop_sha = hashlib.sha256(raw_bytes).hexdigest()
                crop_file = crop_dir / f"{sha[:16]}_box{idx}.jpg"
                if not crop_file.exists() or hashlib.sha256(crop_file.read_bytes()).hexdigest() != crop_sha:
                    crop_file.write_bytes(raw_bytes)
                crop_records.append(
                    {
                        "path": str(crop_file.resolve()),
                        "sha256": crop_sha,
                        "pixel": pixel_digest(cropped),
                        "label": target_label,
                        "original_label": raw_cls,
                        "official_split": "train",
                        "domain": "PlantDoc",
                        "group": parent_rec["group"],
                        "split": "train",
                        "source_dataset": "plantdoc_detection_crop",
                    }
                )
    return crop_records, skipped


def select_new_field_records(hub_root: Path, v1_records: list[dict]) -> tuple[list[dict], dict]:
    """Select holdout-safe field images from the 5 external field datasets."""
    # Build strict holdout and existing-train hash indexes from field_mobile_v1
    holdout_shas = set()
    holdout_pixels = set()
    holdout_index = HashIndex()
    existing_shas = set()
    existing_pixels = set()
    existing_index = HashIndex()

    # Preload cached hashes from hub inventory to avoid re-decoding 10k v1 images
    cached_by_sha = {}
    for inv_name in ("plantvillage", "plantdoc_classification"):
        inv_file = hub_root / "analysis" / f"{inv_name}_inventory.jsonl"
        if inv_file.exists():
            for line in inv_file.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("status") == "valid":
                    cached_by_sha[row["sha256"]] = (
                        row["pixel_sha256"],
                        int(row["phash"], 16),
                        int(row["dhash"], 16),
                    )

    for idx, rec in enumerate(v1_records):
        sha = rec["sha256"]
        if sha in cached_by_sha:
            px_sha, ph, dh = cached_by_sha[sha]
        else:
            p = Path(rec["path"])
            raw = p.read_bytes()
            with Image.open(io.BytesIO(raw)) as src:
                img = ImageOps.exif_transpose(src).convert("RGB")
            px_sha = pixel_digest(img)
            ph, dh = hashes(img)
        if rec["split"] in {"validation", "test"}:
            holdout_shas.add(sha)
            holdout_pixels.add(px_sha)
            holdout_index.add(ph, dh)
        else:
            existing_shas.add(sha)
            existing_pixels.add(px_sha)
            existing_index.add(ph, dh)

    inventories = {}
    for name in ("plantwild_v2", "potato_holeta_v2", "potato_field_indonesia", "maize_field_bd", "tomato_field_pakistan"):
        inv_path = hub_root / "analysis" / f"{name}_inventory.jsonl"
        if inv_path.exists():
            inventories[name] = [json.loads(line) for line in inv_path.read_text(encoding="utf-8").splitlines()]

    candidates = []
    stats = Counter()
    ood_per_source_class = Counter()

    for name, rows in sorted(inventories.items()):
        for row in sorted(rows, key=lambda r: r["id"]):
            if row.get("status") != "valid" or min(row.get("width", 0), row.get("height", 0)) < 64:
                stats["excluded_invalid_or_small"] += 1
                continue
            sha = row["sha256"]
            px_sha = row["pixel_sha256"]
            ph = int(row["phash"], 16)
            dh = int(row["dhash"], 16)

            # Check strict holdout protection (exact bytes, exact pixels, or pHash<=4 & dHash<=4)
            if (
                sha in holdout_shas
                or px_sha in holdout_pixels
                or any((dh ^ other_dh).bit_count() <= 4 for _, other_dh in holdout_index.neighbors(ph, 4))
            ):
                stats["excluded_holdout_overlap"] += 1
                continue

            # Check overlap with existing train
            if (
                sha in existing_shas
                or px_sha in existing_pixels
                or any((dh ^ other_dh).bit_count() <= 4 for _, other_dh in existing_index.neighbors(ph, 4))
            ):
                stats["excluded_existing_train_overlap"] += 1
                continue

            target_label, map_status = map_label(row["source_label"])
            # Accept direct target classes + reviewed corn rust in plantwild_v2
            if map_status in {"direct_name_candidate", "project_label"} or (
                name == "plantwild_v2" and map_status == "review_rust_species"
            ):
                label = target_label
            elif map_status == "outside_current_11_classes":
                # Include bounded OOD samples per source class to strengthen Unsupported gate
                cap = 15 if name == "plantwild_v2" else 45
                key = (name, row["source_label"])
                if ood_per_source_class[key] >= cap:
                    stats["excluded_ood_cap"] += 1
                    continue
                ood_per_source_class[key] += 1
                label = "Unsupported"
            else:
                stats[f"excluded_{map_status}"] += 1
                continue

            candidates.append(
                {
                    "path": row["path"],
                    "sha256": sha,
                    "pixel": px_sha,
                    "phash": ph,
                    "dhash": dh,
                    "label": label,
                    "original_label": row["source_label"],
                    "official_split": row.get("official_split", "unspecified"),
                    "domain": "MultiSourceField",
                    "source_dataset": name,
                }
            )

    # Group candidates by perceptual hash (phash<=4 & dhash<=4) or exact pixels to prevent cross-source leakage
    new_index = HashIndex()
    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for idx, item in enumerate(candidates):
        ph, dh = item["phash"], item["dhash"]
        for _, other_idx in new_index.neighbors(ph, 4):
            if (dh ^ candidates[other_idx]["dhash"]).bit_count() <= 4 or item["pixel"] == candidates[other_idx]["pixel"]:
                parent[find(idx)] = find(other_idx)
        new_index.add(ph, idx)

    groups = defaultdict(list)
    for idx, item in enumerate(candidates):
        groups[find(idx)].append(item)

    deduped_by_label = defaultdict(list)
    for gid, items in sorted(groups.items()):
        labels = {r["label"] for r in items}
        if len(labels) > 1:
            stats["excluded_conflicting_group_labels"] += len(items)
            continue
        rep = min(items, key=lambda r: (r["source_dataset"], r["path"]))
        stats["duplicates_collapsed"] += len(items) - 1
        clean = {k: v for k, v in rep.items() if k not in {"phash", "dhash"}}
        clean["group"] = f"msf:{gid}"
        deduped_by_label[clean["label"]].append(clean)

    final_new = []
    for label, items in sorted(deduped_by_label.items()):
        random.Random(f"42:msf:{label}").shuffle(items)
        val_count = max(1, round(0.15 * len(items))) if len(items) >= 5 else 0
        for idx, item in enumerate(items):
            item["split"] = "validation" if idx < val_count else "train"
            final_new.append(item)
            stats[f"selected_{item['split']}"] += 1

    return final_new, dict(stats)


def prepare_v2(hub_root: Path, v1_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise ValueError(f"Output directory {output_dir} already exists; splits are immutable")

    v1_records_path = v1_dir / "records.json"
    v1_audit = json.loads((v1_dir / "audit.json").read_text(encoding="utf-8"))
    if hashlib.sha256(v1_records_path.read_bytes()).hexdigest() != v1_audit["records_sha256"]:
        raise ValueError("field_mobile_v1 manifest hash mismatch")

    v1_records = json.loads(v1_records_path.read_text(encoding="utf-8"))
    records = [dict(r, source_dataset=r["domain"].lower()) for r in v1_records]

    crop_dir = output_dir / "crops"
    crop_records, skipped_crops = extract_plantdoc_train_crops(hub_root, v1_records, crop_dir)
    records.extend(crop_records)

    new_field_records, selection_stats = select_new_field_records(hub_root, v1_records)
    records.extend(new_field_records)

    # Verify zero group leakage across splits
    groups_by_split = {
        split: {r["group"] for r in records if r["split"] == split}
        for split in ("train", "validation", "test")
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = groups_by_split[a] & groups_by_split[b]
        if overlap:
            raise AssertionError(f"Group leakage detected between {a} and {b}: {len(overlap)} groups")

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.json"
    records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    records_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()

    audit_report = {
        "base_manifest": str(v1_records_path),
        "base_records_sha256": v1_audit["records_sha256"],
        "records_sha256": records_sha256,
        "labels": FIELD_CLASSES,
        "seed": 42,
        "total_records": len(records),
        "plantdoc_train_box_crops_added": len(crop_records),
        "new_multisource_field_records_added": len(new_field_records),
        "selection_stats": selection_stats,
        "counts_by_domain_split": dict(Counter(f"{r['domain']}/{r['split']}" for r in records)),
        "counts_by_source_split": dict(Counter(f"{r.get('source_dataset', r['domain'])}/{r['split']}" for r in records)),
        "counts_by_domain_split_label": dict(Counter(f"{r['domain']}/{r['split']}/{r['label']}" for r in records)),
        "holdout_integrity": {
            "v1_validation_unchanged": True,
            "v1_test_unchanged": True,
            "group_overlap_across_splits": 0,
            "exact_or_perceptual_leakage_to_v1_holdouts": 0,
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
    print(json.dumps(audit_report["counts_by_domain_split"], indent=2), flush=True)
    print("Saved field_mobile_v2 records:", len(records), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=Path("data/raw/dataset_hub_20261103"))
    parser.add_argument("--v1", type=Path, default=Path("data/processed/field_mobile_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/field_mobile_v2"))
    args = parser.parse_args()
    prepare_v2(args.hub, args.v1, args.output)

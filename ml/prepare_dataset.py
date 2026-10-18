"""Audit PlantVillage RGB files and create reproducible, leaf-disjoint manifests."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import warnings

from PIL import Image, ImageOps

from backend.config import CLASS_NAMES
from ml.labels import canonical_label, CLASS_TO_INDEX

FIELDS = ["path", "label", "class_index", "group", "sha256", "pixel_sha256", "width", "height"]


def leaf_group(filename, folder, leaf_map):
    identifier = Path(filename).stem.replace("_final_masked", "").split("___")[-1]
    identifier = identifier.split("copy")[0].strip().lower()
    suggestions = leaf_map.get(identifier, [])
    if len(suggestions) == 1:
        return suggestions[0], True
    matches = [value for value in suggestions if value.startswith(folder + ":::")]
    if len(matches) == 1:
        return matches[0], True
    return f"fallback:{canonical_label(folder)}:{identifier}", False


def stratified_groups(records, seed):
    result = {name: [] for name in ("train", "validation", "test")}
    for label in CLASS_NAMES:
        groups = defaultdict(list)
        for row in records:
            if row["label"] == label:
                groups[row["group"]].append(row)
        if len(groups) < 3:
            raise ValueError(f"{label}: need at least 3 distinct leaf groups; found {len(groups)}")
        items = sorted(groups.items())
        random.Random(f"{seed}:{label}").shuffle(items)
        items.sort(key=lambda pair: -len(pair[1]))
        total = sum(len(rows) for _, rows in items)
        targets = dict(zip(result, (total * .7, total * .15, total * .15)))
        counts = dict.fromkeys(result, 0)
        for index, (_, rows) in enumerate(items):
            empty = [name for name in counts if counts[name] == 0]
            candidates = empty if len(items) - index == len(empty) else list(counts)
            chosen = max(candidates, key=lambda name: targets[name] - counts[name])
            result[chosen].extend(rows)
            counts[chosen] += len(rows)
    for rows in result.values():
        rows.sort(key=lambda row: row["path"])
    return result


def prepare(source, output, seed=42, leaf_map_path=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f"Dataset directory does not exist: {source}")
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the raw dataset directory.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; use a new output directory to preserve prior manifests.")
    leaf_map_path = Path(leaf_map_path) if leaf_map_path else source / "leaf-map.json"
    if not leaf_map_path.exists():
        raise ValueError("A leaf-map.json is required to prevent same-leaf train/test leakage.")
    leaf_map_bytes = leaf_map_path.read_bytes()
    leaf_map = json.loads(leaf_map_bytes)
    source_metadata = source / "source.json"
    provenance = json.loads(source_metadata.read_text(encoding="utf-8")) if source_metadata.exists() else {}
    if provenance.get("failures"):
        raise ValueError("Dataset download is incomplete; rerun the downloader first.")
    selected_paths = {item["path"] for item in provenance.get("files", [])}
    excluded, records = [], []
    matched, fallback = 0, 0
    paths = sorted(source.rglob("*"))
    print(f"Auditing {len(paths)} paths in {source}", flush=True)
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        relative = path.relative_to(source).as_posix()
        if selected_paths and relative not in selected_paths:
            continue
        label = canonical_label(path.parent.name)
        if label is None:
            excluded.append({"path": relative, "reason": "unsupported_class"})
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    if image.format not in {"JPEG", "PNG"}:
                        raise ValueError("content is not JPEG/PNG")
                    if min(image.size) < 32 or image.width * image.height > 20_000_000:
                        raise ValueError("resolution outside 32px minimum / 20 megapixel maximum")
                    image.verify()
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    width, height = image.size
            group, found = leaf_group(path.name, path.parent.name, leaf_map)
            matched += int(found)
            fallback += int(not found)
            records.append(dict(path=relative, label=label, class_index=CLASS_TO_INDEX[label],
                                group=group, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                pixel_sha256=pixel_hash, width=width, height=height))
            if len(records) % 1000 == 0:
                print(f"Audited {len(records)} readable images; excluded={len(excluded)}", flush=True)
        except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            excluded.append({"path": relative, "reason": "invalid_image", "detail": str(exc)})

    # Merge leaf identities linked by identical RGB content before deduplication.
    parents = {row["group"]: row["group"] for row in records}

    def root(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    hashes = {}
    for row in records:
        previous = hashes.setdefault(row["pixel_sha256"], row)
        left, right = root(previous["group"]), root(row["group"])
        parents[max(left, right)] = min(left, right)
    labels_by_group = defaultdict(set)
    for row in records:
        row["group"] = root(row["group"])
        labels_by_group[row["group"]].add(row["label"])
    seen, curated = set(), []
    for row in records:
        if len(labels_by_group[row["group"]]) > 1:
            excluded.append({"path": row["path"], "reason": "conflicting_labels_same_leaf_or_pixels"})
        elif row["pixel_sha256"] in seen:
            excluded.append({"path": row["path"], "reason": "duplicate_rgb_pixels"})
        else:
            seen.add(row["pixel_sha256"])
            curated.append(row)
    splits = stratified_groups(curated, seed)
    for field in ("group", "sha256", "pixel_sha256"):
        sets = [{row[field] for row in rows} for rows in splits.values()]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError(f"Split leakage detected for {field}")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    labels = {"class_names": CLASS_NAMES, "class_to_index": CLASS_TO_INDEX}
    (output / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    manifest_hashes = {name: hashlib.sha256((output / f"{name}.csv").read_bytes()).hexdigest()
                       for name in splits}
    report = {
        "source_root": str(source), "seed": seed, "target_ratios": {"train": .7, "validation": .15, "test": .15},
        "split_policy": "stratified by class; all images of a leaf remain in one split",
        "leaf_map_sha256": hashlib.sha256(leaf_map_bytes).hexdigest(),
        "source": {key: value for key, value in provenance.items() if key != "files"},
        "input_valid_count": len(records), "curated_count": len(curated),
        "leaf_map_matches": matched, "leaf_map_fallbacks": fallback,
        "excluded_count": len(excluded), "excluded_reasons": dict(Counter(row["reason"] for row in excluded)),
        "splits": {name: {"count": len(rows), "ratio": len(rows) / len(curated),
                          "per_class": dict(Counter(row["label"] for row in rows))} for name, rows in splits.items()},
        "class_names": CLASS_NAMES, "manifest_sha256": manifest_hashes,
        "leakage_checks": {"leaf_groups_disjoint": True, "file_hashes_disjoint": True, "pixel_hashes_disjoint": True},
        "limitations": ["Metadata cannot prove biological labels are correct.",
                        "Exact RGB deduplication does not detect every visually similar image.",
                        "Fallback leaf identifiers do not guarantee physical leaf identity.",
                        "Controlled-background PlantVillage results do not establish field performance."],
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "excluded.json").write_text(json.dumps(excluded, indent=2), encoding="utf-8")
    print(json.dumps({"curated": len(curated), "excluded": len(excluded),
                      "splits": {name: len(rows) for name, rows in splits.items()}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/plantvillage"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--leaf-map", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed, args.leaf_map)


if __name__ == "__main__":
    main()

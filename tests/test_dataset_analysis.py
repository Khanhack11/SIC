"""Verify label ambiguity, duplicate hashes and annotation checks with fixtures."""

import io
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ml.analyze_recommended_datasets import (
    audit_boxes,
    candidates,
    hashes,
    inspect_image,
    map_label,
    multi_source_candidates,
)


class DatasetAnalysisTests(unittest.TestCase):
    def test_ambiguous_labels_are_not_direct_training_candidates(self):
        self.assertEqual(map_label("corn northern leaf blight"), ("Corn_Northern_Leaf_Blight", "direct_name_candidate"))
        self.assertEqual(map_label("Corn leaf blight"), ("", "ambiguous_corn_blight"))
        self.assertEqual(map_label("Corn Corn_Leaf_Blight"), ("", "ambiguous_corn_blight"))
        self.assertEqual(map_label("Corn rust leaf"), ("Corn_Common_rust", "review_rust_species"))
        self.assertEqual(map_label("Potato leaf")[1], "review_generic_healthy_label")
        self.assertEqual(map_label("Potato Phytophthora"), ("Potato_Late_blight", "direct_name_candidate"))
        self.assertEqual(map_label("Potato LateBlight"), ("Potato_Late_blight", "direct_name_candidate"))
        self.assertEqual(map_label("Potato Fungi"), ("", "ambiguous_broad_fungal_label"))
        self.assertEqual(
            map_label("unlabeled_field_potato_mixed_stages"),
            ("", "unlabeled_corpus_no_per_image_ground_truth"),
        )
        self.assertEqual(map_label("tomato leaf mold"), ("", "outside_current_11_classes"))

    def test_hashes_group_rotations_and_pixel_hash_includes_shape(self):
        image = Image.fromarray(np.random.default_rng(12).integers(0, 256, (80, 96, 3), dtype=np.uint8))
        self.assertEqual(hashes(image), hashes(image.transpose(Image.Transpose.ROTATE_90)))
        results = []
        for size in ((32, 64), (64, 32)):
            buffer = io.BytesIO()
            Image.new("RGB", size, "green").save(buffer, "PNG")
            results.append(inspect_image(({"id": str(size)}, buffer.getvalue())))
        self.assertNotEqual(results[0]["pixel_sha256"], results[1]["pixel_sha256"])

    def test_corrupt_images_and_source_hash_mismatch_are_detected(self):
        result = inspect_image(({"id": "broken"}, b"broken image"))
        self.assertEqual(result["status"], "invalid_or_excluded")
        with self.assertRaisesRegex(ValueError, "Git blob mismatch"):
            inspect_image(({"id": "modified", "expected_git_sha1": "0" * 40}, b"changed"))

    def test_boxes_outside_image_and_dimension_mismatch_are_flagged(self):
        images = [{"id": "example", "annotation_key": "TRAIN/leaf.jpg", "path": "leaf.jpg",
                   "raw_width": 100, "raw_height": 80, "exif_orientation": 1, "annotation_count": 2,
                   "status": "valid"}]
        common = {"filename": "leaf.jpg", "official_split": "train", "class": "Tomato early blight leaf",
                  "width": "100", "height": "80", "xmin": "0", "ymin": "0", "xmax": "80", "ymax": "70"}
        boxes = [dict(common), dict(common, xmax="101", width="90")]
        summary = audit_boxes(images, boxes)
        self.assertEqual(summary["counts"]["valid_geometry"], 1)
        self.assertIn("invalid_or_outside_image_box", boxes[1]["annotation_status"])
        self.assertIn("annotation_image_size_mismatch", boxes[1]["annotation_status"])

    def test_training_candidates_exclude_protected_holdout_and_new_duplicates(self):
        common = dict(dataset="plantwild_v2", target_label="Tomato_Early_blight", mapping_status="direct_name_candidate",
                      status="valid", width=224, height=224, duplicate_group="new", group_protected_holdout=False,
                      group_previous_train=False, group_label_conflict=False)
        rows = [dict(common, id="a", duplicate_group="protected", group_protected_holdout=True),
                dict(common, id="b"), dict(common, id="c")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counts = candidates(root, rows)
            self.assertEqual(counts["exclude_overlap_with_protected_holdout"], 1)
            self.assertEqual(counts["needs_visual_label_review"], 1)
            self.assertEqual(counts["exclude_duplicate_within_new_source"], 1)
            with (root / "analysis/classification_candidates.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual({r["split_assigned"] for r in csv.DictReader(stream)}, {"False"})


if __name__ == "__main__":
    unittest.main()
